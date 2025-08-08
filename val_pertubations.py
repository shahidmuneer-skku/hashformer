import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import datetime
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from dataloaders.datasets import HashFormerDataset
from dataloaders.faceswapVideoDataset import FaceswapVideoDataset
from models.model import HashFormer
from models.decoders import VisualDecoder16x16
from util import seed_worker, set_seed, compute_eer
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve,roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import silhouette_score
from torch.cuda.amp import autocast, GradScaler
# torch.autograd.set_detect_anomaly(True)
import torchvision
import random
from datetime import timedelta
from transformers import get_linear_schedule_with_warmup
import cv2

# optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=total_steps)

# torch.set_printoptions(threshold=10_0000)

def _to_tchw(arr4d):
    """
    Accepts 4‑D array in any of the common orders and returns (T,C,H,W).
    """
    if arr4d.ndim != 4:
        raise ValueError("Need a 4‑D array/tensor")

    s = arr4d.shape
    # try to find the axis with 1 or 3 → that *must* be channels
    chan_axes = [ax for ax, dim in enumerate(s) if dim in (1, 3)]
    if not chan_axes:
        raise ValueError("Could not identify channel axis (no dim of size 1 or 3)")
    c_ax = chan_axes[0]

    # heuristically pick a *frame* axis: smallest dim that is NOT channel and ≤ 500
    frame_axes = [ax for ax in range(4) if ax != c_ax and s[ax] <= 500]
    if not frame_axes:
        frame_axes = [ax for ax in range(4) if ax != c_ax]   # fallback: just pick first
    t_ax = frame_axes[0]

    # remaining two axes are height / width
    rem_axes = [ax for ax in range(4) if ax not in (t_ax, c_ax)]
    h_ax, w_ax = rem_axes

    arr_tchw = np.transpose(arr4d, (t_ax, c_ax, h_ax, w_ax))
    return arr_tchw  # (T,C,H,W)

def _save_with_cv2(video_4d, path="recon.mp4", fps=25):
    """
    video_4d : torch.Tensor | np.ndarray  (any of the 4 common layouts)
               float∈[-1,1]∪[0,1]  or uint8
    """
    # ---- Torch → NumPy ----------------------------------------------------
    if isinstance(video_4d, torch.Tensor):
        arr = video_4d.detach().cpu().numpy()
    else:
        arr = video_4d

    # ---- Re‑order to (T,C,H,W) -------------------------------------------
    arr = _to_tchw(arr)                         # (T,C,H,W)
    T, C, H, W = arr.shape

    # ---- Float → uint8 ----------------------------------------------------
    if issubclass(arr.dtype.type, np.floating):
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0             # [-1,1] → [0,1]
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # ---- (T,C,H,W) → (T,H,W,C)  for OpenCV -------------------------------
    arr = np.transpose(arr, (0, 2, 3, 1))       # (T,H,W,C)

    # ---- Write with OpenCV -----------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for frame in arr:
        if C == 1:                              # gray → 3‑channel
            frame = np.repeat(frame, 3, axis=2)
        frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])  # RGB→BGR, positive strides
        writer.write(frame_bgr)

    writer.release()
    print(f"✅ Saved reconstructed clip ➜ {path}")


def save_reconstructed_video(tensor_4d, path="recon.mp4", fps=25):
    """
    tensor_4d: (T, C, H, W) or (C, T, H, W) — float32 or float16 in [0,1] or [-1,1].
    Saves to .mp4 using torchvision.
    """
    import torchvision

    vid = tensor_4d.detach().cpu()

    # Convert (C, T, H, W) → (T, C, H, W)
    if vid.shape[0] in (1, 3):
        vid = vid.permute(1, 0, 2, 3)  # now (T, C, H, W)

    if vid.dtype in (torch.float32, torch.float16):
        if vid.min() < 0:
            vid = (vid + 1) / 2  # convert from [-1, 1] to [0, 1]
        vid = (vid.clamp(0, 1) * 255).to(torch.uint8)  # to [0, 255]

    # (T, C, H, W) → (T, H, W, C)
    vid = vid.permute(0, 2, 3, 1).contiguous()  # must be contiguous

    # Fix: ensure it's a torch.Tensor on CPU with uint8 dtype
    if not isinstance(vid, torch.Tensor):
        vid = torch.from_numpy(vid)
    if vid.device.type != "cpu":
        vid = vid.cpu()
    if vid.dtype != torch.uint8:
        vid = vid.to(torch.uint8)

    torchvision.io.write_video(filename=path, video_array=vid, fps=fps, video_codec='libx264')
    print(f"✅ Saved reconstructed video ➜ {path}")


class CL(torch.nn.Module):
    def __init__(self, config, bit):
        super(CL, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bit = bit

    def forward(self, h1, h2, weighted, labels):
        try:
            logits = torch.einsum('ik,jk->ij', h1, h2)
            logits = logits / self.bit / 0.3

            balance_logits = h1.sum(0) / h1.size(0)
            reg = self.mse(balance_logits, torch.zeros_like(balance_logits)) - self.mse(h1, torch.zeros_like(h1))
            weighted = torch.where(
                labels == 0,
                torch.zeros_like(weighted),
                weighted
            )
            loss = self.ce(logits, weighted.long()) + reg
        except Exception as e:
            print(f"Contrastive loss error: {e}")
            loss = torch.tensor(1.0, requires_grad=True).to(h1.device)
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits

    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def smooth_labels(labels, smoothing=0.1):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        labels = labels * (1 - smoothing) + smoothing / 2
    return labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def random_drop(batch, drop_prob=0.2, keys=("mfcc", "video")):
    """
    For every key in `keys` that is not None, drop it with `drop_prob`
    BUT never drop all of them in the same call.
    """
    present = [k for k in keys if batch[k] is not None]          # modalities that actually exist
    if len(present) <= 1:                                        # nothing to drop or only one present
        return batch

    # decide independently which present modalities to drop
    to_drop = [k for k in present if random.random() < drop_prob]

    # make sure at least one modality stays
    if len(to_drop) == len(present):
        keep_one = random.choice(to_drop)
        to_drop.remove(keep_one)

    # apply the drops
    for k in to_drop:
        batch[k] = None
        if f"{k}_aug" in batch:          # also clear *_aug if it exists
            batch[f"{k}_aug"] = None
    return batch


        
def forward_mse_loss(target, pred):
        # if self.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(var + 1e-6)
        
        loss = (pred - target).pow(2)
        loss = loss.mean()
    
        return loss

def main(args):
    
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30) )  # Initialize DDP

    local_rank = dist.get_rank() #int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # device = torch.device(f"cuda")
    # torch.cuda.set_device(local_rank)
    
    torch.backends.cudnn.benchmark = True  # selects best conv algos for your input sizes
    torch.backends.cudnn.enabled   = True
    scaler = GradScaler()
    resume_training = args.load_from is not None

    # Create the dataset
    path = args.base_dir
    

    
    def collate_fn(batch):
        # 1) turn each sample tuple into a unified dict (as before)…
        unified = []
        for sample in batch:
            if len(sample) == 13:
                (mfcc, mfcc_aug, audio,audio_aug, video, video_aug,
                text, landmarks, label, filenames, _, multi_label, flow) = sample
                d = {
                    "mfcc":      mfcc,      "mfcc_aug": mfcc_aug,
                    "audio":     audio,"audio_aug":audio_aug,       "video":     video,
                    "video_aug": video_aug, "text":      None,
                    "landmarks": landmarks, "flow":      None,
                    "images":    None,      "images_aug":None,
                    "labels":     label,"multi_label": multi_label
                }
            elif len(sample) == 6:
                video,video_aug,fake_video, fake_video_aug, label, diff_video = sample
                d = {
                    "mfcc":      None,      "mfcc_aug": None,
                    "audio":     None, "audio_aug":None,        "video":     video,
                    "video_aug": fake_video,      "text":      None,
                    "landmarks": None,      "flow":      None,
                    "images":    None,    "images_aug":None,
                    "labels":     label,"multi_label": diff_video
                }
                
            elif len(sample) == 4:
                video,video_aug, label, multi_label = sample
                
                d = {
                    "mfcc":      None,      "mfcc_aug": None,
                    "audio":     None,"audio_aug":None,      "video":     video,
                    "video_aug": video_aug,      "text":      None,
                    "landmarks": None,      "flow":      None,
                    "images":    None,    "images_aug":None,
                    "labels":     label,"multi_label":label
                }
                # print("these are 4")
                # exit()
            
            else:
                raise ValueError(f"Unexpected sample length {len(sample)}")
            unified.append(d)

        # 2) batch each field only if it's a Tensor; leave others as lists
        batched = {}
        for k in unified[0].keys():
            vals = [d[k] for d in unified]
            # if nobody in the batch has that modality
            if all(v is None for v in vals):
                batched[k] = None
                continue

            # find first non-None entry
            template = next(v for v in vals if v is not None)

            if isinstance(template, torch.Tensor):
                # fill any Nones with zero-tensors of the right shape
                filled = []
                for v in vals:
                    if v is None:
                        filled.append(torch.zeros_like(template))
                    else:
                        filled.append(v)
                # print(filled)
                batched[k] = torch.stack(filled, dim=0)
            else:
                # non-Tensor field: keep the raw list (ints, strings, etc.)
                batched[k] = vals

        return batched




    print("Before fake avceleb")
    val_loaders_saturations = []
    for sat in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": sat,
                "contrast":   None,
                "blockwise":  None,
                "gaussian_noise":     None,
                "gaussian_blur":      None,
                "jpeg":               None,
                "jpeg_compression":   None,
                "video_compression":  None
            }
        )
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_saturations.append(loader)

        
    val_loaders_contrast = []
    for contrast in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   contrast,
                "blockwise":  None,
                "gaussian_noise":     None,
                "gaussian_blur":      None,
                "jpeg":               None,
                "jpeg_compression":   None,
                "video_compression":  None
            }
        )
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_contrast.append(loader)



    val_loaders_blocwise = []
    for block in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   None,
                "blockwise":  block,
                "gaussian_noise":     None,
                "gaussian_blur":      None,
                "jpeg":               None,
                "jpeg_compression":   None,
                "video_compression":  None
            }
        )
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_blocwise.append(loader)



    val_loaders_g_noise = []
    for noise in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   None,
                "blockwise":  None,
                "gaussian_noise":     noise,
                "gaussian_blur":      None,
                "jpeg":               None,
                "jpeg_compression":   None,
                "video_compression":  None
            }
        )
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_g_noise.append(loader)



    val_loaders_gaussian_blur = []
    for blur in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   None,
                "blockwise":  None,
                "gaussian_noise":     None,
                "gaussian_blur":      blur,
                "jpeg":               None,
                "jpeg_compression":   None,
                "video_compression":  None
            }
        )
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_gaussian_blur.append(loader)

        
    val_loaders_jpeg = []
    for jpeg in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   None,
                "blockwise":  None,
                "gaussian_noise":     None,
                "gaussian_blur":      None,
                "jpeg":               jpeg,
                "jpeg_compression":   None,
                "video_compression":  None
            }
        )
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_jpeg.append(loader)



        
    val_loaders_jpeg_compressions = []
    for cmp in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   None,
                "blockwise":  None,
                "gaussian_noise":     None,
                "gaussian_blur":      None,
                "jpeg":               None,
                "jpeg_compression":   cmp,
                "video_compression":  None
            }
        )

        
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_jpeg_compressions.append(loader)


        
    val_loaders_video_compression = []
    for cmp in range(1, 6):  # 1,2,3,4,5
        ds = HashFormerDataset(
            path,
            partition="test",
            take_datasets="REAL_TEST,FakeAVCeleb",
            visual_preturb={
                "saturation": None,
                "contrast":   None,
                "blockwise":  None,
                "gaussian_noise":     None,
                "gaussian_blur":      None,
                "jpeg":               None,
                "jpeg_compression":   None,
                "video_compression":  cmp
            }
        )

        
        sampler = DistributedSampler(ds, shuffle=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders_video_compression.append(loader)

    # print("Dfdc loaded successfully ")
    # exit()
           



    pretraining = False
    # num_classes = train_dataset.class_numbers
    model = HashFormer(pretraining=pretraining)
    video_decoder = VisualDecoder16x16(n_frames=16, tubelet_size=4, embed_dim=512, depth=6, num_heads=4, encoder_embed_dim=512)
    video_decoder.cuda(local_rank)
    model.cuda(local_rank)     
    total_parameters = count_parameters(model=model)
    print(f"Model created. Trainable parameters: {total_parameters / 1e6:.2f}M")

    # optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
   # ----------------  optimiser  ----------------
    base_lr = 3e-4 * args.batch_size / 64          # linear-scaling rule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = base_lr,
        betas        = (0.9, 0.95),
        weight_decay = 1e-4
    )

    # ----------------  schedulers  ----------------
    warmup_iters = 500                              # <-- define before use

    # 1) Linear warm-up from 1 e-7 → base_lr over `warmup_iters` steps
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-7 / base_lr,              # relative factor, not absolute LR
        end_factor   = 1.0,                         # reach full base_lr
        total_iters  = warmup_iters
    )

    # 2) Cosine decay for the remaining steps
   
    start_epoch = 0
    

    # model = nn.DataParallel(model)
    # model.apply(reset_weights) 
    # dist.broadcast_parameters(model)
    

    if resume_training:
        checkpoint_path = os.path.join(args.load_from, "checkpoints", "model_state.pt")
        raw   = torch.load(checkpoint_path, map_location=device)
        # model.load_state_dict(state, strict=True)
        # if you saved the full dict with optimizer & friends, pull out the model_state
        sd    = raw.get("model_state_dict", raw)

        # strip off any "module." prefixes
        new_sd = {}
        for k, v in sd.items():
            new_key = k.replace("module.", "")  
            new_sd[new_key] = v
            
        model.load_state_dict(new_sd)        # strict=True by default
        

        # model.load_state_dict(state, strict=False)
        # optimizer.load_state_dict(raw['optimizer_state_dict'])
        # scheduler.load_state_dict(raw['scheduler_state_dict'])
        # start_epoch = raw['epoch']
        
        
        checkpoint_path = os.path.join(args.load_from, "checkpoints", "decoder_state.pt")
        raw   = torch.load(checkpoint_path, map_location=device)
        sd    = raw.get("model_state_dict", raw)

        new_sd = {}
        for k, v in sd.items():
            new_key = k.replace("module.", "")  
            new_sd[new_key] = v
            
        video_decoder.load_state_dict(new_sd)

        # visual_decoder.load_state_dict(torch.load(os.path.join(args.load_from, "decoder_state.pt")))
        
        
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)


        # log_dir = args.log_dir
    # else:
    log_dir = os.path.join(args.log_dir, args.encoder)
    os.makedirs(log_dir, exist_ok=True)
   
    
    if dist.get_rank() == 0:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

    # writer = SummaryWriter(log_dir=log_dir)
    if dist.get_rank() == 0:
        writer   = SummaryWriter(log_dir=log_dir)
        # model.apply(reset_weights)
    else: 
        writer = None
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))

    bit = 64
    config = None
    criterion = nn.BCEWithLogitsLoss()
    criterion_contrastive = CL(config, bit)
    CLASSIFICATION_WEIGHT = 1
    CONTRASTIVE_WEIGHT = 1
    best_val_eer = 1.0
    global_step = 0
    l2_lambda = 0.001


    intra_w_start     = 0.1
    intra_w_end       = 0.1
    inter_w_start     = 0.1
    inter_w_end       = 0.2
    lipsync_w_start   = 0.0
    lipsync_w_end     = 0.0
    moe_start   = 0.0
    moe_end     = 0.0
    recon_w_start     = 0.8
    recon_w_end       = 0.8
    cls_w_start       = 0.9
    cls_w_end         = 0.9   

    # Initialize BERT tokenizer (using "bert-base-uncased")

    best_threshold=0.5

    model.eval()

    # named_loaders = [
    # (val_loaders_saturations,       "saturation"),
    # (val_loaders_contrast,          "contrast"),
    # (val_loaders_blockwise,         "blockwise"),
    # (val_loaders_g_noise,           "gaussian_noise"),
    # (val_loaders_gaussian_blur,     "gaussian_blur"),
    # (val_loaders_jpeg,              "jpeg"),
    # (val_loaders_jpeg_compressions, "jpeg_compression"),
    # (val_loaders_video_compression, "video_compression"),
    # ]

    # 2) enumerate them, convert idx→str, and unpack name
    # for idx, (loader, loader_name) in enumerate(named_loaders, start=1):
    #     idx_str = str(idx)
    #     print(f"Loader #{idx_str} → {name}")

    for loader_info in [
                    {"loader":val_loaders_saturations, "name":    "saturation"},
                    {"loader":val_loaders_contrast, "name":       "contrast"},
                    {"loader":val_loaders_blocwise, "name":      "blockwise"},
                    {"loader":val_loaders_g_noise, "name":        "gaussian_noise"},
                    {"loader":val_loaders_gaussian_blur, "name":  "gaussian_blur"},
                    {"loader":val_loaders_jpeg, "name":           "jpeg"},
                    {"loader":val_loaders_jpeg_compressions, "name":"jpeg_compression"},
                    {"loader":val_loaders_video_compression,"name": "video_compression"},
                    ]:

        data_loader = loader_info['loader']
        loader_name = loader_info['name']
        for strength,sub_loader in enumerate(data_loader):
            val_loss = 0
            correct_predictions = 0
            total_samples = 0
            pos_samples, neg_samples, all_preds, all_labels = [], [], [], []
            # model.pretraining = pretraining
            model.is_training=False
            with torch.no_grad():
                for i, batch in enumerate(tqdm(sub_loader, desc="Validation")):

                    mfcc,mfcc_aug, audio,audio_aug, video,video_aug, text_tokens, landmarks, label, filenames, _ ,multi_label, optical_flow= batch
                    # batch = random_drop(batch, drop_prob=0.2, keys=("mfcc", "video"))
                    for mod in batch:
                        if batch[mod] is not None: 
                            if mod=="labels":
                                label = batch[mod] = batch[mod].float().to(device)
                            else:
                                batch[mod]=batch[mod].to(device)

                    # landmarks= None
                    # classification_logits, contrastive_loss = model(mfcc, audio, video, landmarks, text_tokens, label, epoch)
                    # with autocast(dtype=torch.float16):
                    classification_logits,losses, label, v_tok  = model(**batch)
                    
                    mse_loss_video = 0 
                    losses["reconstruction"] = {}
                    if batch.get("video") is not None:
                        # predictor(v_tok)
                        video_recon = video_decoder(v_tok)
                        video_recon = video_decoder.unpatch_to_img(video_recon)
                        # mse_loss_video = forward_mse_loss(batch["video"].permute(0,2,1,3,4), video_recon)
                        losses["reconstruction"]["video"] = forward_mse_loss(
                            batch["video"].permute(0, 2, 1, 3, 4), video_recon
                        )



                    intra_sum = sum(losses["intra"].values())
                    inter_sum = sum(losses["inter"].values())
                    reconstructionLoss = sum(losses["reconstruction"].values())
                    # reconstructionLoss = mse_loss_video
                    cls_loss = sum(losses["cls_loss"].values())

                    total_loss = (
                    0.1   * intra_sum
                    + 0.1   * inter_sum
                    # + w_lipsync * lipsyncLoss
                    + 0.8   * reconstructionLoss
                    )

                    loss = 0.8 * cls_loss + total_loss * 0.2
                    

                    val_loss += loss
                    pred = torch.sigmoid(classification_logits)
                    pred_class = (pred > best_threshold).long().squeeze()
                    correct_predictions += (pred_class == label).sum().item()
                    total_samples += label.size(0)
                    pos_samples.append(pred[label == 1].detach().cpu().numpy())
                    neg_samples.append(pred[label == 0].detach().cpu().numpy())
                    all_preds.extend(pred.detach().cpu().numpy())
                    all_labels.extend(label.detach().cpu().numpy())
                        
                    # if writer is not None:
                    #     print(pred.min(), pred.max(), pred.mean())
                    #     print("Labels are: ",torch.unique(label, return_counts=True))
                    
                    #     if total_norm < 1e-5 or total_norm > 1e3:
                    #         print("Unstable gradient norm:", total_norm)
                    #     print("Logits:", logits[:5])
                    #     print("Pred min and max",pred.min(), pred.max(), pred.mean())
                    #     print("Preds:", torch.sigmoid(classification_logits)[:5])
                    #     print("Labels:", label[:5])
                    # if writer is not None:
                    #     if i==1:
                    #         _save_with_cv2(video_recon[0],  # first sample in batch
                    #                                     path=f"{log_dir}/recon_{loader_name}.mp4",
                    #                                     fps=25)
                

                val_loss /= len(sub_loader)
                val_accuracy = correct_predictions / total_samples
                # scheduler.step(val_accuracy)

                val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
                roc_auc_val = roc_auc_score(all_labels, all_preds)
                precision_val, recall_val, _ = precision_recall_curve(all_labels, all_preds)
                auc_pr_val = auc(recall_val, precision_val)
                    
                # fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=2)
                # auc_r_val = auc(fpr, tpr)
                ap_score_val = average_precision_score(all_labels, all_preds)
                # precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
                # f1_scores = 2 * precision * recall / (precision + recall)
                # best_threshold = thresholds[np.argmax(f1_scores)]

                if writer is not None:
                    writer.add_scalar(f"{loader_name}/{strength}/Val/AvgLoss", val_loss, 0)
                    writer.add_scalar(f"{loader_name}/{strength}/Val/Accuracy", val_accuracy, 0)
                    writer.add_scalar(f"{loader_name}/{strength}/Val/ROC_AUC", roc_auc_val, 0)
                    writer.add_scalar(f"{loader_name}/{strength}/Val/AUC", auc_pr_val, 0)
                    # writer.add_scalar("Val/AUC_R", auc_r_val, 0)
                    writer.add_scalar(f"{loader_name}/{strength}/Val/AP", ap_score_val, 0)
                    writer.add_scalar(f"{loader_name}/{strength}/EER/val", val_eer, 0)

            # if val_eer < best_val_eer:
                #     best_val_eer = val_eer
                #     if dist.get_rank() == 0:
                #         torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))

            # if epoch % 3 == 0:
            #     torch.save(model.state_dict(),
            #                os.path.join(checkpoint_dir,
            #                             f"model_{epoch}_EER_{val_eer:.4f}_loss_{val_loss:.4f}_acc_{val_accuracy:.4f}.pt"))


    # if writer is not None:
        # writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, default="rawnet", help="The encoder to use.")
    parser.add_argument("--batch_size", type=int, default=24, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=10, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")
    parser.add_argument("--load_from", type=str, default=None, help="The path to the checkpoint to load from.")

    args = parser.parse_args()
    main(args)
