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
from dataloaders.datasets import MMModeratorDataset
from dataloaders.faceswapDatasetAligned import FaceswapDataset
from models.aaai.model import MMModerator
from util import seed_worker, set_seed, compute_eer
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import silhouette_score
from torch.cuda.amp import autocast, GradScaler
# torch.autograd.set_detect_anomaly(True)

from datetime import timedelta
from transformers import get_linear_schedule_with_warmup

# optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=total_steps)

# torch.set_printoptions(threshold=10_0000)

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


def main(args):
    
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30) )  # Initialize DDP

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # device = torch.device(f"cuda")
    torch.cuda.set_device(local_rank)
    
    torch.backends.cudnn.benchmark = True  # selects best conv algos for your input sizes
    torch.backends.cudnn.enabled   = True
    scaler = GradScaler()
    resume_training = args.load_from is not None

    # Create the dataset
    path = args.base_dir
    # train_dataset = MMModeratorDataset(path, partition="train", take_datasets="1,2")
    val_dataset = MMModeratorDataset(path, partition="test", take_datasets="1,2")
    # train_size = len(train_dataset)
    # val_size = int(train_size * 0.1)
    # remaining_size = train_size - val_size

    # val_subset, train_subset = random_split(train_dataset, [val_size, remaining_size])

    # faceswap_dev_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=0, worker_init_fn=seed_worker)
    # dev_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=0)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # faceswap_dev_loader = DataLoader(train_dataset, batch_size=args.batch_size,
    #                       sampler=train_sampler,
    #                       num_workers=args.num_workers,
    #                       pin_memory=True,
    #                       drop_last=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # dev_loader = DataLoader(val_dataset, batch_size=args.batch_size,
    #                     sampler=val_sampler,
    #                     num_workers=args.num_workers,
    #                     pin_memory=True,
    #                     drop_last=True)



    # faceswap_train_dataset = FaceswapDataset(path, partition="train", take_datasets="1,2")
    faceswap_val_dataset = FaceswapDataset(path, partition="test", take_datasets="1,2")


    # train_sampler = torch.utils.data.distributed.DistributedSampler(faceswap_train_dataset)
    # faceswap_faceswap_dev_loader = DataLoader(faceswap_train_dataset, batch_size=128,
    #                       sampler=train_sampler,
    #                       num_workers=args.num_workers,
    #                       pin_memory=True,
    #                       drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(faceswap_val_dataset, shuffle=False)
    faceswap_dev_loader = DataLoader(faceswap_val_dataset, batch_size=128,
                        sampler=val_sampler,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True)


    # Create the model
    pretraining = False
    model = MMModerator(pretraining=pretraining)
    model = model.to(device)
    total_parameters = count_parameters(model=model)
    print(f"Model created. Trainable parameters: {total_parameters / 1e6:.2f}M")

    # optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
   # ----------------  optimiser  ----------------
    base_lr = 3e-4 * args.batch_size / 256          # linear-scaling rule
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
    total_steps  = args.epochs * len(faceswap_dev_loader)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = total_steps - warmup_iters,       # only after warm-up
        eta_min = 1e-6                              # final LR
    )

    # 3) Chain them in the correct order
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_iters]                 # switch after warm-up_iters
    )

    start_epoch = 0
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        

    # model = nn.DataParallel(model)
    model.apply(reset_weights) 
    # dist.broadcast_parameters(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
    # 
    if resume_training:
        state = torch.load(os.path.join(args.load_from, "checkpoints", "model_state.pt"))
        model.load_state_dict(state['model_state_dict'], strict=False)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch']
        # model.load_state_dict(torch.load(os.path.join(args.load_from, "best_pretrained.pt")))
        # log_dir = args.log_dir
    # else:
    log_dir = os.path.join(args.log_dir, args.encoder)
    os.makedirs(log_dir, exist_ok=True)
   

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
    intra_w_end       = 0.01
    inter_w_start     = 0.2
    inter_w_end       = 0.05
    lipsync_w_start   = 0.1
    lipsync_w_end     = 0.01
    recon_w_start     = 0.6
    recon_w_end       = 0.1
    cls_w_start       = 0.8
    cls_w_end         = 0.8   # keep classification weight fixed


    # Initialize BERT tokenizer (using "bert-base-uncased")

    best_threshold=0.5
    for epoch in range(0, 1):
        model.train()
        model.total_samples = len(faceswap_dev_loader)
        train_loss_epoch = 0.0
        correct_predictions = 0
        total_samples = 0
        pos_samples, neg_samples, all_preds, all_labels = [], [], [], []
        grad_norms = []
        # model.pretraining = True
       
        # Evaluate on the validation set.
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0
        pos_samples, neg_samples, all_preds, all_labels = [], [], [], []
        # model.pretraining = pretraining
        with torch.no_grad():
            for i, batch in enumerate(tqdm(faceswap_dev_loader, desc="Validation")):
                
                if len(batch)==11:
                    mfcc,mfcc_aug, audio, video,video_aug, text_tokens, landmarks, label, filenames, _ , optical_flow= batch
                    # mfcc,audio, video, text_tokens.float(), landmarks.float(), label, os.path.basename(file_path),data_label,optical_flow.float()
                    audio = audio.to(device)
                    video = video.to(device)
                    video_aug = video_aug.to(device)
                    mfcc = mfcc.to(device)
                    mfcc_aug = mfcc_aug.to(device)
                    
                    images = torch.randn((video.shape[0],3,224,224)).to(device)
                    # landmarks = landmarks.to(device)
                    label = label.to(device, dtype=torch.float)
                    
                    soft_label = smooth_labels(labels=label)

                    text_tokens = text_tokens.to(device)
                    optical_flow  = None
                    text_tokens  = None
                    images = None
                    images_aug = None
                elif len(batch)==4:
                    images, images_aug, label, filenames  = batch
                    audio = None
                    video = None
                    video_aug = None
                    mfcc = None
                    mfcc_aug = None
                    landmarks = None #landmarks.to(device)
                    label = label.to(device, dtype=torch.float)
                    soft_label = smooth_labels(labels=label)
                    optical_flow  = None
                    text_tokens  = None
                    images = images.to(device)
                    # fft_magnitude = fft_magnitude.to(device)
                    images_aug = images_aug.to(device)

                # landmarks= None
                # classification_logits, contrastive_loss = model(mfcc, audio, video, landmarks, text_tokens, label, epoch)
                # with autocast(dtype=torch.float16):
                classification_logits,losses, label  = model(mfcc=mfcc,mfcc_aug=mfcc_aug, audio=audio, video=video,video_aug=video_aug,  landmarks=landmarks, flow=optical_flow, text=text_tokens,images=images, images_aug=images_aug,labels=label)
                
                if pretraining:
                    intra_sum = sum(losses["intra"].values())
                    inter_sum = sum(losses["inter"].values())
                    lipsyncLoss = sum(losses["lipsyncLoss"].values())
                    reconstructionLoss = sum(losses["reconstruction"].values())
                    # print(reconstructionLoss)
                    # exit()
                    # total_loss = 0.3 * intra_sum + 0.3 * inter_sum + 0.1 * lipsyncLoss + 0.3 * reconstructionLoss
                    total_loss = (
                            0.1   * intra_sum
                            + 0.2 * inter_sum
                            + 0.1   * lipsyncLoss
                            + 0.6 * reconstructionLoss   # was 0.001 already—just anneal
                        )
                    loss_cls = criterion(logits.view(-1), label)
                    loss = 0.8 * loss_cls + total_loss * 0.2
                else:
                    
                    intra_sum = sum(losses["intra"].values())
                    inter_sum = sum(losses["inter"].values())
                    lipsyncLoss = sum(losses["lipsyncLoss"].values())
                    reconstructionLoss = sum(losses["reconstruction"].values())
                    cls_loss = sum(losses["cls_loss"].values())

                    w_intra = 0.1
                    w_inter = 0.2
                    w_lipsync= 0
                    w_recon = 0.6
                    total_loss = (
                    w_intra   * intra_sum
                    + w_inter   * inter_sum
                    + w_lipsync * lipsyncLoss
                    + w_recon   * reconstructionLoss
                    )
                    # print("total loss",total_loss)
                    # print(intra_sum,inter_sum,lipsyncLoss,reconstructionLoss,cls_loss)

                    # cls_loss = sum(losses["cls_loss"].values())
                    # print("cls_loss",cls_loss)
                    loss = 0.8 * cls_loss + total_loss * 0.2
                    loss = loss.mean()

                val_loss += loss.item()
                pred = torch.sigmoid(classification_logits)
                pred_class = (pred > best_threshold).long().squeeze()
                correct_predictions += (pred_class == label).sum().item()
                total_samples += label.size(0)
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                all_preds.extend(pred.detach().cpu().numpy())
                all_labels.extend(label.detach().cpu().numpy())

            val_loss /= len(faceswap_dev_loader)
            val_accuracy = correct_predictions / total_samples
            # scheduler.step(val_accuracy)
            val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
            roc_auc_val = roc_auc_score(all_labels, all_preds)
            precision_val, recall_val, _ = precision_recall_curve(all_labels, all_preds)
            auc_pr_val = auc(recall_val, precision_val)
            ap_score_val = average_precision_score(all_labels, all_preds)
            # precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
            # f1_scores = 2 * precision * recall / (precision + recall)
            # best_threshold = thresholds[np.argmax(f1_scores)]

            if writer is not None:
                writer.add_scalar("Val/AvgLoss", val_loss, epoch)
                writer.add_scalar("Val/Accuracy", val_accuracy, epoch)
                writer.add_scalar("Val/ROC_AUC", roc_auc_val, epoch)
                writer.add_scalar("Val/AUC_PR", auc_pr_val, epoch)
                writer.add_scalar("Val/AP", ap_score_val, epoch)
                writer.add_scalar("EER/val", val_eer, epoch)

                
                print("Val/AvgLoss", val_loss, epoch)
                print("Val/Accuracy", val_accuracy, epoch)
                print("Val/ROC_AUC", roc_auc_val, epoch)
                print("Val/AUC_PR", auc_pr_val, epoch)
                print("Val/AP", ap_score_val, epoch)
                print("EER/val", val_eer, epoch)

            if val_eer < best_val_eer:
                best_val_eer = val_eer
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))

            # if epoch % 5 == 0:
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
