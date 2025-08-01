import argparse
import av
import cv2
import torch
import numpy as np
import librosa
from decord import VideoReader, cpu
from transformers import BertTokenizer
import mediapipe as mp
import traceback
from models.model import HashFormer  # Your model definition
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import warnings
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore", category=UserWarning)  # optional – quiets sklearn


from tqdm import tqdm
from dataloaders.faceswapVideoDataset import FaceswapVideoDataset
# your dataset
faceswap_val_video_dataset = FaceswapVideoDataset(
    "<path_to_dataset>/deepspeak/",
    partition="test",
    take_datasets=["NeuralTextures"]
)

# simple DataLoader
faceswap_video_val_loader = DataLoader(
    faceswap_val_video_dataset,
    batch_size=1,
    shuffle=False,       # no DistributedSampler, so specify shuffle directly
    num_workers=1,
    pin_memory=True,
    drop_last=True       # keep same behavior as before
)

# def load_model(checkpoint_path: str, device: torch.device) -> MMModerator:
#     """
#     Load the MMModerator model from checkpoint and set to eval mode.
#     """
#     model = MMModerator(pretraining=False)
#     state = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state)
#     model.to(device).eval()
#     return model

def load_model(checkpoint_path, device):
    raw   = torch.load(checkpoint_path, map_location=device)
    # if you saved the full dict with optimizer & friends, pull out the model_state
    sd    = raw.get("model_state_dict", raw)

    # strip off any "module." prefixes
    new_sd = {}
    for k, v in sd.items():
        new_key = k.replace("module.", "")  
        new_sd[new_key] = v

    model = HashFormer(pretraining=False)
    model.load_state_dict(new_sd)        # strict=True by default
    return model.to(device).eval()


def extract_audio(video_path: str,
                  sample_rate: int = int(16000),
                  clip_len_sec: int = 1
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read entire audio, split into 1s clips, return:
      mfcc: (N, n_mfcc, T) tensor
      audio: (N, clip_len_sec*sample_rate) tensor
    """
    container = av.open(video_path)
    stream = next((s for s in container.streams if s.type == 'audio'), None)
    if stream is None:
        container.close()
        raise RuntimeError(f"No audio stream in {video_path}")

    resampler = av.audio.resampler.AudioResampler(
        format='s16', layout='mono', rate=sample_rate
    )
    decoded = []
    for frame in container.decode(audio=0):
        for sfr in resampler.resample(frame):
            arr = sfr.to_ndarray().reshape(-1)
            decoded.append(arr)
    container.close()

    if not decoded:
        audio_np = np.zeros(clip_len_sec * sample_rate, dtype=np.float32)
    else:
        audio_np = np.concatenate(decoded).astype(np.float32) / 32768.0
    clip_len = sample_rate * clip_len_sec
    total_samples = len(audio_np)
    n_clips = max(1, total_samples // clip_len)
    if total_samples < n_clips * clip_len:
        pad = np.zeros(n_clips * clip_len - total_samples, dtype=np.float32)
        audio_np = np.concatenate([audio_np, pad])
    else:
        audio_np = audio_np[:n_clips * clip_len]

    audio_clips = audio_np.reshape(n_clips, clip_len)
    mfcc_list = []
    for clip in audio_clips:
        m = librosa.feature.mfcc(
            y=clip, sr=sample_rate, n_mfcc=40,
            hop_length=512, n_fft=2048
        )
        m_db = librosa.power_to_db(m, ref=np.max)
        m_norm = librosa.util.normalize(m_db)
        mfcc_list.append(torch.tensor(m_norm, dtype=torch.float32))
    mfcc = torch.stack(mfcc_list, dim=0)  # (N_audio, n_mfcc, T)
    audio = torch.tensor(audio_clips, dtype=torch.float32)  # (N_audio, clip_len)
    return mfcc, audio

def equalize_frames(frames_np: np.ndarray) -> torch.Tensor:
    """
    Histogram-equalize the Y channel of RGB frames.
    Input: (T, H, W, 3) numpy RGB; output: (T, 3, H, W) float [0,1].
    """
    eq = []
    for f in frames_np:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        eq.append(rgb)
    eq_np = np.stack(eq, 0)
    return torch.from_numpy(eq_np).permute(0,3,1,2).float() / 255.0



def extract_and_crop_frames(video_path: str,
                            clip_len_sec: int = 1,
                            fps: int = 16
                            ) -> torch.Tensor:
    """
    Read entire video, split into 1s clips, crop faces per frame with margin, resize to 224×224.
    Returns: (N_video, T, C, H, W) tensor
    """

    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    step = int(round(native_fps / fps))  # subsample step
    frames = vr.get_batch(range(0,len(vr),step)).asnumpy()  # (F, H, W, 3)
    clip_len = int(clip_len_sec * fps)
    
    total_frames = frames.shape[0]
    n_clips = max(1, total_frames // clip_len)
    if total_frames < n_clips * clip_len:
        pad_frames = np.zeros((n_clips * clip_len - total_frames, *frames.shape[1:]), dtype=frames.dtype)
        frames = np.concatenate([frames, pad_frames], axis=0)
    else:
        frames = frames[:n_clips * clip_len]

    frames_clips = frames.reshape(n_clips, clip_len, *frames.shape[1:])

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    crops = []
    margin = 40
    for clip in frames_clips:
        clip_tensors = []
        for frame in clip:

            frame = equalize_frames(frame[np.newaxis])[0].permute(1,2,0).numpy()*255
            frame = frame.astype(np.uint8)
            res = detector.process(frame)
            h, w, _ = frame.shape
            # if res.detections:
            
            detected_any = True
            if not res.detections:
                # skip this frame entirely
                detected_any = False
                continue
            det = max(
                res.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width *
                    d.location_data.relative_bounding_box.height
                )
            )
            r = det.location_data.relative_bounding_box
            x1 = int(r.xmin * w)
            y1 = int(r.ymin * h)
            x2 = x1 + int(r.width * w)
            y2 = y1 + int(r.height * h)
            x_min = max(x1 - margin, 0)
            y_min = max(y1 - margin, 0)
            x_max = min(x2 + margin, w)
            y_max = min(y2 + margin, h)
            crop_img = frame[y_min:y_max, x_min:x_max]
            # else:
            #     # crop_img = frame
            #     crop_img = np.zeros_like(frame) 

            crop_img = cv2.resize(crop_img, (224, 224))
            t = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0
            clip_tensors.append(t)
            if len(clip_tensors)==4:
                break
        if detected_any:
            crops.append(torch.stack(clip_tensors, dim=0))
    detector.close()
    return torch.stack(crops, dim=0)  # (N_video, T, C, H, W)


def inference(video_path: str, checkpoint: str, device: torch.device, model , tokenizer, label):
    
    mfcc, audio = extract_audio(video_path)
    video = extract_and_crop_frames(video_path)
    
    
    # mfcc = torch.zeros((27, 40, 32), dtype=torch.float32)
    # audio = torch.zeros((27,48000), dtype=torch.float32)
    # Align batch sizes
    MAX_VIDEOS = 2*2
    B_audio = mfcc.shape[0]
    B_video = video.shape[0]
    
    # mfcc = torch.zeros((B_video, 40, 368), dtype=torch.float32)
    # audio = torch.zeros((B_video, 16000), dtype=torch.float32)
    mfcc_aug = mfcc.clone()
    B = min(B_audio, B_video)
    B = min(B,MAX_VIDEOS)
    mfcc = mfcc[:B]
    audio = audio[:B]
    video = video[:B]

    
    # out_dir = 'output_dir'
    # os.makedirs(out_dir, exist_ok=True)
    # for i in range(B):
    #     clip = (video[i] * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
    #     writer = cv2.VideoWriter(
    #         os.path.join(out_dir, f'clip_{i}.mp4'),
    #         cv2.VideoWriter_fourcc(*'mp4v'),
    #         24,
    #         (224, 224)
    #     )
    #     for frame in clip:
    #         # // convert RGB back to BGR for OpenCV
    #             bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             writer.write(bgr_frame)
    #     writer.release()
    #     print(f"Saved test clip clip_{i}.mp4 to {out_dir}")



    mfcc_aug = mfcc.clone()
    # labels = video.clone()

    # Prepare other inputs
    texts = ["dummy text"] * B
    txt = tokenizer(
        texts,
        max_length=20,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids  # (B, 20)

    T = video.size(1)
    lm = torch.zeros((B, T, 2), dtype=torch.float32)
    flow = torch.zeros((B, T, 2), dtype=torch.float32)
    

    # Move to device
    mfcc = mfcc.to(device)
    mfcc_aug = mfcc_aug.to(device)
    audio = audio.to(device)
    video = video.to(device)
    video_aug = video.clone().to(device)
    txt = txt.to(device)
    lm = lm.to(device)
    flow = flow.to(device)
    labels = torch.zeros((video.shape[0],), dtype=torch.float32).to(device)
    # print(video.shape, mfcc.shape, labels.shape)
    chunk_size=2
    correct_predictions_array=[]
    scores_array=[]
    label_array=[]
    for start in range(0, video.shape[0], chunk_size):
        end = min(start + chunk_size, video.shape[0])
        if video[start:end].shape[0]<2:
            continue

        print(video[start:end].shape, audio[start:end].shape, labels[start:end].shape)
        with torch.no_grad():
            logits, *_ = model(
                mfcc=None,
                mfcc_aug=None,
                audio=audio[start:end],
                video=video[start:end],
                video_aug=video_aug[start:end],
                text=None,
                landmarks=None,
                flow=None,
                images=None,
                images_aug=None,
                labels=labels[start:end],
                multi_label=labels[start:end].long()
            )
                        
            probs = torch.sigmoid(logits).squeeze()   # [T] or [B, T]
            scores = probs.cpu()                      # Keep it in PyTorch for topk
            scores_array.extend(scores.tolist())
            pred_class = (scores > 0.5).long().squeeze()


            label_chunk = torch.full_like(pred_class, fill_value=label, dtype=torch.long)
            label_array.extend(label_chunk.tolist())
            
            # label_chunk = torch.zeros_like(pred_class)
            # label_array.extend(label_chunk.cpu().tolist())  

            correct_predictions = (pred_class == label).sum().item()
            correct_predictions_array.append(correct_predictions)

            k = min(5, scores.numel())                # Handle short sequences
            topk_mean = torch.topk(scores, k=k).values.mean().item()
            median_score = scores.median().item()
            q90 = torch.quantile(scores, 0.9).item()
            print(scores.numpy())                     # Optional: convert to numpy just for printing
            print("number of fake clip detected ",(scores > 0.5).sum().item(),"number of real clip detected ", (scores < 0.5).sum().item())
            print(f"Clip: Forgery score (mean)    = {scores.mean():.4f}")
            print(f"Clip: Forgery score (top {k}) = {topk_mean:.4f}")
            print(f"Clip: Forgery Median Score = {median_score:.4f}")
            print(f"Clip: Forgery Quartile = {q90:.4f}")
    total_predictions = sum(correct_predictions_array)
    total_clips = len(scores_array)
    return total_predictions, total_clips, scores_array, label_array

    # for i, batch in enumerate(tqdm(faceswap_video_val_loader, desc="Validation")):
    #     video, video_aug, label  = batch
    #     video = video.to(device)
    #     video_aug = video_aug.to(device)
    #     label = label.to(device, dtype=torch.float)
    #     # # mfcc, audio = extract_audio(video_path)
    #     # video = extract_and_crop_frames(video_path)
        
        
    #     # # mfcc = torch.zeros((27, 40, 32), dtype=torch.float32)
    #     # # Align batch sizes
        
    #     # # B_audio = mfcc.shape[0]
    #     # B_video = video.shape[0]
        
    #     # mfcc = torch.zeros((B_video, 40, 368), dtype=torch.float32)
    #     # audio = torch.zeros((B_video, 16000), dtype=torch.float32)
    #     # mfcc_aug = mfcc.clone()
    #     # # B = min(B_audio, B_video)
    #     # B=B_video
    #     # mfcc = mfcc[:B]
    #     # audio = audio[:B]
    #     # video = video[:B]

        
    #     # out_dir = 'output_dir'
    #     # os.makedirs(out_dir, exist_ok=True)
    #     # for i in range(B):
    #     #     clip = (video[i] * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
    #     #     writer = cv2.VideoWriter(
    #     #         os.path.join(out_dir, f'clip_{i}.mp4'),
    #     #         cv2.VideoWriter_fourcc(*'mp4v'),
    #     #         24,
    #     #         (224, 224)
    #     #     )
    #     #     for frame in clip:
    #     #         # // convert RGB back to BGR for OpenCV
    #     #             bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     #             writer.write(bgr_frame)
    #     #     writer.release()
    #     #     print(f"Saved test clip clip_{i}.mp4 to {out_dir}")



    #     # mfcc_aug = mfcc.clone()
    #     # label = video.clone()

    #     # # Prepare other inputs
    #     # texts = ["dummy text"] * B
    #     # txt = tokenizer(
    #     #     texts,
    #     #     max_length=20,
    #     #     padding="max_length",
    #     #     truncation=True,
    #     #     return_tensors="pt"
    #     # ).input_ids  # (B, 20)

    #     # T = video.size(1)
    #     # lm = torch.zeros((B, T, 2), dtype=torch.float32)
    #     # flow = torch.zeros((B, T, 2), dtype=torch.float32)
    #     # labels = torch.zeros((B,), dtype=torch.float32)

    #     # # Move to device
    #     # mfcc = mfcc.to(device)
    #     # mfcc_aug = mfcc_aug.to(device)
    #     # audio = audio.to(device)
    #     # video = video.to(device)
    #     # video_aug = video_aug.to(device)
    #     # txt = txt.to(device)
    #     # lm = lm.to(device)
    #     # flow = flow.to(device)
    #     # labels = labels.to(device)
    #     # print(video.shape, mfcc.shape)
    #     with torch.no_grad():
    #         logits, *_ = model(
    #             mfcc=None,
    #             mfcc_aug=None,
    #             audio=None,
    #             video=video,
    #             video_aug=video_aug,
    #             text=None,
    #             landmarks=None,
    #             flow=None,
    #             images=None,
    #             images_aug=None,
    #             labels=label
    #         )
    #         scores = torch.sigmoid(logits).cpu().numpy()
    #         print(scores)
    #         print(f"Clip: Forgery score = {scores.mean():.4f}")
    #         # exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    correct = 0 
    total = 0 
    all_probs, all_labels = [], []          # collectors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = load_model(args.checkpoint, device)

    print(f"Processing: video {args.video_path}")
    # result = inference(args.video_path, args.checkpoint, device, model,tokenizer, 0)
    # print(result)
    # exit()

    for main_dir in os.listdir(args.video_path):
        total_cls = 0 
        main_dir_path = os.path.join(args.video_path, main_dir)
        if "real" not in main_dir and "fake" not in main_dir:
            continue
        if not os.path.isdir(main_dir_path):   # skip if not a directory
            continue


        for video in os.listdir(os.path.join(args.video_path, main_dir)):
            path = os.path.join(args.video_path ,main_dir, video)
            if "mp4" in video:
                try:
                    print(f"Processing: video {main_dir}, {video}")
                    label = 0 if "real" in main_dir else 1
                    print(f"The video label is {label}, which is {main_dir}")
                    correct_predictions, num_tot, probs, labels = inference(path, args.checkpoint, device, model,tokenizer, label)
                    all_probs.append(probs)
                    all_labels.append(labels)
                    
                    correct += correct_predictions
                    total += num_tot

                    print(f"Total {correct} are correct out of {total} - {correct/total:.4f}")
                   
                    total_cls+=1
                    # if total_cls == 50:
                    #     break
                # exit()
                except Exception as e:

                    print(f"Error processing {path}: {e}")
                    traceback.print_exc()
                    continue

    print(f"Total {correct} are correct out of {total}")
    print(f"Accuracy is {correct/total:.4f}")

    # ---------- metric computation ----------
    y_score = np.concatenate(all_probs)     # shape (N_clips,)
    y_true  = np.concatenate(all_labels)    # shape (N_clips,)
    y_pred  = (y_score > 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_score)
    ap   = average_precision_score(y_true, y_score)
# Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # False negative rate
    fnr = 1 - tpr

    # EER is the point where FPR and FNR are equal (or as close as possible)
    # find the threshold where |FPR − FNR| is minimal
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    print("\n=====  FINAL METRICS  =====")
    print(f"EER threshold       : {eer_threshold:.4f}")
    print(f"Equal-Error-Rate (EER): {eer:.4f}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print(f"ROC-AUC           : {auc:.4f}")
    print(f"Average Precision : {ap:.4f}")
 