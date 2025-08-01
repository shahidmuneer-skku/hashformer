import os
import random
import traceback
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from audiomentations import (
    AddGaussianNoise,
    BandPassFilter,
    Gain,
    Normalize,
    PitchShift,
    PolarityInversion,
    SevenBandParametricEQ,
    Shift,
    TimeStretch,
)
from decord import VideoReader, cpu
from kornia import augmentation as K
from torch.utils.data import Dataset
from transformers import BertTokenizer

###########################################################################
# ----------------------------  helper utils  --------------------------- #
###########################################################################

def equalize_frames(frames_np: np.ndarray) -> torch.Tensor:
    """Histogram‑equalise luminance channel of a (T, H, W, C) RGB numpy array."""
    eq = []
    for f in frames_np:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        eq.append(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
    eq = np.stack(eq, 0)  # (T, H, W, C)
    return torch.from_numpy(eq).permute(0, 3, 1, 2).float() / 255  # (T, C, H, W)


def get_video_path(root, method, pair_filename):
    return os.path.join(
        root,
        "video_data",
        "data_raw",
        "manipulated_sequences",
        method,
        "raw",
        "cropped_faces",
        pair_filename,
    )


def get_real_path(root, video_id):
    return os.path.join(
        root,
        "video_data",
        "data_raw",
        "original_sequences",
        "youtube",
        "raw",
        "videos",
        "features",
        f"{video_id}",
        "cropped_face.mp4"
    )

###########################################################################
# --------------------------  MAIN DATASET  ----------------------------- #
###########################################################################

class FaceswapVideoDataset(Dataset):
    """Every item is a fixed‑length clip (default 3 s)."""

    def __init__(
        self,
        base_dir: str | Path,
        partition: str = "train",
        clip_len_sec: int = 1,
        clip_stride_sec: int | None = 1,
        fps: int = 16,
        take_datasets=["NeuralTextures","FaceSwap", "FaceShifter", "Face2Face","Deepfakes"]
    ):
        super().__init__()
        assert partition in {"train", "dev", "test"}
        self.partition = partition
        self.base_dir = Path(base_dir)
        self.is_train = partition == "train"
        self.fps = fps
        self.clip_len = clip_len_sec * fps
        self.clip_stride = (clip_stride_sec or clip_len_sec) * fps

        # -------------  augs & tokenizer (same as before) ----------------
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.video_aug_ori = K.AugmentationSequential(
            K.Resize((224, 224)),
            K.RandomHorizontalFlip(p=0.2),
            K.RandomHorizontalFlip(p=0.2),
            K.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.9, 1.1), p=0.2),
            K.RandomErasing(scale=(0.02, 0.1), ratio=(0.2, 1.6), p=0.1),
        )

        self.video_aug_val = K.AugmentationSequential(
            K.Resize((224, 224)),
        )
        self.video_aug = K.AugmentationSequential(
            K.Resize((224, 224)),
            K.RandomHorizontalFlip(p=0.9),
            K.ColorJitter(0.2, 0.2, 0.2, 0.0, p=0.9),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.9),
            K.RandomMotionBlur(kernel_size=3, angle=15.0, direction=0.5, p=0.9),
            K.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.9, 1.1), p=0.6),
            K.RandomErasing(scale=(0.02, 0.33), ratio=(0.5, 3.6), p=0.9),
            same_on_batch=True,
        )

        # -------------  build list of video files (unchanged) ------------
        self.file_list: list[dict] = []  # each {path, label}
        ff_root = Path("/media/NAS/DATASET/FaceForensics_Dec2020/FaceForensics++")
        json_split = "train.json" if partition == "train" else "test.json"
        df = pd.read_json(ff_root / "author_splits" / json_split)
        
        # if partition == "train":
        #     df = pd.read_json(ff_root / "author_splits" / "train.json")
        #     # df_val = pd.read_json(ff_root / "author_splits" / "val.json")
        #     # df = pd.concat([df_train, df_val], ignore_index=True)
        # else:
        #     df = pd.read_json(ff_root / "author_splits" / "test.json")
        fake_methods = {}
        for data in take_datasets: 
            fake_methods[data] = []
        # fake_methods = {"NeuralTextures": []}
        for _, row in df.iterrows():
            p1, p2 = str(row[0]).zfill(3), str(row[1]).zfill(3)
            pair_ids = [f"{p1}_{p2}.avi", f"{p2}_{p1}.avi"]
            # fake
            for pair in pair_ids:
                for method in fake_methods:
                    fp = get_video_path(ff_root, method, pair)
                    if os.path.exists(fp):
                        self.file_list.append({"path": fp, "label": 1})
            # real
            for vid in (p1, p2):
                rp = get_real_path(ff_root, vid)
                if os.path.exists(rp):
                    self.file_list.append({"path": rp, "label": 0})

        # -------------  build CLIP index --------------------------------
        self.clip_index: list[tuple[int, int, int, int]] = []  # (vid_idx, start, end, label)
        print("Indexing clips …")
        REAL=0
        FAKE=0
        for vid_idx, item in enumerate(self.file_list):
            try:
                vr = VideoReader(item["path"], ctx=cpu(0))
            except Exception as e:
                print("⛔", item["path"], "→", e)
                continue
            total = len(vr)
            # if partition != "train":
            #     total = self.clip_len
            label = self.file_list[vid_idx]["label"]
            for st in range(0, max(1, total - self.clip_len + 1), self.clip_stride):
                en = st + self.clip_len
                if label == 1:
                    FAKE+=1
                else:
                    REAL+=1
                self.clip_index.append((vid_idx, st, en, item["label"]))
        print(f"▶︎ {len(self.file_list)} videos → {len(self.clip_index)} clips, REAL = {REAL} | FAKE = {FAKE}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.clip_index)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        vid_idx, st, en, label = self.clip_index[idx]
        path = self.file_list[vid_idx]["path"]
        try:
            vr = VideoReader(path, ctx=cpu(0))
            frames_np = vr.get_batch(range(st, min(en, len(vr)))).asnumpy()
        except Exception:
            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self))

        # pad last clip if shorter
        if frames_np.shape[0] < self.clip_len:
            pad = np.zeros((self.clip_len - frames_np.shape[0], *frames_np.shape[1:]), dtype=frames_np.dtype)
            frames_np = np.concatenate([frames_np, pad], 0)

        video = equalize_frames(frames_np)  # (T, C, H, W)
        # video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255 
        
        video_aug = video
        if self.is_train:
            video = self._apply_video_aug_ori(video)
        else:
            video = self._apply_video_aug_val(video)
            

        # if self.is_train:
        video_aug = self._apply_video_aug(video)

        return video, video_aug, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    def _apply_video_aug_ori(self, vid: torch.Tensor) -> torch.Tensor:
        frames = [self.video_aug_ori(f.unsqueeze(0)).squeeze(0) for f in vid]
        return torch.stack(frames)

    def _apply_video_aug_val(self, vid: torch.Tensor) -> torch.Tensor:
        frames = [self.video_aug_val(f.unsqueeze(0)).squeeze(0) for f in vid]
        return torch.stack(frames)

    def _apply_video_aug(self, vid: torch.Tensor) -> torch.Tensor:
        frames = [self.video_aug(f.unsqueeze(0)).squeeze(0) for f in vid]
        return torch.stack(frames)
