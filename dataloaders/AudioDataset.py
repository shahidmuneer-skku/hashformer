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

import soundfile as sf
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
def _read_audio_segment(path, frames, st_sample) -> torch.Tensor:
    # frames = en_sample - st_sample
    sr=st_sample
    try:
        audio, r_sr = sf.read(path, start=st_sample, frames=frames, dtype="float32", always_2d=False)
        if r_sr != sr:                           # rare — resample if file SR ≠ desired SR
            # audio = librosa.resample(audio, r_sr, sr)
            audio = librosa.resample(y=audio, orig_sr=r_sr, target_sr=sr)
    except Exception:                           # soundfile failed → safe fallback
        duration = frames 
        audio, _ = librosa.load(path,
                                offset=st_sample ,
                                duration=duration,
                                )   # explicit so future librosa won’t complain
    if len(audio) < frames:                      # pad last chunk if needed
        audio = np.pad(audio, (0, frames - len(audio)))
    return torch.tensor(audio[:frames], dtype=torch.float32)
    

class AudioDataset(Dataset):
    """Every item is a fixed‑length clip (default 3 s)."""

    def __init__(
        self,
        base_dir: str | Path,
        partition: str = "train",
        clip_len_sec: int = 1,
        clip_stride_sec: int | None = 1,
        fps: int = 16,
        audio_preturb={},
        take_datasets=""
    ):
        super().__init__()
        assert partition in {"train", "dev", "test"}
        self.partition = partition
        self.base_dir = Path(base_dir)
        self.is_train = partition == "train"
        self.fps = fps
        self.clip_len = clip_len_sec * fps
        self.clip_stride = (clip_stride_sec or clip_len_sec) * fps
        self.audio_aug = self._build_audio_aug()
        # -------------  augs & tokenizer (same as before) ----------------
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        sample_rate = 16000
        self.fps = fps
        self.clip_len = clip_len_sec * fps
        self.clip_stride = (clip_stride_sec or clip_len_sec) * fps
        self.sample_rate = sample_rate
        self.clip_len_samples = clip_len_sec * sample_rate
        self.clip_stride_samples = (clip_stride_sec or clip_len_sec) * sample_rate
        self.duration = sample_rate*clip_len_sec
        # -------------  build list of video files (unchanged) ------------
        self.file_list: list[dict] = []  # each {path, label}
        
        self.n_mfcc = 40
        # AVSpoof2021
        # base_dir = "<path_to_dataset>/ASVspoof2021/ASVspoof2021_DF_eval/"

        # spoof_entries = []
        # bonafide_entries = []

        # with open(f"{base_dir}trial_metadata.txt", "r") as f:  # replace with your actual file name
        #     for line in f:
        #         parts = line.strip().split()
        #         if len(parts) < 6:  # skip malformed lines
        #             continue
        #         label = parts[5].lower()
        #         if label == "spoof":
        #             spoof_entries.append(line.strip())
        #         elif label == "bonafide":
        #             bonafide_entries.append(line.strip())

        # # Save to separate files (optional)
        # with open("spoof_entries.txt", "w") as f:
        #     f.write("\n".join(spoof_entries))

        # with open("bonafide_entries.txt", "w") as f:
        #     f.write("\n".join(bonafide_entries))

        # print(f"Total Spoof: {len(spoof_entries)}")
        # print(f"Total Bonafide: {len(bonafide_entries)}")
        # exit()
        
        # AVSpoof2019

        base_dir = "<path_to_dataset>/ASVspoof2019/dataset/LA/"
        
        max_spoof = 100000000
        max_bonafide = 100000000
        spoof_count = 0
        bonafide_count = 0

        spoof_entries = []
        bonafide_entries = []

        with open(f"{base_dir}ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt", "r") as f:
            for line in f:
                if spoof_count >= max_spoof and bonafide_count >= max_bonafide:
                    break  # stop once both limits are reached

                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                label = parts[2].lower()
                label_2 = parts[3].lower()
                path = os.path.join(base_dir, "ASVspoof2019_LA_eval", "flac", f"{parts[1]}.flac")

                if not os.path.exists(path):
                    continue

                # --- Check first label ---
                if label == "spoof" and spoof_count < max_spoof:
                    spoof_entries.append(line.strip())
                    self.file_list.append({"path": path, "label": 1})
                    spoof_count += 1
                elif label == "bonafide" and bonafide_count < max_bonafide:
                    bonafide_entries.append(line.strip())
                    self.file_list.append({"path": path, "label": 0})
                    bonafide_count += 1

                # --- Check second label ---
                if spoof_count >= max_spoof and bonafide_count >= max_bonafide:
                    break  # stop early after adding first label

                if label_2 == "spoof" and spoof_count < max_spoof:
                    spoof_entries.append(line.strip())
                    self.file_list.append({"path": path, "label": 1})
                    spoof_count += 1
                elif label_2 == "bonafide" and bonafide_count < max_bonafide:
                    bonafide_entries.append(line.strip())
                    self.file_list.append({"path": path, "label": 0})
                    bonafide_count += 1

        print(f"Total Spoof: {len(spoof_entries)}")
        print(f"Total Bonafide: {len(bonafide_entries)}")
 
      
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.file_list)

    def _build_audio_aug(self):
        return (
            AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=0.9),
            SevenBandParametricEQ(p=0.9),
            PitchShift(min_semitones=-2, max_semitones=+2, p=0.9),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.9),
            Shift(p=0.5),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.9),
            Normalize(p=0.6),
            PolarityInversion(p=0.9),
            BandPassFilter(min_center_freq=150.0, max_center_freq=3500.0, p=0.9),
        )

        
    def _extract_mfcc(self, audio: torch.Tensor):
        mfcc = librosa.feature.mfcc(
            y=audio.cpu().numpy(), sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=512, n_fft=2048
        )
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        return torch.tensor(librosa.util.normalize(mfcc), dtype=torch.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        path = self.file_list[idx]["path"]
        label = self.file_list[idx]["label"]

        if not os.path.exists(path):
                # Faceswap data (no audio) → dummy zeros
            audio = torch.zeros(self.clip_len_samples)
        else:
            # audio, _ = librosa.load(audio_path, sr=self.sample_rate,
            #                         offset=st_s / self.sample_rate,
            #                         duration=duration)
            audio = _read_audio_segment(path, self.duration ,self.sample_rate)
            if len(audio) < self.clip_len_samples:
                audio = np.pad(audio, (0, self.clip_len_samples - len(audio)))
            audio = audio[: self.clip_len_samples]
            

        # au = audio.detach().clone()
        # audio = torch.tensor(au, dtype=torch.float32)

        # strong audio aug (SimCLR‑style) – keep original too
        audio_aug = audio.detach().clone()
        if self.is_train and audio.numel():
            audio_np = audio.cpu().numpy()
            for aug in self.audio_aug:
                audio_np = aug(samples=audio_np, sample_rate=self.sample_rate)
            audio_aug = torch.tensor(audio_np, dtype=torch.float32)

        # --------- MFCCs -------------------------
        mfcc = self._extract_mfcc(audio)
        mfcc_aug = self._extract_mfcc(audio_aug)
        
        return audio, audio_aug, torch.tensor(label, dtype=torch.long)

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
