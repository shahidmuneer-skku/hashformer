from __future__ import annotations
from collections import defaultdict
import os
import random
import traceback
from pathlib import Path
import imageio
import numpy as np
import cv2
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
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
from torchvision.io import write_video
from tqdm import tqdm
from PIL import Image, ImageEnhance

import tempfile

import imageio



class BlockShuffle(nn.Module):
    """Shuffle non-overlapping blocks of size (bh×bw) within each image."""
    def __init__(self, block_size=(32, 32)):
        super().__init__()
        self.bh, self.bw = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % self.bh == 0 and W % self.bw == 0, \
            "H and W must be divisible by block height/breadth"
        h_blocks, w_blocks = H // self.bh, W // self.bw

        # 1) reshape into blocks: (B, hb, bh, wb, bw, C)
        xb = x.view(B, C, h_blocks, self.bh, w_blocks, self.bw)
        xb = xb.permute(0, 2, 4, 1, 3, 5).contiguous()  
        # now (B, hb, wb, C, bh, bw)

        # 2) flatten block dims → (B, hb*wb, C, bh, bw)
        xb = xb.view(B, h_blocks * w_blocks, C, self.bh, self.bw)

        # 3) shuffle blocks independently per sample
        out = xb.clone()
        for i in range(B):
            idx = torch.randperm(h_blocks * w_blocks, device=x.device)
            out[i] = xb[i, idx]

        # 4) un-flatten & permute back
        out = out.view(B, h_blocks, w_blocks, C, self.bh, self.bw)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        # → (B, C, H, W)
        return out.view(B, C, H, W)
def _read_audio_segment(path: str, st_sample: int, en_sample: int, sr: int) -> torch.Tensor:
    frames = en_sample - st_sample
    try:
        audio, r_sr = sf.read(path, start=st_sample, frames=frames, dtype="float32", always_2d=False)
        if r_sr != sr:                           # rare — resample if file SR ≠ desired SR
            # audio = librosa.resample(audio, r_sr, sr)
            audio = librosa.resample(y=audio, orig_sr=r_sr, target_sr=sr)
    except Exception:                           # soundfile failed → safe fallback
        duration = frames / sr
        audio, _ = librosa.load(path,
                                sr=sr,
                                offset=st_sample / sr,
                                duration=duration,
                                )   # explicit so future librosa won’t complain
    if len(audio) < frames:                      # pad last chunk if needed
        audio = np.pad(audio, (0, frames - len(audio)))
    return torch.tensor(audio[:frames], dtype=torch.float32)
    

class RandomTemporalErasing(nn.Module):
    def __init__(self, erase_prob=0.5, num_frames_to_erase=1, erase_value=0.0):
        super().__init__()
        self.erase_prob = erase_prob
        self.num_frames_to_erase = num_frames_to_erase
        self.erase_value = erase_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [B, T, C, H, W]
        """
        if not self.training or random.random() > self.erase_prob:
            return x

        T, C, H, W = x.shape
        erase_indices = random.sample(range(T), min(self.num_frames_to_erase, T))
        for t in erase_indices:
            x[t] = self.erase_value
        return x


def perturb_saturation(frame: np.ndarray, factor: float) -> np.ndarray:
    img = Image.fromarray(frame)
    img = ImageEnhance.Color(img).enhance(factor)
    return np.array(img)

def perturb_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
    img = Image.fromarray(frame)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return np.array(img)

def perturb_blockwise(frame: np.ndarray, block_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def perturb_gaussian_noise(frame: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def perturb_gaussian_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return frame
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)

def perturb_jpeg_compression(frame: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

# ────────────────────────────────────────────────────────────
# Video‐level wrappers
# ────────────────────────────────────────────────────────────

def perturb_video_saturation(frames_np: np.ndarray, factor: float) -> np.ndarray:
    """
    frames_np: (T, H, W, C)
    """
    return np.stack([perturb_saturation(f, factor) for f in frames_np], axis=0)

def perturb_video_contrast(frames_np: np.ndarray, factor: float) -> np.ndarray:
    return np.stack([perturb_contrast(f, factor) for f in frames_np], axis=0)

def perturb_video_blockwise(frames_np: np.ndarray, block_size: int) -> np.ndarray:
    return np.stack([perturb_blockwise(f, block_size) for f in frames_np], axis=0)

def perturb_video_gaussian_noise(frames_np: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([perturb_gaussian_noise(f, sigma) for f in frames_np], axis=0)

def perturb_video_gaussian_blur(frames_np: np.ndarray, kernel_size: int) -> np.ndarray:
    return np.stack([perturb_gaussian_blur(f, kernel_size) for f in frames_np], axis=0)

def perturb_video_jpeg_compression(frames_np: np.ndarray, quality: int) -> np.ndarray:
    return np.stack([perturb_jpeg_compression(f, quality) for f in frames_np], axis=0)

def perturb_video_compression(
    frames_np: np.ndarray,
    crf: int = 23,
    codec: str = 'libx264',
    fps: float = 25.0
) -> np.ndarray:
    """
    frames_np: (T, H, W, C)
    returns: (T, H, W, C)
    """
    # write to temp MP4
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.close()
    writer = imageio.get_writer(
        tmp.name,
        fps=fps,
        codec=codec,
        ffmpeg_params=['-crf', str(crf), '-preset', 'medium']
    )
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()

    # read back
    reader = imageio.get_reader(tmp.name)
    out = []
    for im in reader:
        out.append(im)
    reader.close()
    os.remove(tmp.name)

    return np.stack(out, axis=0)

# build_file_list(self.base_dir, self.partition, take_datasets)

def build_file_list(base_dir, partition, take_datasets):
    # print("started to load data")
    features = {"landmarks":{}, "audio_path":{}}
    file_list=[]
    video_files = {}
    base_dir_v2="<path_to_dataset>deepspeak_v2"

    features_v2 = {"landmarks":{}, "audio_path":{}}
    video_files_v2 = {}
    for dirs in os.listdir(f"{base_dir}/dataset/"):
        subdir = f"{base_dir}/features/features_mtcnn/{dirs}"
        landmarks = f"{base_dir}/features/features_mtcnn/{dirs}"
        
        if not os.path.exists(subdir):
            continue
        for file in os.listdir(subdir):
            video_file = os.path.join(subdir,os.path.join(file,"crop_face_00.mp4"))
           
            # if(video_file is not None):
            #     print("This is not noe")
            # video_file = os.path.join(subdir,file)
            if not os.path.exists(video_file) and video_file == None: 
                continue
            if not os.path.exists(video_file.replace("mp4","wav")): 
                continue
            if not os.path.exists(video_file.replace("crop_face_00.mp4","landmarks.npy")): 
                continue
            file = file+".mp4"
            video_files[file]= video_file
            features["audio_path"][file]=os.path.join(video_file.replace("mp4","wav"))
            features["landmarks"][file]=os.path.join(landmarks,os.path.join(file.replace(".mp4",""),"landmarks.npy"))


        for dirs in os.listdir(f"{base_dir_v2}/dataset/"):
            subdir = f"{base_dir_v2}/dataset_features/{dirs}"
            landmarks = f"{base_dir_v2}/dataset_features/{dirs}"
            
            if not os.path.exists(subdir):
                continue
            for file in os.listdir(subdir):
                video_file = os.path.join(subdir,os.path.join(file,"cropped_face.mp4"))
            
                if not os.path.exists(video_file) and video_file == None: 
                    continue
                if not os.path.join(subdir,os.path.join(file,file.replace(".mp4","_audio.wav"))): 
                    continue
                if not os.path.exists(video_file.replace("crop_face_00.mp4","landmarks.npy")): 
                    continue
                file = file+".mp4"
                video_files_v2[file]= video_file
                features["audio_path"][file]=os.path.join(subdir,os.path.join(file,file.replace(".mp4","_audio.wav")))
                features_v2["landmarks"][file]=os.path.join(landmarks,os.path.join(file.replace(".mp4",""),"landmarks.npy"))


                
    train_test_split = pd.read_csv(f"{base_dir}/annotations-split-def.csv")
    real_annotations = pd.read_csv(f"{base_dir}/annotations-real.csv")
    fake_annotations = pd.read_csv(f"{base_dir}/annotations-fake.csv")
    
    train_test_split = train_test_split[train_test_split['split'] == partition]
    matching_real_annotations = real_annotations[real_annotations["identity"].isin(train_test_split["identity"])]#.head(10000)

    
    train_test_split = train_test_split[train_test_split['split'] == partition]
    matching_real_annotations = real_annotations[real_annotations["identity"].isin(train_test_split["identity"])]#.head(10000)

    
    train_test_split_v2 = pd.read_csv(f"{base_dir_v2}/annotations-split-def.csv")
    real_annotations_v2 = pd.read_csv(f"{base_dir_v2}/annotations-real.csv")
    fake_annotations_v2 = pd.read_csv(f"{base_dir_v2}/annotations-fake.csv")

    
    
    train_test_split_v2 = train_test_split_v2[train_test_split_v2['split'] == partition]
    matching_real_annotations_v2 = real_annotations_v2[real_annotations_v2["identity"].isin(train_test_split_v2["identity"])]#.head(10000)
    matching_fake_annotations_v2 = fake_annotations_v2[fake_annotations_v2["identity-source"].isin(train_test_split_v2["identity"])]
   
    if "1" in take_datasets:
        for index,row in matching_real_annotations.iterrows():
            # print(row)
            if video_files.get(row["video-file"]) is None:
                continue

            # print(row["video-file"])
            file_list.append({"path": video_files.get(row["video-file"]),
                                    "audio_path":features["audio_path"].get(row["video-file"]),
                                    "landmarks":features["landmarks"].get(row["video-file"]),
                                    "multi_label": "real",
                                    "label": 0,
                                    "data_label":1})
 
    if "REAL_TEST" in take_datasets:
        
        for index,row in matching_real_annotations.iloc[:400].iterrows():
            # print(row)
            if video_files.get(row["video-file"]) is None:
                continue

            # print(row["video-file"])
            file_list.append({"path": video_files.get(row["video-file"]),
                                    "audio_path":features["audio_path"].get(row["video-file"]),
                                    "landmarks":features["landmarks"].get(row["video-file"]),
                                    "multi_label": "real",
                                    "label": 0,
                                    "data_label":1})

                                    
    if "REAL_FACESWAP" in take_datasets:
        
        for index,row in matching_real_annotations.iloc[:730].iterrows():
            # print(row)
            if video_files.get(row["video-file"]) is None:
                continue

            # print(row["video-file"])
            file_list.append({"path": video_files.get(row["video-file"]),
                                    "audio_path":features["audio_path"].get(row["video-file"]),
                                    "landmarks":features["landmarks"].get(row["video-file"]),
                                    "multi_label": "real",
                                    "label": 0,
                                    "data_label":1})
               
   
            
    if "2" in take_datasets:
        matching_fake_annotations = fake_annotations[fake_annotations["identity-source"].isin(train_test_split["identity"])]
        for index,row in matching_fake_annotations.iterrows():
            
            if video_files.get(row["video-file"]) is None:
                continue
            file_list.append({"path": video_files.get(row["video-file"]),
                                    "audio_path":features["audio_path"].get(row["video-file"]),
                                    "landmarks":features["landmarks"].get(row["video-file"]),
                                    "multi_label": row["engine"],
                                    "label": 1,    
                                    "data_label":2})
        
        
    if "v_real" in take_datasets:
       
        for index,row in matching_real_annotations_v2.iterrows():
            if video_files_v2.get(row["video-file"]) is None:
                continue
       
            file_list.append({"path": video_files_v2.get(row["video-file"]),
                                    "landmarks":features_v2["landmarks"].get(row["video-file"]),
                                    "audio_path":features["audio_path"].get(row["video-file"]),
                                    "label": 0,
                                    "multi_label": "real",
                                    "data_label":1})
            
    if "v_fake" in take_datasets:
        for index,row in matching_fake_annotations_v2.iterrows():
            
            if video_files_v2.get(row["video-file"]) is None:
                continue
            file_list.append({"path": video_files_v2.get(row["video-file"]),
                                    "audio_path":features["audio_path"].get(row["video-file"]),
                                    "landmarks":features_v2["landmarks"].get(row["video-file"]),
                                    "multi_label": row["engine"],
                                    "label": 1,    
                                    "data_label":2})

                                                                

    if "TIMIT_HQ" in take_datasets:
        root_dir = "<path_to_dataset>DeepfakeTIMIT/features_mtcnn/higher_quality"
            # Traverse the directory structure
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    audio_path = os.path.join(root, file.replace("crop_face_00.mp4", "cropped_video.wav"))
                    landmarks_path = os.path.join(root, file.replace("crop_face_00.mp4", "landmarks.npy"))

                    # Check if the video and audio files exist
                    if os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(audio_path):
                        # Label as fake if "fake" is in the file name, otherwise real
                        label = 1
                        file_list.append({
                            "path": video_path,
                            "audio_path":audio_path,
                            "landmarks": landmarks_path,
                            "label": label,
                            "data_label":3
                        })


    if "TIMIT_LQ" in take_datasets:
        root_dir = "<path_to_dataset>DeepfakeTIMIT/features_mtcnn/low_quality"
            # Traverse the directory structure
        total = 0 
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".mp4"):
                    total+=1
                    # if total>100:
                    #     break
                    # Construct paths
                    video_path = os.path.join(root, file)
                    audio_path = os.path.join(root, file.replace("crop_face_00.mp4", "cropped_video.wav"))
                    landmarks_path = os.path.join(root, file.replace("crop_face_00.mp4", "landmarks.npy"))

                    # Check if the video and audio files exist
                    if os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(audio_path):
                        # Label as fake if "fake" is in the file name, otherwise real
                        label = 1

                        # Append to file list
                        file_list.append({
                            "path": video_path,
                            "audio_path":audio_path,
                            "landmarks": landmarks_path,
                            "label": label,
                            "data_label":3
                        })
        
            

                    
    if partition=="test":
        if "FakeAVCeleb" in take_datasets:
            fakeavceleb_path = "<path_to_dataset>FakeAVCeleb_v1.2/" 
            csv_path = f'{fakeavceleb_path}meta_data.csv'  # Change this to the actual path of your CSV file
            data = pd.read_csv(csv_path)
            if partition=="test":
                # data = data.loc[data["type"] == "FakeVideo-RealAudio"].iloc[17000:] # for cross-dataset generalization 
                data = data.iloc[17000:] # for cross-dataset generalization 

           

            count_real = 0
            for index, row in data.iterrows():
                path = os.path.join(fakeavceleb_path,row['Unnamed: 9'].replace("FakeAVCeleb/", "landmark_features/features_mediapipe/"))
                feature_path = os.path.join(fakeavceleb_path,row['Unnamed: 9'].replace("FakeAVCeleb/", "landmark_features/features_mediapipe/"))
                

                video_path = os.path.join(path, row["path"].replace(".mp4","/crop_face_00.mp4"))
                audio_path = os.path.join(path, row["path"].replace(".mp4",f"/{video}_audio.wav"))
                
                landmark_features = os.path.join(feature_path, row["path"].replace(".mp4", "/landmarks_00.npy"))
            
                label = 0 if row["type"]=="real" else 1   
                if(row["type"]=="real"):
                    # real_samples = real_samples+1
                    continue
                # else:
                    # fake_samples = fake_samples+1                
                if not os.path.exists(video_path) and not os.path.exists(landmark_features):
                    # print(f"file does not existss {video_path}")
                    continue

                file_list.append({
                                        "path":video_path,
                                        "audio_path":audio_path,
                                        "landmarks":landmark_features,
                                        "deepspeak":0,
                                        "label":label,
                                        "data_label":4})  # 1 for real

   

    if partition=="train":
        
        if "FakeAVCeleb" in take_datasets:
            fakeavceleb_path = "<path_to_dataset>FakeAVCeleb_v1.2/" 
            csv_path = f'{fakeavceleb_path}meta_data.csv'  # Change this to the actual path of your CSV file
            
            data = pd.read_csv(csv_path)
            # data = data[data["method"]=="faceswap"]# for testing on faceforensics++ faceswap ds
            data = data[:17000]
            
            
            count_real = 0
            for index, row in data.iterrows():
                path = os.path.join(fakeavceleb_path,row['Unnamed: 9'].replace("FakeAVCeleb/", "landmark_features/features_mediapipe/"))
                feature_path = os.path.join(fakeavceleb_path,row['Unnamed: 9'].replace("FakeAVCeleb/", "landmark_features/features_mediapipe/"))
                
                # exit()
                video_path = os.path.join(path, row["path"].replace(".mp4","/crop_face_00.mp4"))
                audio_path = os.path.join(path, row["path"].replace(".mp4",f"/{video}_audio.wav"))
                landmark_features = os.path.join(feature_path, row["path"].replace(".mp4", "/landmarks_00.npy"))
            
                label = 0 if row["type"]=="real" else 1   
                # if(row["type"]=="real"):
                #     # real_samples = real_samples+1
                #     continue
                # else:
                    # fake_samples = fake_samples+1                
                if os.path.exists(video_path) and os.path.exists(landmark_features)  :
                    # print(f"file does not existss {video_path}")
                        
                    file_list.append({
                                            "path":video_path, 
                                            "audio_path":audio_path,
                                            "landmarks":landmark_features,
                                            "deepspeak":0,
                                            "label":label,
                                            "data_label":4})  # 1 for real


    if "DFDC" in take_datasets:
        root_dir = "<path_to_dataset>DFDC-Official/dfdc_features/"
        dataset = pd.read_json(os.path.join(root_dir,"dataset.json"))
        dataset = dataset.T
        if partition=="test":
             dataset = dataset[:400]
        for index,row in dataset.iterrows():
            if(os.path.exists(os.path.join(root_dir,index.replace(".mp4","/crop_face_00.mp4"))) 
            and os.path.exists(os.path.join(root_dir,index.replace(".mp4",f"/{video}_audio.wav")))
            and os.path.exists(os.path.join(root_dir,index.replace(".mp4","/landmarks_00.npy")))
            ):
                # print(index,row["label"])
                if os.path.exists(os.path.join(root_dir,index.replace(".mp4",f"/{video}_audio.wav"))):
                    file_list.append({"path": os.path.join(root_dir,index.replace(".mp4","/crop_face_00.mp4")),
                            "audio_path":os.path.join(root_dir,index.replace(".mp4",f"/{video}_audio.wav")),
                            "landmarks": os.path.join(root_dir,index.replace(".mp4","/landmarks_00.npy")),
                            "label": 1 if row["label"]=="fake" else 0,"data_label":5 if row["label"]=="fake" else 1})

          
    if "KoDF" in take_datasets:
        root_dir = "<path_to_dataset>KoDF/features_mtcnn/"
        dataset = pd.read_csv(os.path.join(root_dir,"validate_meta_data","fake.csv"))
        dataset["folder2"] = dataset["folder2"].fillna(0)
        if partition=="test":
            dataset = dataset.iloc[:400]

        for index,row in dataset.iterrows():
            # print(os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4","/crop_face_00.mp4")))
            if(os.path.exists(os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4","/crop_face_00.mp4")))
            and os.path.exists(os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4",f"/{video}_audio.wav")))
                and os.path.exists(os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4","/landmarks_00.npy")))
            ):
                # print(index,row["video_label"])
                file_list.append({"path": os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4","/crop_face_00.mp4")),
                "audio_path": os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4",f"/{video}_audio.wav")),
                "landmarks": os.path.join(root_dir,row["folder"],str(int(row["folder2"])),row["filename"].replace(".mp4","/landmarks_00.npy")),
                "label": row["label"],
                "data_label":6 })

    if "CUSTOM_DATASET" in take_datasets:
        root_dir = "<path_to_dataset>/custom_dataset_features"
        total = 0 

        for directory in os.listdir(root_dir):
            for video in os.listdir(os.path.join(root_dir,directory)):
                video_path = os.path.join(root_dir, directory,video,"crop_face_00.mp4")
                audio_path = os.path.join(root_dir, directory,video,f"{video}_audio.wav")
                landmarks_path = os.path.join(root_dir, directory,video,f"landmarks_00.npy")
                print(video_path, audio_path)
                # Check if the video and audio files exist
                if os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(landmarks_path):
                    # Label as fake if "fake" is in the file name, otherwise real
                    
                    label = 1 if directory == "fake" else 0
                    
                    # Append to file list
                    file_list.append({
                        "path": video_path,
                        "audio_path":audio_path,
                        "landmarks": landmarks_path,
                        "label": label,
                        "data_label":3
                    })

    print(f"Total files in filelist of  {partition} are {len(file_list)} {take_datasets}")
    return file_list

def _equalize_frames(frames_np: np.ndarray) -> torch.Tensor:
    """Histogram‑equalise luminance channel of a (T, H, W, C) RGB numpy array."""
    eq = []
    for f in frames_np:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        eq.append(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
    eq = np.stack(eq, 0)  # (T, H, W, C)
    return torch.from_numpy(eq).permute(0, 3, 1, 2).float() / 255  # (T, C, H, W)


def _normalize_landmarks_standardize(landmarks: np.ndarray | torch.Tensor):
    lm = landmarks if isinstance(landmarks, np.ndarray) else landmarks.numpy()
    mean = lm.mean(0)
    std = lm.std(0); std[std == 0] = 1
    lm = (lm - mean) / std
    return torch.from_numpy(lm.astype(np.float32)) if isinstance(landmarks, np.ndarray) else landmarks


# -------------------------------------------------------------
# --------------------- main dataset class --------------------
# -------------------------------------------------------------

class HashFormerDataset(Dataset):
    """Clip‑level dataloader that mirrors FaceswapVideoDataset behaviour."""

    def __init__(
        self,
        base_dir: str | Path,
        partition: str = "train",
        take_datasets: str = "1,2,3,4,5,6",
        clip_len_sec: int = 1,
        clip_stride_sec: int | None = 1,
        fps: int = 16,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        visual_preturb={
            "saturation": None,
            "contrast": None,
            "blockwise": None,
            "gaussian_noise": None,
            "gaussian_blur": None,
            "jpeg": None,
            "jpeg_compression": None,
            "video_compression": None
        }
    ):
    
        super().__init__()
        # print(take_datasets)
        assert partition in {"train", "dev", "test"}, "partition must be one of train/dev/test"
        self.partition = partition
        self.base_dir = Path(base_dir)
        self.is_train = partition == "train"
        self.visual_preturb=visual_preturb
        self.temporal_aug = RandomTemporalErasing(erase_prob=0.2, num_frames_to_erase=1)

        # --------------  clip parameters ---------------------
        self.fps = fps
        self.clip_len = clip_len_sec * fps
        self.clip_stride = (clip_stride_sec or clip_len_sec) * fps
        self.sample_rate = sample_rate
        self.clip_len_samples = clip_len_sec * sample_rate
        self.clip_stride_samples = (clip_stride_sec or clip_len_sec) * sample_rate

        # --------------  augs & tokenizer --------------------
        self.audio_aug = self._build_audio_aug()
        self.video_aug_ori, self.video_aug = self._build_video_aug()
        
        self.video_aug_val = K.AugmentationSequential(
            K.Resize((224, 224)),
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # --------------  heavy: build file_list ---------------
        self.file_list: list[dict] = []  # same structure as original → {path, label, landmarks, data_label}
        self._build_file_list(take_datasets)

        # --------------  index every clip ---------------------
        self.clip_index: list[tuple[int, int, int, int, int]] = []  # (file_idx, st_f, en_f, st_s, en_s)
        print("⇢ Indexing clips … (this might take a minute the first time)")
        REAL = 0
        FAKE = 0
        
        REAL_CLIPS = []
        FAKE_CLIPS = []
        CLASS_COUNTS = defaultdict(int)
        for file_idx, item in enumerate(self.file_list):
            try:
                vr = VideoReader(item["path"], ctx=cpu(0))
                n_frames = len(vr)
            except Exception:
                print("This file is not read")
                continue
            # simple assumption: audio has *perfect* sync (16 kHz) with video fps
            if partition=="CUSTOM_DATASET":
                n_frames = int(n_frames/3)
            else:
                n_frames = self.clip_len
            # if partition=="test":
            
            # n_frames = self.clip_len
            # n_frames = max(0, (n_frames - self.clip_len) // 2)

            # n_frames = n_frames/2
            # n_samples = int(n_frames / self.fps * self.sample_rate)

            for st_f in range(0, max(1, n_frames - self.clip_len + 1), self.clip_stride):
                if item["label"]==0:
                    REAL+=1
                else:
                    FAKE+=1

                en_f = st_f + self.clip_len
                st_s = int(st_f / self.fps * self.sample_rate)
                en_s = st_s + self.clip_len_samples
                self.clip_index.append((file_idx, st_f, en_f, st_s, en_s))
                label_str = item.get("multi_label", "unknown")
                CLASS_COUNTS[label_str] += 1

        print(f"▶︎ {len(self.file_list)} videos → {len(self.clip_index)} clips (len={clip_len_sec}s) where {REAL} are real and {FAKE} are fakes")
        print("▶︎ Class-wise clip counts:")
        for cls_name in sorted(CLASS_COUNTS):
            print(f"   {cls_name:10} = {CLASS_COUNTS[cls_name]}")


        all_labels = [item["multi_label"] for item in self.file_list if "multi_label" in item]
        self.class_names = sorted(set(all_labels))
        self.class_numbers=len(self.class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}


        # --------------  MFCC params --------------------------
        self.n_mfcc = n_mfcc

    # -----------------------------------------------------
    #                initialisation helpers               
    # -----------------------------------------------------
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
        

    def _build_video_aug(self):
        video_aug_ori = K.AugmentationSequential(
            K.Resize((224, 224)),
            K.RandomHorizontalFlip(p=0.3),
            K.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.9, 1.1), p=0.3),
            K.ColorJitter(0.2, 0.2, 0.2, 0.0, p=0.1),
            # K.RandomErasing(scale=(0.02, 0.1), ratio=(0.2, 1.6), p=0.1),
        )
        video_aug = K.AugmentationSequential(
    # resize + flip
            K.Resize((224, 224)),
            K.RandomHorizontalFlip(p=0.9),

            # color jitter + explicit saturation & contrast
            K.ColorJitter(0.2, 0.2, 0.2, 0.0, p=0.9),
            K.RandomSaturation(0.5, p=0.9),
            K.RandomContrast(0.5, p=0.9),

            # blockwise spatial shuffle
            K.RandomChannelShuffle( p=0.9),
            BlockShuffle(block_size=(32, 32)),
            # blurs
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.9),
            K.RandomMotionBlur(kernel_size=3, angle=15.0, direction=0.5, p=0.9),
            K.RandomBoxBlur(kernel_size=(5, 5), p=0.9),

            # geometry + erasing
            K.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.9, 1.1), p=0.6),
            K.RandomErasing(scale=(0.02, 0.33), ratio=(0.5, 3.6), p=0.9),

            same_on_batch=True,
        )
        return video_aug_ori, video_aug

    def _build_file_list(self, take_datasets: str):
        """⚠️  *UNCHANGED*: reuse the (long) logic from the original implementation.
        For brevity we import it instead of inlining. Feel free to copy‑paste
        the original `_build_file_list` body here if you prefer a single file."""
        # from mmmoderator_original import build_file_list  # type: ignore
        self.file_list = build_file_list(self.base_dir, self.partition, take_datasets)
     

    # -----------------------------------------------------
    #                        API                          
    # -----------------------------------------------------
    def __len__(self):
        return len(self.clip_index)

    def __getitem__(self, idx):
        try:
            file_idx, st_f, en_f, st_s, en_s = self.clip_index[idx]
            item = self.file_list[file_idx]
            audio_path = item["audio_path"]
            lm_item = item["landmarks"]
           
            try:
                # --------- VIDEO (T, C, H, W) ---------
                vr = VideoReader(item["path"], ctx=cpu(0))
                frames_np = vr.get_batch(range(st_f, min(en_f, len(vr)))).asnumpy()
            except Exception:
                traceback.print_exc(); return self.__getitem__((idx + 1) % len(self))
        
            # pad last clip if shorter (should not happen except end‑of‑file)
            if frames_np.shape[0] < self.clip_len:
                pad = np.zeros((self.clip_len - frames_np.shape[0], *frames_np.shape[1:]), dtype=frames_np.dtype)
                frames_np = np.concatenate([frames_np, pad], 0)
            frames_np = frames_np[:self.clip_len]
            video = frames_np 
            if self.visual_preturb["saturation"] is not None:
                video   = perturb_video_saturation(frames_np, factor=self.visual_preturb["saturation"])
                
            if self.visual_preturb["contrast"] is not None:
                video   = perturb_video_contrast(frames_np, factor=self.visual_preturb["contrast"])
                
            if self.visual_preturb["blockwise"] is not None:
                video   = perturb_video_blockwise(frames_np, block_size=self.visual_preturb["blockwise"])

            if self.visual_preturb["gaussian_noise"] is not None:
                video = perturb_video_gaussian_noise(frames_np, sigma=self.visual_preturb["gaussian_noise"] )
                
            if self.visual_preturb["gaussian_blur"] is not None:
                video  = perturb_video_gaussian_blur(frames_np, kernel_size=self.visual_preturb["gaussian_blur"])

            if self.visual_preturb["jpeg_compression"] is not None:
                video   = perturb_video_jpeg_compression(frames_np, quality=self.visual_preturb["jpeg_compression"])

            if self.visual_preturb["video_compression"] is not None:
                video    = perturb_video_compression(frames_np, crf=28)


            video = _equalize_frames(video)                           # weak std ↑
            # video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255 

            video_aug = video.detach().clone()
            if self.is_train:
                video_ori = self._apply_video_aug_ori(video)
            else:
                video_ori = self._apply_video_aug_val(video)
                
                                                  # strong aug view default = weak
            if self.is_train:
                video_aug = self._apply_video_aug(video)
                video_aug = self.temporal_aug(video_aug)
                
            # self.save_video_imageio("debug_videos/000001_ori.mp4", video_ori)
            # self.save_video_imageio("debug_videos/000001_aug.mp4", video_aug)
            # exit()

            # --------- AUDIO (L,) ---------------------
            # audio_path = item["path"].replace("crop_face_00.mp4", "cropped_video.wav")
            if not os.path.exists(audio_path):
                # Faceswap data (no audio) → dummy zeros
                audio = torch.zeros(self.clip_len_samples)
            else:
                duration = (en_s - st_s) / self.sample_rate
                # audio, _ = librosa.load(audio_path, sr=self.sample_rate,
                #                         offset=st_s / self.sample_rate,
                #                         duration=duration)
                audio = _read_audio_segment(audio_path, st_s, en_s, self.sample_rate)
                if len(audio) < self.clip_len_samples:
                    audio = np.pad(audio, (0, self.clip_len_samples - len(audio)))
                audio = audio[: self.clip_len_samples]

            au = audio.detach().clone()
            audio = torch.tensor(au, dtype=torch.float32)

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

            # --------- LANDMARKS ---------------------
            # lm = torch.zeros_like(video[:, :2])  # (T, 2, H,W)
           
            lm = np.load(lm_item,  allow_pickle=True)
            if not isinstance(lm, np.ndarray):
                # log & skip this example, or replace with zeros:
                print(f"❗️ malformed landmarks at {lm_item}, skipping clip")
                raise ValueError(f"❗️ malformed landmarks at {lm_item}, skipping clip")
            lm = lm.astype(np.float32)
            lm = _normalize_landmarks_standardize(lm)
            lm = self._pad_or_trim(lm, self.clip_len)  # (T, …)
            # --------- optional optical flow ---------- (dummy example)
            optical_flow = torch.zeros_like(video[:, :2])  # (T, 2, H, W)
            txt = self.tokenizer("dummy text", max_length=20, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
            multi_label_str = item.get("multi_label", "real")
            multi_label_idx = self.class_to_idx.get(multi_label_str, 0)
            return (
                mfcc,
                mfcc_aug,
                audio,
                audio_aug,
                video_ori,
                video_aug,
                txt.float(),
                lm.float(),
                torch.tensor(item["label"], dtype=torch.long),
                os.path.basename(item["path"]),
                torch.tensor(item["data_label"], dtype=torch.long),
                torch.tensor(multi_label_idx, dtype=torch.long),
                optical_flow.float(),
            )

        except Exception:

            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self))

    # -----------------------------------------------------
    #                 small util functions                 
    # -----------------------------------------------------
    def _apply_video_aug_ori(self, vid: torch.Tensor) -> torch.Tensor:
        frames = [self.video_aug_ori(f.unsqueeze(0)).squeeze(0) for f in vid]
        return torch.stack(frames)
    def _apply_video_aug_val(self, vid: torch.Tensor) -> torch.Tensor:
        frames = [self.video_aug_val(f.unsqueeze(0)).squeeze(0) for f in vid]
        return torch.stack(frames)

    def _apply_video_aug(self, vid: torch.Tensor) -> torch.Tensor:
        frames = [self.video_aug(f.unsqueeze(0)).squeeze(0) for f in vid]
        return torch.stack(frames)

    def _extract_mfcc(self, audio: torch.Tensor):
        mfcc = librosa.feature.mfcc(
            y=audio.cpu().numpy(), sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=512, n_fft=2048
        )
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        return torch.tensor(librosa.util.normalize(mfcc), dtype=torch.float32)

    def _pad_or_trim(self, x: torch.Tensor, T: int):
        if x.shape[0] < T:
            pad = torch.zeros((T - x.shape[0], *x.shape[1:]), dtype=x.dtype)
            x = torch.cat([x, pad], 0)
        else:
            x = x[:T]
        return x

    def save_video_imageio(self,path, video_tensor, fps=25):
        video_np = (
            video_tensor.detach().cpu().clamp(0, 1) * 255
        ).byte().permute(0, 2, 3, 1).numpy()  # (T, H, W, C)

        writer = imageio.get_writer(path, fps=fps)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()