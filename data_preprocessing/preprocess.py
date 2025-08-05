import os, glob, cv2, logging, subprocess, multiprocessing as mp, warnings, ffmpeg
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from scipy.signal import resample_poly
from scipy.io.wavfile import write as wav_write
from tqdm import tqdm
import mediapipe as mp_face
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- CONFIG ----------------
MAX_FRAMES = 1_377      # process up to this many frames
CROP_SIZE = 224         # output crop size

# ---------------- UTILITIES --------------
def setup_file_logger(path: str):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh   = logging.FileHandler(path); fh.setFormatter(fmt)
    if all(getattr(h, "baseFilename", None) != fh.baseFilename for h in root.handlers):
        root.addHandler(fh)

def save_wav(audio_f32: np.ndarray, sr: int, out: str):
    wav_write(out, sr, (audio_f32 * 32767).astype(np.int16))
    logging.info(f"Audio saved → {out}")

def read_audio_ffmpeg(video_path: str, target_sr: int = 16_000) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-vn", "-ac", "1", "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        logging.warning("ffmpeg audio extraction failed.")
        return None
    audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
    try:
        sr0 = int(ffmpeg.probe(video_path, select_streams="a")["streams"][0]["sample_rate"])
    except Exception:
        sr0 = target_sr
    if sr0 != target_sr:
        g = np.gcd(sr0, target_sr)
        audio = resample_poly(audio, target_sr // g, sr0 // g)
    return audio

def has_audio_stream(video_path: str) -> bool:
    try:
        info = ffmpeg.probe(video_path)
        return any(s.get("codec_type") == "audio" for s in info["streams"])
    except Exception:
        return False

def init_facemesh(max_faces: int = 5):
    return mp_face.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.4,
    )

# ---------------- SINGLE VIDEO -----------
def process_single_video(args):
    video_path, save_root, device = args
    os.makedirs(save_root, exist_ok=True)
    log_path = os.path.join(save_root, "processing.log")
    setup_file_logger(log_path)
    logging.info(f"[{device}] ▶ {video_path}")

    # --- load video ---
    try:
        clip = VideoFileClip(video_path, audio=False)  # audio handled separately
    except Exception as e:
        logging.error(f"MoviePy error: {e}")
        return

    fps = clip.fps or 25
    face_mesh = init_facemesh()

    face_frames: dict[int, list[np.ndarray]] = {}
    face_lmarks: dict[int, list[list[tuple[int, int]]]] = {}

    with face_mesh:
        for idx, frame_rgb in enumerate(clip.iter_frames(dtype="uint8", fps=fps)):
            if idx >= MAX_FRAMES:
                break

            results = face_mesh.process(frame_rgb)
            if not results.multi_face_landmarks:
                continue

            h, w, _ = frame_rgb.shape
            for fid, lmset in enumerate(results.multi_face_landmarks):
                pts = [(int(p.x * w), int(p.y * h)) for p in lmset.landmark]
                face_lmarks.setdefault(fid, []).append(pts)

                xs, ys = zip(*pts)
                x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, w)
                y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, h)
                crop_bgr = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                face_frames.setdefault(fid, []).append(crop_bgr)

    clip.close()

    # --- save crops & landmarks (one .npy per face, all frames in it) ---
    if not face_frames:
        logging.warning("No faces detected.")
    for fid, frames in face_frames.items():
        if not frames:
            continue

        lm_arr = np.array(face_lmarks[fid], dtype=np.int32)  # (num_frames, 468, 2)
        np.save(os.path.join(save_root, f"landmarks_{fid:02d}.npy"), lm_arr)
        logging.info(f"Saved landmarks for face {fid}")

        seq = [cv2.cvtColor(cv2.resize(f, (CROP_SIZE, CROP_SIZE)), cv2.COLOR_BGR2RGB)
               for f in frames]
        out_clip = ImageSequenceClip(seq, fps=fps)
        out_path = os.path.join(save_root, f"crop_face_{fid:02d}.mp4")
        out_clip.write_videofile(out_path,
                                 codec="libx264",
                                 audio=False,
                                 preset="ultrafast",
                                 verbose=False,
                                 logger=None)
        out_clip.close()
        logging.info(f"Saved cropped video {out_path}")

    # --- save audio once per video ---
    if has_audio_stream(video_path):
        wav_path = os.path.join(save_root,
                                os.path.splitext(os.path.basename(video_path))[0] + "_audio.wav")
        audio = read_audio_ffmpeg(video_path)
        if audio is not None:
            save_wav(audio, 16_000, wav_path)
        else:
            try:
                (ffmpeg.input(video_path)
                       .output(wav_path, ar=16_000, ac=1, acodec="pcm_s16le")
                       .overwrite_output()
                       .run(quiet=True))
                logging.info(f"Fallback audio → {wav_path}")
            except Exception as e:
                logging.warning(f"Audio fallback failed: {e}")

# ---------------- MULTIPROCESS -----------
def worker_process(args):
    gpu_id, todo = args
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    for vp, out in tqdm(todo, desc=f"GPU {gpu_id}", position=gpu_id):
        process_single_video((vp, out, device))

def collect_tasks(base_dir: str, output_dir: str):
    tasks = []
    for root, _, _ in os.walk(base_dir):
        for mp4 in glob.glob(os.path.join(root, "*.mp4")):
            rel = os.path.relpath(root, base_dir)
            out_dir = os.path.join(output_dir, rel,
                                   os.path.splitext(os.path.basename(mp4))[0])
            if not os.path.exists(out_dir):
                tasks.append((mp4, out_dir))
    return tasks

def main():
    import argparse
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_gpus", type=int, default=1)
    args = ap.parse_args()

    tasks = collect_tasks(args.base_dir, args.output_dir)
    if not tasks:
        print("No videos found.")
        return
    print(f"▶ {len(tasks)} videos queued.")

    buckets = [[] for _ in range(args.num_gpus)]
    for i, t in enumerate(tasks):
        buckets[i % args.num_gpus].append(t)

    procs = []
    for gid, bucket in enumerate(buckets):
        p = mp.Process(target=worker_process, args=((gid, bucket),))
        p.start(); procs.append(p)
    for p in procs: 
        p.join()

if __name__ == "__main__":
    main()
