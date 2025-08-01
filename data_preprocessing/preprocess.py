import os
import glob
import cv2
import numpy as np
import subprocess
import logging
import traceback
import torchaudio
import librosa
import av
import ffmpeg
from scipy.signal import resample_poly
from scipy.io.wavfile import write as wav_write
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
import mediapipe as pipe
import torch
import multiprocessing as mp


def save_wav_scipy(audio_array, sr, output_path):
    audio_int16 = (audio_array * 32767).astype(np.int16)
    wav_write(output_path, sr, audio_int16)
    logging.info(f"Audio saved to {output_path}")


def read_audio_ffmpeg(video_path: str, target_sr: int = 16000) -> np.ndarray | None:
    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vn", "-ac", "1", "-f", "s16le", "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.warning(f"[FFmpeg] extraction failed: {e.stderr.decode().strip()}")
        return None
    audio_int16 = np.frombuffer(proc.stdout, dtype=np.int16)
    if audio_int16.size == 0:
        logging.warning("[FFmpeg] no audio frames extracted.")
        return None
    audio = audio_int16.astype(np.float32) / 32768.0
    try:
        probe = ffmpeg.probe(video_path, select_streams="a")
        orig_sr = int(probe["streams"][0]["sample_rate"])
    except Exception:
        orig_sr = target_sr
    if orig_sr != target_sr:
        g = np.gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    return audio


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def initialize_models(max_faces: int = 5):
    mp_face_mesh = pipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.4
    )
    return face_mesh

    
def safe_iter_frames(video_path: str):
    """
    Yield RGB (uint8) frames using PyAV.
    Any corrupt frame is skipped; if the whole file cannot be opened
    the generator is empty (so your loop simply runs 0 times).

    >>> for f in safe_iter_frames("video.mp4"):
    ...     # f is a (H,W,3) uint8 RGB array
    ...     pass
    """
    try:
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                try:
                    yield frame.to_ndarray(format="rgb24")
                except (av.AVError, ValueError) as e:
                    logging.warning(f"[safe_iter_frames] skipped a frame: {e}")
    except av.AVError as e:
        logging.error(f"[safe_iter_frames] cannot open {video_path}: {e}")
        return


def has_audio_stream(video_path):
    try:
        probe_result = ffmpeg.probe(video_path)
        return any(s.get('codec_type') == 'audio' for s in probe_result['streams'])
    except Exception as e:
        logging.warning(f"ffmpeg.probe failed for {video_path}: {e}")
        return False


def process_single_video(args):
    video_path, savepath, device = args
    os.makedirs(savepath, exist_ok=True)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    log_file = os.path.join(savepath, f'{filename}_processing.log')
    setup_logging(log_file)
    logging.info(f'Processing: {video_path} on {device}')

    # Load video
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        logging.error(f"Could not load video: {e}")
        return

    # Prepare face detector
    face_model = initialize_models(max_faces=5)

    # Dictionaries to collect frames and landmarks per face index
    face_frames = {}
    face_landmarks = {}

    with face_model:
        # for frame_id, frame_rgb in enumerate(clip.iter_frames(dtype='uint8')):
        for frame_id, frame_rgb in enumerate(safe_iter_frames(video_path)):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results = face_model.process(frame_bgr)
            if not results.multi_face_landmarks:
                continue

            h, w, _ = frame_bgr.shape
            for idx, lm_set in enumerate(results.multi_face_landmarks):
                # Extract landmark coordinates
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in lm_set.landmark]

                # Append to landmarks list
                face_landmarks.setdefault(idx, []).append(coords)

                # Crop face region
                xs, ys = zip(*coords)
                x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, w)
                y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, h)
                crop = frame_bgr[y1:y2, x1:x2]
                face_frames.setdefault(idx, []).append(crop)

    # Write one cropped video per face and save landmarks
    for idx, frames in face_frames.items():
        if not frames:
            continue

        # --- save landmarks for this face ---
        lm_arr = np.array(face_landmarks[idx], dtype=object)  # shape: (num_frames, 468, 2)
        lm_path = os.path.join(savepath, f'landmarks.npy')
        np.save(lm_path, lm_arr)
        logging.info(f'Saved landmarks to {lm_path}')

        # --- save cropped face video ---
        seq = []
        for f in frames:
            f_resized = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
            seq.append(cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB))

        out_clip = ImageSequenceClip(seq, fps=clip.fps)
        out_path = os.path.join(savepath, f'cropped_face.mp4')
        if clip.audio is not None:
            out_clip.audio = clip.audio

        try:
            out_clip.write_videofile(out_path, codec='libx264', audio_codec='aac')
            logging.info(f'Saved video {out_path}')
        except Exception as e:
            logging.warning(f'Failed with audio, retrying without: {e}')
            out_clip.audio = None
            out_clip.write_videofile(out_path, codec='libx264')

    # Extract and save audio once
    if has_audio_stream(video_path):
        audio_path = os.path.join(savepath, f'{filename}_audio.wav')
        audio = read_audio_ffmpeg(video_path, target_sr=16000)
        if audio is not None:
            save_wav_scipy(audio, sr=16000, output_path=audio_path)
        else:
            try:
                ffmpeg.input(video_path).output(audio_path, ar=16000, ac=1,
                                                acodec='pcm_s16le')\
                      .overwrite_output().run(quiet=True)
                logging.info(f'Fallback extracted audio to {audio_path}')
            except Exception as e:
                logging.warning(f'Audio extraction fallback failed: {e}')

    clip.reader.close()
    clip.close()


def worker_process(args):
    gpu_id, tasks = args
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    for video_path, savepath in tqdm(tasks, desc=f'GPU {gpu_id}'):
        process_single_video((video_path, savepath, device))


def process_directory(directory, base_dir, output_dir):
    rel = os.path.relpath(directory, base_dir)
    out_dir = os.path.join(output_dir, rel)
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for f in glob.glob(os.path.join(directory, '*.mp4')):
        name = os.path.splitext(os.path.basename(f))[0]
        savepath = os.path.join(out_dir, name)
        if not os.path.exists(savepath):
            tasks.append((f, savepath))
    return tasks


def main():
    import argparse
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()

    # Gather all video tasks
    all_tasks = []
    for root, _, _ in os.walk(args.base_dir):
        all_tasks.extend(process_directory(root, args.base_dir, args.output_dir))

    if not all_tasks:
        logging.info('No videos found.')
        return
    print(f"Total {len(all_tasks)} videos found")

    # Distribute tasks across GPUs
    distributed = [[] for _ in range(args.num_gpus)]
    for i, task in enumerate(all_tasks):
        distributed[i % args.num_gpus].append(task)

    # Launch workers
    procs = []
    for gpu_id, tasks in enumerate(distributed):
        p = mp.Process(target=worker_process, args=((gpu_id, tasks),))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
