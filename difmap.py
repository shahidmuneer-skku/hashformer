import cv2
import numpy as np
import torch
import lpips
from torchvision import transforms
from PIL import Image

# === 1. Load real and fake images ===
def load_image(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# === 2. Pixel-wise difference map ===
# def pixel_diff_map(real_img, fake_img):
#     diff = cv2.absdiff(real_img, fake_img)
#     diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
#     norm_diff = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
#     return norm_diff.astype(np.uint8)




def pixel_diff_map(real_img, fake_img, threshold=10):
    # Compute absolute color difference
    diff = cv2.absdiff(real_img, fake_img)

    # Create mask where difference is significant
    mask = (diff > threshold).any(axis=2)  # Shape: (H, W)

    # Initialize white background
    result = np.ones_like(real_img) * 255  # White background

    # Paste only the diff pixels onto the white background
    result[mask] = diff[mask]

    return result
# === 3. LPIPS perceptual difference map ===
def lpips_diff_map(real_img, fake_img, device='cuda' if torch.cuda.is_available() else 'cpu'):
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    real_tensor = transform(real_img).unsqueeze(0).to(device)
    fake_tensor = transform(fake_img).unsqueeze(0).to(device)

    with torch.no_grad():
        dist_map = lpips_fn.forward(real_tensor, fake_tensor)

    diff = dist_map.squeeze().cpu().numpy()
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return diff.astype(np.uint8)

# === 4. Run it ===
if __name__ == "__main__":
    real_path = "/media/NAS/DATASET/faceforensics++/QAD/v1face_data/data_raw/original_sequences/youtube/raw/v1faces/004/0000.png"
    fake_path = "/media/NAS/DATASET/faceforensics++/QAD/v1face_data/data_raw/manipulated_sequences/FaceSwap/raw/v1faces/004_982/0000.png"

    real = load_image(real_path)
    fake = load_image(fake_path)

    # Pixel-wise map
    pixel_map = pixel_diff_map(real, fake)
    cv2.imwrite("pixel_diff_map.png", pixel_map)

    # LPIPS map
    perceptual_map = lpips_diff_map(real, fake)
    cv2.imwrite("lpips_diff_map.png", perceptual_map)

    print("Difference maps saved.")
