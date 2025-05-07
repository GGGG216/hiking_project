import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from path_highliter import UNet


VAL_DIR = "D:\= =\\4544\hiking_project\data\path_val"
MODEL_PATH = "unet_road.pth"
OUT_PATH = "val_grid.jpg"
IMG_SIZE = 256  


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def overlay_mask(image, mask, alpha=0.5):
    """PIL image (RGB, 0-255), mask: numpy 0~1, return numpy (H,W,3) 0~255"""
    img_np = np.array(image).astype(np.float32)
    mask = mask.squeeze()

    red = np.zeros_like(img_np)
    red[..., 0] = 255  # R
    red[..., 1] = 128  # G
    red[..., 2] = 128  # B
    mask_3ch = np.stack([mask]*3, axis=2)
    overlay = img_np * (1 - mask_3ch * alpha) + red * (mask_3ch * alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return overlay


img_files = sorted([f for f in os.listdir(VAL_DIR) if f.lower().endswith('.jpg')])[:9]

visuals = []
for fname in img_files:

    img_path = os.path.join(VAL_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred = model(x).cpu().numpy()[0,0]

    vis = overlay_mask(img_resized, mask_pred, alpha=0.5)
    visuals.append(vis)


rows = []
for i in range(3):
    row = np.concatenate(visuals[i*3:(i+1)*3], axis=1)
    rows.append(row)
grid = np.concatenate(rows, axis=0)

cv2.imwrite(OUT_PATH, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
