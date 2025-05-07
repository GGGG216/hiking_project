import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import sys
import os
from path_highliter import UNet
import cv2
import mediapipe as mp
import math

# ----- 配置 -----
IMG_SIZE = 224
UNET_PATH = 'unet_road.pth'
RESNET_PATH = 'road_difficulty_resnet.pth'
LABELS = ['难度1', '难度2', '难度3', '难度4', '难度5']
IMG_PATH = "D:\\= =\\4544\\hiking_project\\level4_example_2.jpg"
MASKED_IMG_PATH = 'masked_output.jpg'

# ----- 预处理 -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ----- 载入模型 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet().to(device)
unet.load_state_dict(torch.load(UNET_PATH, map_location=device))
unet.eval()

resnet = models.resnet18(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 5)
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
resnet = resnet.to(device)
resnet.eval()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ----- mask可视化函数 -----
def overlay_mask(image, mask, alpha=0.5):
    """PIL image (RGB, 0-255), mask: numpy 0~1, return numpy (H,W,3) 0~255"""
    img_np = np.array(image).astype(np.float32)
    mask = mask.squeeze()
    # 生成红色叠加
    red = np.zeros_like(img_np)
    red[..., 0] = 255  # R
    red[..., 1] = 128  # G
    red[..., 2] = 128  # B
    mask_3ch = np.stack([mask]*3, axis=2)
    overlay = img_np * (1 - mask_3ch * alpha) + red * (mask_3ch * alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return overlay

# ----- 人体倾斜角度检测函数 -----
def calculate_tilt_angle(image):
    """Calculate the tilt angle of a human in the image using MediaPipe."""
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # 获取左右肩膀和臀部的关键点
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # 计算肩膀和臀部的中点
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        
        # 计算身体中线相对于垂直线的角度
        delta_y = shoulder_mid[1] - hip_mid[1]
        delta_x = shoulder_mid[0] - hip_mid[0]
        
        # 计算角度（弧度转换为度）
        angle = math.degrees(math.atan2(abs(delta_x), abs(delta_y)))
        return angle
    return 0  # 如果没有检测到人体，返回0度

# ----- 主处理流程 -----
image = Image.open(IMG_PATH).convert('RGB')
img_resized = image.resize((IMG_SIZE, IMG_SIZE))
x = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    # 1. 生成mask
    mask_pred = unet(x).cpu().numpy()[0,0]
    mask_bin = (mask_pred > 0.5).astype(np.float32)  # 二值化

    # 2. masked图片
    img_np = np.array(img_resized).astype(np.float32)/255.0
    masked_img_np = img_np * mask_bin[..., None]
    masked_img_uint8 = (masked_img_np * 255).clip(0,255).astype(np.uint8)
    masked_img_pil = Image.fromarray(masked_img_uint8)
    masked_img_pil.save(MASKED_IMG_PATH)
    print(f"道路masked图片已保存为 {MASKED_IMG_PATH}")

    # 3. mask可视化（淡红色叠加）
    vis = overlay_mask(img_resized, mask_bin, alpha=0.5)
    vis_path = 'overlay_mask.jpg'
    Image.fromarray(vis).save(vis_path)
    print(f"mask可视化已保存为 {vis_path}")

    # 4. 检测人体倾斜角度
    tilt_angle = calculate_tilt_angle(image)
    print(f"检测到的人体倾斜角度：{tilt_angle:.2f}度")
    
    # 5. 难度预测
    masked_tensor = transforms.ToTensor()(masked_img_pil).unsqueeze(0).to(device)
    output = resnet(masked_tensor)
    pred = output.argmax(dim=1).item()
    
    # 如果倾斜角度大于40度，直接判定为难度5
    if tilt_angle > 40:
        pred = 4  # 难度5（索引4）
        print("人体倾斜角度超过40度，强制判定为难度5")
    
    print(f"该照片预测难度为：{LABELS[pred]}")

# 释放MediaPipe资源
pose.close()

# 可选：自动弹窗显示
# masked_img_pil.show()
# Image.fromarray(vis).show()