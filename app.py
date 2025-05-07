import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory
from path_highliter import UNet
import cv2
import mediapipe as mp
import math

IMG_SIZE = 256
UNET_PATH = 'unet_road.pth'
RESNET_PATH = 'road_difficulty_resnet.pth'
LABELS = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
UPLOAD_FOLDER = 'Uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


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
def overlay_mask(image, mask, alpha=0.5):
    img_np = np.array(image).astype(np.float32)
    mask = mask.squeeze()
    red = np.zeros_like(img_np)
    red[..., 0] = 255
    red[..., 1] = 128
    red[..., 2] = 128
    mask_3ch = np.stack([mask]*3, axis=2)
    overlay = img_np * (1 - mask_3ch * alpha) + red * (mask_3ch * alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return overlay
def calculate_tilt_angle(image):
  
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        delta_y = shoulder_mid[1] - hip_mid[1]
        delta_x = shoulder_mid[0] - hip_mid[0]
        angle = math.degrees(math.atan2(abs(delta_x), abs(delta_y)))
        return angle
    return 0 
    
HTML = """
<!doctype html>
<html lang="en">
<head>
    <title>Path Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
        body { font-family: Arial; margin: 2em; background: #f8f8f8; }
        .container { max-width: 600px; margin: auto; background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 0 10px #bbb; }
        h1 { text-align: center; }
        img { max-width: 100%; margin-top: 10px; border-radius: 4px; }
        .result { margin-top: 1.5em; }
        .pred { font-size: 1.3em; color: #d2691e; }
        .file-input-label { 
            display: inline-block; 
            padding: 10px 20px; 
            background: #4CAF50; 
            color: white; 
            border-radius: 4px; 
            cursor: pointer; 
            margin-right: 10px; 
        }
        .file-input { display: none; }
        button { 
            padding: 10px 20px; 
            background: #008CBA; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
        }
        button:hover { background: #005f73; }
        .file-input-label:hover { background: #45a049; }
        .form-group { margin-bottom: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Path Analysis</h1>
    <form method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="file" class="file-input-label">Choose Image</label>
            <input type="file" id="file" name="file" class="file-input" accept="image/*" required>
            <button type="submit">Analyze</button>
        </div>
    </form>
    {% if filename %}
    <div class="result">
        <h3>Original Image:</h3>
        <img src="{{ url_for('uploaded_file', filename=filename) }}">
        <h3>Masked Image</h3>
        <img src="{{ url_for('result_file', filename=masked_name) }}">
        <h3>Highlighted Path</h3>
        <img src="{{ url_for('result_file', filename=overlay_name) }}">
        <div class="pred">Difficulty Prediction: <b>{{ pred_label }}</b></div>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template_string(HTML)
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image = Image.open(file_path).convert('RGB')
        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            mask_pred = unet(x).cpu().numpy()[0,0]
            mask_bin = (mask_pred > 0.5).astype(np.float32)
            img_np = np.array(img_resized).astype(np.float32)/255.0
            masked_img_np = img_np * mask_bin[..., None]
            masked_img_uint8 = (masked_img_np * 255).clip(0,255).astype(np.uint8)
      
            masked_img_pil = Image.fromarray(masked_img_uint8)
            masked_name = f"masked_{filename}.jpg"
            masked_path = os.path.join(app.config['RESULT_FOLDER'], masked_name)
            masked_img_pil.save(masked_path)
        
            vis = overlay_mask(img_resized, mask_bin, alpha=0.5)
            overlay_name = f"overlay_{filename}.jpg"
            overlay_path = os.path.join(app.config['RESULT_FOLDER'], overlay_name)
            Image.fromarray(vis).save(overlay_path)
 
            masked_tensor = transforms.ToTensor()(masked_img_pil).unsqueeze(0).to(device)
            output = resnet(masked_tensor)
            pred = output.argmax(dim=1).item()
            tilt_angle = calculate_tilt_angle(image)
            print(f"Detected human body angle{tilt_angle:.2f} degree")
            if tilt_angle > 30:
                pred = 4 

            pred_label = LABELS[pred]
   
        return render_template_string(HTML,
            filename=filename, masked_name=masked_name, overlay_name=overlay_name,
            pred_label=pred_label)

    return render_template_string(HTML, filename=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7890)
