# Hiking Expert: Path Detection & Difficulty Evaluation

An AI system for automatic hiking path segmentation and difficulty classification from images.

## Features

- **U-Net**: Segments hiking paths in RGB images.
- **ResNet18**: Classifies hiking difficulty (levels 1–5) from masked images.
- **MediaPipe**: Pose estimation improves accuracy on challenging samples.
- **Web Demo**: Upload an image to get path mask and difficulty prediction.

## Difficulty Levels

1. Flat government roads  
2. Government roads with complexity (stairs etc.)  
3. Flat rural paths  
4. Dangerous paths with obstacles  
5. Extremely dangerous, requires both hands

## Project File Structure

```plaintext
.
├── .git/
├── data/
├── app.py
├── main.py 
├── Midas.py
├── UNET training/
│   ├── path_highlight_test.py
│   └──path_highliter.py
│   └──path_lable.py
├── RESNET18 training/
│   ├──train_photo.py
├── README.md
├── svm_&_kmeans.py
├── training results/
│   ├── train_photo.jpg
│   └── unet_train_loss.png
│   └──val_grid.jpg
│   └──masked_output.jpg
│   └──acc_curve_masked.png
│   └──confusion_matrix.png
│   └──confusion_matrix_picture.png
│   └──level4_example_2.jpg
│   └──loss_curve_masked.png
│   └──overlay_mask.jpg
│   └──test.jpg

data/
├── depth_maps/      # Depth maps from MiDaS
├── image_path/      # Training images and masks
├── path_val/        # Validation path masks
├── train/           # Original images & Labels
└── val/             # Validation images & Labels
