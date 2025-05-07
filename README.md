# Hiking Danger Detection and Safety Suggestion Project

This project uses an advanced deep-learning model combined with human pose analysis to assess the danger level of hiking paths. The system:
- Uses a mini‑ResNet-based CNN (in `AdvancedDangerDetector`) to classify images into five danger levels.
- Adjusts the predicted danger level based on human detection and pose analysis (using a pretrained Keypoint R‑CNN available through torchvision). For example, a standing (upright) person may lower the danger level, while a person struggling (bent posture) may indicate a more dangerous path.
- Generates hiking safety suggestions using a GPT‑2 text-generation pipeline (in `SuggestionGenerator`).

## Project File Structure

```plaintext
project/
├── data/
│   ├── train/                # Training images and labels
│   │     ├── image_001.jpg
│   │     ├── image_001.txt  # Example: "3"
│   │     ├── image_002.jpg
│   │     ├── image_002.txt  # Example: "5"
│   │     └── ...
│   └── val/                  # Validation images and labels
│         ├── image_101.jpg
│         ├── image_101.txt  # Example: "2"
│         └── ...
├── models/
│   ├── danger_detector.py       # The AdvancedDangerDetector model and human pose analysis functions
│   └── suggestion_generator.py  # Hiking safety suggestion generator using GPT-2
├── train.py                  # Training script using the custom HikingDataset, with visualization of loss/accuracy and confusion matrix
├── infer.py                  # Inference script that applies human pose analysis and outputs a safety suggestion
├── requirements.txt          # Project dependencies
└── README.md                 # This documentation file