import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.cuda.device_count()
from Midas import get_depth_map




# Dataset Definition
# class DifficultyDataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_dir = image_dir
#         self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         label_path = img_path.replace('.npy', '.txt')

#         # Load Image
#         # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         # image = cv2.imread(img_path)
      
#         # edges = cv2.Canny(image, 100, 200)
#         depth_map = get_depth_map(img_path)
#         # Load Label
#         with open(label_path, 'r') as f:
#             label = int(f.read().strip()) - 1  # Convert labels from 1-5 to 0-4

#         # Transform
#         # if self.transform:
#         #     image = self.transform(image)
#         #     edges = self.transform(edges)

#         return depth_map, label
# Dataset Definition
class DifficultyDataset(Dataset):
    def __init__(self, npy_dir):
        self.npy_dir = npy_dir
        self.npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_name = self.npy_files[idx]
        npy_path = os.path.join(self.npy_dir, npy_name)
        label_path = npy_path.replace('.npy', '.txt')
        depth_map = np.load(npy_path)
        depth_map = np.expand_dims(depth_map, axis=0).astype(np.float32)
        with open(label_path, 'r') as f:
            label = int(f.read().strip()) - 1  
        return torch.tensor(depth_map), torch.tensor(label)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3)) 
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y


# SimpleCNN with SE-Block
class SimpleCNNWithSE(nn.Module):
    def __init__(self):
        super(SimpleCNNWithSE, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.se2 = SEBlock(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.se3 = SEBlock(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.se4 = SEBlock(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)

        # Placeholder for dynamically calculated input size
        self.fc1_input_features = None

        # Placeholder fully connected layers
        self.fc1 = nn.Linear(1, 1024)  # Placeholder
      
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.se1(self.conv1(x))))
        x = self.pool(F.relu(self.se2(self.conv2(x))))
        x = self.pool(F.relu(self.se3(self.conv3(x))))
        x = self.pool(F.relu(self.se4(self.conv4(x))))

        # Dynamically calculate fc1 input size
        if self.fc1_input_features is None:
            self.fc1_input_features = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.fc1_input_features, 1024).to(x.device)  # Ensure fc1 is on the same device

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Data Preparation
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((384, 384)),
#     transforms.ToTensor()
# ])


# Training and Validation Functions
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), correct / total




if __name__ == "__main__":
    # CNN 数据集和加载器
    # train_dataset_cnn = DifficultyDataset('D:\= =\\4544\hiking_project\data\\train', transform=transform)
    # val_dataset_cnn = DifficultyDataset('D:\= =\\4544\hiking_project\data\\val', transform=transform)
    train_dataset_cnn = DifficultyDataset('D:\= =\\4544\hiking_project\data\\depth_maps\\train')
    val_dataset_cnn = DifficultyDataset('D:\= =\\4544\hiking_project\data\\depth_maps\\val')
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=8, shuffle=True,drop_last = True)
    val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=1, shuffle=True,drop_last = True)

    # Set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize the model and move it to the device
    model = SimpleCNNWithSE().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005,weight_decay=1e-5)

    # Training loop
    num_epochs = 50
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader_cnn, optimizer, criterion, device)
        val_loss, val_accuracy = validate_model(model, val_loader_cnn, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the Trained Model
    # Save the model and fc1_input_features
    torch.save({'model_state_dict': model.state_dict(),
    'fc1_input_features': model.fc1_input_features
    }, "simple_cnn_with_se.pth")
    print("Model saved as 'simple_cnn_with_se.pth'")

    # Visualization
    # Loss Curve
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    # Confusion Matrix for CNN
    cnn_preds = []
    cnn_true = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader_cnn:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            cnn_preds.extend(predicted.cpu().numpy())
            cnn_true.extend(labels.cpu().numpy())

    # Add back 1 to the labels for visualization and reports
    cnn_preds = [p + 1 for p in cnn_preds]
    cnn_true = [t + 1 for t in cnn_true]

    # Generate confusion matrix
    cm = confusion_matrix(cnn_true, cnn_preds, labels=[1, 2, 3, 4, 5])  # Ensure labels match the matrix dimensions

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for SimpleCNN with SE-Block")
    plt.show()




    # Display classification reports
    print("Classification Report for CNN:")
    print(classification_report(cnn_true, cnn_preds))
