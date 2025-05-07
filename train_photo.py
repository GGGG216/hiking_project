import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 引入UNet
from path_highliter import UNet

class RoadDifficultyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') and not '_path' in f]
        self.transform = transform
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.img_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert('RGB')
        
        # 读取label
        with open(label_path, 'r') as f:
            label = int(f.read().strip()) - 1

        if self.transform:
            image = self.transform(image)
      
        return image, label

def mask_images(unet, images):
    with torch.no_grad():
        masks = unet(images)
        masks = masks.repeat(1, 3, 1, 1)  # (B,1,H,W) -> (B,3,H,W)
        masked_imgs = images * masks
        return masked_imgs

def train_one_epoch(resnet, unet, train_loader, criterion, optimizer, device):
    resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        mask = mask_images(unet,images)
        outputs = resnet(mask  )
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(resnet, unet, val_loader, criterion, device):
    resnet.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            mask = mask_images(unet, images)
            outputs = resnet(mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet().to(device)
    unet.load_state_dict(torch.load('unet_road.pth', map_location=device))
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False
    
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = RoadDifficultyDataset('D:\= =\\4544\hiking_project\data\\train', transform=transform)
    val_dataset = RoadDifficultyDataset('D:\= =\\4544\hiking_project\data\\val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    resnet = models.resnet18()
    resnet.fc = nn.Linear(resnet.fc.in_features, 5)
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)
    
    num_epochs = 30
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train_one_epoch(resnet, unet, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(resnet, unet, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    torch.save(resnet.state_dict(), 'road_difficulty_resnet.pth')
    print("ResNet模型已保存为 road_difficulty_resnet.pth")

    # Plot loss curve
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig('loss_curve_masked.png')
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve')
    plt.legend()
    plt.savefig('acc_curve_masked.png')
    plt.close()

    # Plot confusion matrix
    class_names = ['Difficulty 1', 'Difficulty 2', 'Difficulty 3', 'Difficulty 4', 'Difficulty 5']
    plot_confusion_matrix(val_labels, val_preds, class_names)
    print("Confusion matrix saved as confusion_matrix.png")