import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch.optim as optim
import matplotlib.pyplot as plt
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))
        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)
        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        return torch.sigmoid(self.final(u1))

class RoadDataset(Dataset):
    def __init__(self, img_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') and not '_path' in f]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.img_dir, img_name.replace('.jpg', '_path.jpg'))
        image = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')
        # 提取mask (假设淡红色 R>200, G<120, B<120)
        mask_np = np.array(mask_img)
        mask = ((mask_np[:,:,0] > 200) & (mask_np[:,:,1] < 120) & (mask_np[:,:,2] < 120)).astype(np.float32)
        mask = Image.fromarray((mask*255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

# # 常用transform
if __name__ == '__main__':
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = RoadDataset('D:\= =\\4544\hiking_project\image_path', transform=transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_losses = []
    for epoch in range(30):
        model.train()
        total_loss = 0
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            loss = criterion(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')
    torch.save(model.state_dict(), 'unet_road.pth')
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('UNet Training Loss Curve')
    plt.grid(True)
    plt.savefig('unet_train_loss.png')
    plt.show()


