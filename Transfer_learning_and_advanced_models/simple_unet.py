import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Transforms
# -----------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# -----------------------------
# Dataset
# -----------------------------
class LevirDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform

        self.images_A = sorted(os.listdir(os.path.join(root_dir, "A")))
        self.images_B = sorted(os.listdir(os.path.join(root_dir, "B")))
        self.labels = sorted(os.listdir(os.path.join(root_dir, "label")))

    def __len__(self):
        return len(self.images_A)

    def __getitem__(self, idx):
        img_A = Image.open(os.path.join(self.root, "A", self.images_A[idx])).convert("RGB")
        img_B = Image.open(os.path.join(self.root, "B", self.images_B[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.root, "label", self.labels[idx])).convert("L")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            mask = self.transform(mask)

        # stack (6 channels)
        image = torch.cat([img_A, img_B], dim=0)

        # binary mask
        mask = (mask > 0).float()

        return image, mask

# -----------------------------
# Data Loaders
# -----------------------------
train_dataset = LevirDataset("C:/Users/HARSHIT/Desktop/archive/LEVIR CD/train", transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = LevirDataset("C:/Users/HARSHIT/Desktop/archive/LEVIR CD/test", transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# -----------------------------
# Model Blocks
# -----------------------------
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

        self.down1 = DoubleConv(6, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        x4 = self.bottleneck(self.pool(x3))

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        return self.final(x)

# -----------------------------
# Loss Functions
# -----------------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

def dice_score(pred, target, smooth=1):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# -----------------------------
# Model Setup
# -----------------------------
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(10):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0
    dice_total = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            preds_sig = torch.sigmoid(preds)
            preds_bin = (preds_sig > 0.5).float()

            correct += (preds_bin == masks).sum().item()
            total += masks.numel()

            dice_total += dice_score(preds_sig, masks).item()

    acc = 100 * correct / total
    avg_dice = dice_total / len(test_loader)

    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | PixelAcc: {acc:.2f}% | Dice: {avg_dice:.4f}")

# -----------------------------
# Visualization
# -----------------------------
model.eval()

with torch.no_grad():
    images, masks = next(iter(test_loader))
    images = images.to(device)
    preds = model(images)

    img_t1 = images[0][:3].permute(1,2,0).cpu()
    img_t2 = images[0][3:].permute(1,2,0).cpu()
    pred = torch.sigmoid(preds[0][0]).detach().cpu()
    gt = masks[0][0]

    plt.figure(figsize=(12,3))
    plt.subplot(1,4,1); plt.title("T1"); plt.imshow(img_t1)
    plt.subplot(1,4,2); plt.title("T2"); plt.imshow(img_t2)
    plt.subplot(1,4,3); plt.title("Pred"); plt.imshow(pred, cmap="gray")
    plt.subplot(1,4,4); plt.title("GT"); plt.imshow(gt, cmap="gray")
    plt.show()

    '''
    Using device: cuda
Epoch 1 | Loss: 0.6808 | PixelAcc: 94.29% | Dice: 0.5174
Epoch 2 | Loss: 0.5663 | PixelAcc: 95.08% | Dice: 0.6478
Epoch 3 | Loss: 0.5161 | PixelAcc: 94.50% | Dice: 0.6306
Epoch 4 | Loss: 0.4758 | PixelAcc: 93.17% | Dice: 0.5975
Epoch 5 | Loss: 0.4319 | PixelAcc: 96.84% | Dice: 0.7086
Epoch 6 | Loss: 0.4043 | PixelAcc: 96.74% | Dice: 0.7224
Epoch 7 | Loss: 0.3740 | PixelAcc: 95.65% | Dice: 0.6951
Epoch 8 | Loss: 0.3477 | PixelAcc: 95.32% | Dice: 0.6888
Epoch 9 | Loss: 0.3168 | PixelAcc: 96.95% | Dice: 0.7344
Epoch 10 | Loss: 0.2838 | PixelAcc: 97.38% | Dice: 0.7510
    '''