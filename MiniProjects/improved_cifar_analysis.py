import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class CIFAR10_model(nn.Module):
    def __init__(self):
        super().__init__()

        #initially image is 32x32
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn4 = nn.BatchNorm2d(64)


        self.pool = nn.MaxPool2d(2,2)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64*8*8,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = torch.flatten(x,1)

        x = self.dropout(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return(x)
    
model = CIFAR10_model().to(device)

model.load_state_dict(torch.load("Improved_cifar_cnn.pth"))

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs,1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CIFAR-10 Confusion Matrix")

plt.show()


def show_mistakes(model, loader):

    model.eval()

    images, labels = next(iter(loader))

    images_gpu = images.to(device)

    with torch.no_grad():

        outputs = model(images_gpu)

        _, preds = torch.max(outputs,1)

    images = images.cpu()

    plt.figure(figsize=(12,6))

    count = 0

    for i in range(len(images)):

        if preds[i] != labels[i]:

            plt.subplot(2,5,count+1)

            img = images[i].permute(1,2,0)

            plt.imshow(img)

            plt.title(f"P:{preds[i]} T:{labels[i]}")

            plt.axis("off")

            count += 1

            if count == 10:
                break

    plt.show()


show_mistakes(model, test_loader)