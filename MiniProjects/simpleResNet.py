import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False
)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # identity shortcut by default
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        shortcut = self.shortcut(x)

        out += shortcut
        out = F.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(64,2,stride=1)
        self.layer2 = self.make_layer(128,2,stride=2)
        self.layer3 = self.make_layer(256,2,stride=2)
        self.layer4 = self.make_layer(512,2,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)

    def make_layer(self,out_channels,blocks,stride):
        strides = [stride] + [1]*(blocks-1)

        layers = []

        for s in strides:
            layers.append(
                ResidualBlock(self.in_channels,out_channels,s)
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x,1)
        x = self.fc(x)

        return x


model = ResNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses=[]
val_acc=[]

for epoch in range(20):

    model.train()

    running_loss=0

    for images,labels in trainloader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    train_losses.append(running_loss/len(trainloader))

    # validation
    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for images,labels in testloader:

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)

            _,predicted=torch.max(outputs,1)

            total+=labels.size(0)

            correct+=(predicted==labels).sum().item()

    acc=100*correct/total

    val_acc.append(acc)

    print(f"Epoch {epoch+1} Loss:{train_losses[-1]:.3f} ValAcc:{acc:.2f}%")


plt.plot(train_losses,label="loss")
plt.plot(val_acc,label="accuracy")
plt.legend()
plt.show()

all_preds=[]
all_labels=[]

model.eval()

with torch.no_grad():

    for images,labels in testloader:

        images=images.to(device)

        outputs=model(images)

        _,preds=torch.max(outputs,1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        

cm=confusion_matrix(all_labels,all_preds)

plt.figure(figsize=(10,8))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()