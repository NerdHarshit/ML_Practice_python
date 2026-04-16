#imports 
import torch 
import torch.nn as nn 
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device",device)

Train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

Test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_dataset = datasets.CIFAR10(
    root="./data",
    download=True,
    train=True,
    transform=Train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=Test_transform
)

train_data = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=64
)

test_data = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=64
)

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

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 10

train_losses = []
train_accuracies = []
eval_accuracies = []

for epoch in range(epochs):

    model.train()

    total = 0
    correct = 0
    running_loss = 0

    for images,labels in train_data:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        lossval = loss(outputs,labels)

        optimizer.zero_grad()
        lossval.backward()
        optimizer.step()

        running_loss += lossval.item()
        
        _,predicted = torch.max(outputs,1)

        total +=labels.size(0)
        correct+=(predicted == labels).sum().item()

    epoch_loss = running_loss/len(train_data)
    train_accuracy = 100* correct /total

    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images , labels in test_data:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs,1)

            total +=labels.size(0)
            correct += (predicted == labels).sum().item()
        
    eval_accuracy = 100*correct/total
    eval_accuracies.append(eval_accuracy)
    print(f"Epoch{epoch+1}/{epochs} | Loss {epoch_loss:.4f} | train acc: {train_accuracy: .2f}% | val accu :{eval_accuracy : .2f}%")


torch.save(model.state_dict(),"Improved_cifar_cnn.pth")