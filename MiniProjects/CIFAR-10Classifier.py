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

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            padding=1,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(2,2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128*4*4,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x,1)

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


torch.save(model.state_dict(),"cifar_cnn.pth")

'''
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(eval_accuracies, label="Validation Accuracy")

plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

#confusion matrix
all_preds = []
all_labels = []

model.eval()

with torch.no_grad():

    for images, labels in test_data:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

#visualize predictions 
def visualize_predictions(model, loader):

    model.eval()

    images, labels = next(iter(loader))

    images_gpu = images.to(device)

    with torch.no_grad():

        outputs = model(images_gpu)

        _, preds = torch.max(outputs,1)

    plt.figure(figsize=(10,5))

    for i in range(10):

        plt.subplot(2,5,i+1)

        img = images[i].squeeze()

        plt.imshow(img)

        plt.title(f"P:{preds[i].item()} T:{labels[i].item()}")

        plt.axis("off")

    plt.show()

visualize_predictions(model, test_data)

#run inference on real image of yours
def predict_image(image_path):

    transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    img = Image.open(image_path)

    img = transform(img)

    img = img.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        output = model(img)

        _, prediction = torch.max(output,1)

    print("Predicted Digit:", prediction.item())

    plt.imshow(img.cpu().squeeze())
    plt.title(f"Prediction: {prediction.item()}")
    plt.axis("off")
    plt.show()

#predict_image("image path here in 28x28 black and white png")

'''



