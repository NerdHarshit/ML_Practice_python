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

#device select - gpu if available else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device ",device)

#this transform (comprising of image->tensor and normalization (centring around 0))will apply on all images (test and train)
transform = transforms.Compose([
    transforms.ToTensor(),#pixel alpha [0,255]->tensor [0,1]
    transforms.Normalize((0.5,),(0.5,))#values in tensor get centered around zero so mean = 0 std = 1
])

#downloading the dataset 
train_dataset = datasets.MNIST(
    root="./data", #download directory
    train=True, #this data to be used as train split
    download=True,
    transform=transform #the transform we made should be applied on the images
)

test_dataset = datasets.MNIST(
    root="./data", #download directory
    train=False, #this data to be used as test split
    download=True,
    transform=transform #the transform we made should be applied on the images
)

#dataloader handels shuffling of the dataset and also batching so we dont have to do it manually .. shuffling helps in preventing model from memorization

#load training data as chunks of 64 images , shuffled from the train dataset images in directory
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

#load testing data as chunks of 64 images , un-shuffled from the test dataset images in directory
#no shuffle here as then ???
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

#making the cnn by creating a class that inherits from nn.module as a standard practice
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, #channel(1= Black and white)
            out_channels=16, #16 kernels to be found 
            kernel_size=3, #3x3 matrix
            padding=1) 
        
        '''
         since 28x28 images in mnist , we want in and out size same 28
         so by formula 
         output_size = (input_size + 2*padding - kernel size) / stride + 1
         stride = 1 default
         input = 28
         output needed = 28
         kernel = 3 
         so padding must be 1
        '''
        #similarly for this too
        self.conv2 = nn.Conv2d(
            in_channels =16,
            out_channels=32,
            kernel_size=3,
            padding=1)

        self.pool = nn.MaxPool2d(2,2) #

        self.fc1 = nn.Linear(32*7*7,128)#since after 2 conv and pooling layers the out image will be 7x7 and when flattened it will be 7*7*32 as total out channels =32
        self.fc2 = nn.Linear(128,10) #128 features from previous layer go to 10 classes (0-9 digits)

        self.relu = nn.ReLU() #activation for conv and also for fc1

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = torch.flatten(x,1) #7*7*32 length vector

        x = self.relu(self.fc1(x)) #activation done on vector x using relu 
        x = self.fc2(x) #activated values passed to fc2 layer but not activated  

        return x
    

model = MNISTCNN().to(device)#send model and parameters to gpu

loss = nn.CrossEntropyLoss() #cross entropy loss for multi class classification

optimizer = optim.Adam(model.parameters(),lr=0.001) #trainable params passed to adam optimizer for gradients , low lr learns better

epochs = 10

train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images , labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        lossval = loss(outputs,labels)

        optimizer.zero_grad()
        lossval.backward()
        optimizer.step()

        running_loss += lossval.item()
        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct+=(predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100*correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)

    #validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images , labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs,1)

            total +=labels.size(0)
            correct += (predicted == labels).sum().item()
        
    
    val_accuracy = 100*correct/total
    val_accuracies.append(val_accuracy)

    print(f"Epoch{epoch+1}/{epochs} | Loss {epoch_loss:.4f} | train acc: {train_accuracy: .2f}% | val accu :{val_accuracy : .2f}%")


plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

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

    for images, labels in test_loader:

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

        plt.imshow(img, cmap="gray")

        plt.title(f"P:{preds[i].item()} T:{labels[i].item()}")

        plt.axis("off")

    plt.show()

visualize_predictions(model, test_loader)

#run inference on real image of yours
def predict_image(image_path):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path)

    img = transform(img)

    img = img.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        output = model(img)

        _, prediction = torch.max(output,1)

    print("Predicted Digit:", prediction.item())

    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.title(f"Prediction: {prediction.item()}")
    plt.axis("off")
    plt.show()

#predict_image("image path here in 28x28 black and white png")