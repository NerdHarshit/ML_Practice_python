import torch
import torch.nn as nn
from torchvision import models
import torchvision
from torchvision import transforms
import torch.optim as optim
import cv2
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------------------------
# Load pretrained MobileNet
# -------------------------

model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Replace classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    10
)

model = model.to(device)


# -------------------------
# Freeze backbone
# -------------------------

for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last MobileNet block
for param in model.features[-1].parameters():
    param.requires_grad = True

# Train classifier
for param in model.classifier.parameters():
    param.requires_grad = True


# -------------------------
# Transforms (ImageNet normalization)
# -------------------------

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -------------------------
# Dataset
# -------------------------

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)


# -------------------------
# Training setup
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

train_losses=[]
val_acc=[]


# -------------------------
# Training Loop
# -------------------------

for epoch in range(10):

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

    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for images,labels in testloader:

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)

            _,pred=torch.max(outputs,1)

            total+=labels.size(0)
            correct+=(pred==labels).sum().item()

    acc=100*correct/total
    val_acc.append(acc)

    print(f"Epoch {epoch+1} Loss:{train_losses[-1]:.3f} ValAcc:{acc:.2f}%")



# =========================
# Grad-CAM Visualization
# =========================

target_layer = model.features[-1][0]

features=None
gradients=None


def forward_hook(module,input,output):
    global features
    features = output


def backward_hook(module,grad_in,grad_out):
    global gradients
    gradients = grad_out[0]


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


image,label = testset[0]

input_tensor = image.unsqueeze(0).to(device)

model.eval()

output = model(input_tensor)

pred = output.argmax()

model.zero_grad()

output[0,pred].backward()


pooled_grad = torch.mean(gradients,dim=(0,2,3))


for i in range(features.shape[1]):
    features[:,i,:,:] *= pooled_grad[i]


heatmap = torch.mean(features,dim=1).squeeze().detach().cpu().numpy()

heatmap = np.maximum(heatmap,0)

if heatmap.max()!=0:
    heatmap /= heatmap.max()


img = image.cpu().permute(1,2,0).numpy()

heatmap = cv2.resize(heatmap,(224,224))
heatmap = np.uint8(255*heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed = heatmap*0.4 + img*255

plt.imshow(superimposed.astype(np.uint8))
plt.axis("off")
plt.show()