import torch 
import torch.nn as nn

x = torch.randn(1,3,64,64)

conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)

pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

activation = nn.ReLU()

y1 = conv1(x)
y_activated1 = activation(y1)
y_pooled1 = pool1(y_activated1)

conv2 = nn.Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    padding=1
)

pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

y2 = conv2(y_pooled1)
y_activated2 = activation(y2)
y_pooled2 = pool2(y_activated2)

print(x.shape) 
print(y_pooled1.shape)
print(y_pooled2.shape)


