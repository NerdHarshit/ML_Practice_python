import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)

pool = nn.MaxPool2d(kernel_size=2,stride=2)

x = torch.randn(8,3,244,244)

y = conv(x)
print(y.shape)

y = pool(y)

print(y.shape)