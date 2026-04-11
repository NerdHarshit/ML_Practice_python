import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor(3.0,requires_grad=True)

y = x**2 + 2*x
y.backward()
print(x.grad)