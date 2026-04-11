import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1,2,3]).to(device)

y = torch.ones(3,3).to(device)

z = x+y
print(z)