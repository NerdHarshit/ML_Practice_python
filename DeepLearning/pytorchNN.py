import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------
# 1. Create dummy dataset
# ------------------------------------------------

X = torch.randn(100,10)     # 100 samples, 10 features
y = torch.randint(0,2,(100,1)).float()   # binary labels

# ------------------------------------------------
# 2. Define the neural network
# ------------------------------------------------

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(10,50)
        self.fc2 = nn.Linear(50,1)

    def forward(self,x):

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


model = Net()

# ------------------------------------------------
# 3. Define loss function
# ------------------------------------------------

criterion = nn.BCELoss()

# ------------------------------------------------
# 4. Define optimizer
# ------------------------------------------------

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------
# 5. Training loop
# ------------------------------------------------

epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()        # reset gradients

    outputs = model(X)           # forward pass

    loss = criterion(outputs,y)  # compute loss

    loss.backward()              # backpropagation

    optimizer.step()             # update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")