import numpy as np

np.random.seed(0)

n_samples = 100
n_features = 6

X = np.random.randn(n_samples, n_features)

true_w = np.array([4.0, -3.0, 0.0, 0.0, 0.0, 0.0])
noise = np.random.randn(n_samples) * 2

y = X @ true_w + 5 + noise

W = np.zeros(n_features)
b = 0

lr = 0.01
epochs = 3000
l = 0.01

for i in range(epochs):

    z = X @ W + b

    loss = np.mean((y - z)**2) + l * abs(W) #mse plus l1

    dw = (1/n_samples) * (X.T @ (z - y)) + l*np.sign(W)
    db = np.mean(z - y)

    W -= lr * dw
    b -= lr * db

    if i % 100 == 0:
        print("loss:", loss)

print("W:", W)
print("b:", b)