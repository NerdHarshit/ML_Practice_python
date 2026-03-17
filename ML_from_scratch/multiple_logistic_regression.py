import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def loss_fn(y, p):
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

# dataset (2 features)
X = np.array([
    [1,2],
    [2,1],
    [3,4],
    [4,3],
    [5,5]
])

y = np.array([0,0,1,1,1])

n_samples, n_features = X.shape #we seperate as matrix could be non square as well

W = np.zeros(n_features)#as w vector should match number of rows ..this is why we split in above step
b = 0

lr = 0.1
epochs = 1000

for i in range(epochs):

    z = X @ W + b
    p = sigmoid(z)

    loss = loss_fn(y, p)

    dw = (1/n_samples) * (X.T @ (p - y))# as mean should be taken over number of cols ie number of values of each variable/parameter/feature whatever u call it
    db = np.mean(p - y)

    W -= lr * dw
    b -= lr * db

    if i % 100 == 0:
        print("loss:", loss)

print("W:", W)
print("b:", b)