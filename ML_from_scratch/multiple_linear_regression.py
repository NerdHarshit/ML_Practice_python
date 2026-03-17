import numpy as np

# dataset (2 features)
X = np.array([
    [1,2],
    [2,1],
    [3,4],
    [4,3],
    [5,5]
])

y = np.array([0,0,1,1,1])

n_samples, n_features = X.shape

W = np.zeros(n_features)
b = 0

lr = 0.1
epochs = 1000

def get_mse_loss(y,y_p):
    mse = np.mean((y-y_p)**2)
    return mse

def get_dw(x,y,yp):
    dw = 1/n_samples * (x@(y-yp))
    return dw

def get_db(y,yp):
    db = -2*np.mean(y-yp)
    return db


for i in range(epochs):

    z = X @ W + b
   
    loss = get_mse_loss(y,z)

    dw = get_dw(X.T,y,z)
    db = get_db(y,z)

    W -= lr * dw
    b -= lr * db

    if i % 100 == 0:
        print("loss:", loss)

print("W:", W)
print("b:", b)