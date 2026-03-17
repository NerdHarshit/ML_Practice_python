'''
🧪 Your Task Now

Before Day 3:

Implement logistic regression from scratch and test:

Example dataset:

X = [1,2,3,4,5,6]
y = [0,0,0,1,1,1]

Check:

loss decreases

predictions become correct

weights stabilize
'''
import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def bce_loss(y, p):
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def get_dw(x,y,p):
    return np.mean((p - y)*x)

def get_db(y,p):
    return np.mean(p - y)

epochs = 1000
lr = 0.1

w = 0
b = 0

x = np.array([1,2,3,4,5,6])
y = np.array([0,0,0,1,1,1])

for i in range(epochs):

    z = w*x + b
    p = sigmoid(z)

    loss = bce_loss(y, p)

    dw = get_dw(x,y,p)
    db = get_db(y,p)

    w -= lr*dw
    b -= lr*db

    if i % 100 == 0:
        print("Loss:", loss)

print("w:", w, "b:", b)