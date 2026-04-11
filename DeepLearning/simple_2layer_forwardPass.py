import numpy as np

X = np.array([[1,2]])

W1 = np.random.randn(2,3)
b1 = np.zeros((1,3))

W2 = np.random.randn(3,1)
b2 = np.zeros((1,1))

def relu(x):
    return np.maximum(0,x)

# forward
z1 = X @ W1 + b1
a1 = relu(z1)

z2 = a1 @ W2 + b2
y = 1/(1+np.exp(-z2)) #sigmoid 

print(y)