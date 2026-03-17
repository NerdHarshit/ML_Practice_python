'''
Question from gpt - 

Part 6 — Experiment (20 min) 

Change dataset:

y = 3x + 5

Example:

X = [1,2,3,4]
y = [8,11,14,17]

Train again.

Check if model learns:

w ≈ 3
b ≈ 5

If yes → you understand gradient descent.

'''

import numpy as np

x = np.array([1,2,3,4])
y = np.array([8,11,14,17])

#goal is to make model learn that y = 3x + 5 as the relationship between x and y as given in Q
#now define y_pred and lr and loop iterations and for us now w and b are unknown

lr = 0.1
training_iterations = 100

#init w and b to random 4 numbers

rng = np.random.default_rng()

w = rng.random()
b = rng.random()

y_pred = w*x + b #initally y+pred too is full of shit

def get_mse_loss(y,y_p):
    mse = np.mean((y-y_p)**2)
    return mse

def get_dw(x,y,yp):
    dw = -2*np.mean(x*(y-yp))
    return dw

def get_db(y,yp):
    db = -2*np.mean(y-yp)
    return db

while training_iterations !=0:
    dw = get_dw(x,y,y_pred)
    db = get_db(y,y_pred)

    w = w - lr*dw
    b = b - lr*db

    y_pred = w*x + b

    print("mse is " ,get_mse_loss(y,y_pred))
    training_iterations -=1

print(w,b)
    





