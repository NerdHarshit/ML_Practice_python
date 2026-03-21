'''here we demonstrate how to do test train splitting on a dataset
and comparing the loss values of test and train split to see if model over or under fits'''

#this one does it with simple linear regression but its applicable as it is in every other code in ml

import numpy as np

'''
ideally split the dataset ie y and x matrices into 80 - 20 split
irl you will be given the csv file ..from that you will seperat out the column of Y 
rest of the dataset becomes the x matrix

'''
#uncomment whatever you are using

'''
Dataset 2 - more noise
import numpy as np
np.random.seed(1)

X = np.linspace(1, 100, 200)
noise = np.random.randn(200) * 15

y = 2.5*X + 20 + noise
'''

'''
dataset 3 - has overfitting risk
import numpy as np
np.random.seed(2)

X = np.linspace(1, 50, 80)
noise = np.random.randn(80) * 20

y = 0.5*X**2 + 3*X + 10 + noise
'''

'''
dataset 4 - has underfitting risk
import numpy as np
np.random.seed(3)

X = np.linspace(1, 20, 20)
noise = np.random.randn(20) * 10

y = 4*X + 5 + noise
'''

import numpy as np

#dataset 1
np.random.seed(0)
x = np.linspace(1, 50, 100)
noise = np.random.randn(100) * 5
y = 3*x + 10 + noise



# Shuffle (IMPORTANT)
indices = np.random.permutation(len(x))
x = x[indices]
y = y[indices]

lr = 0.001
#very high iterations as lr very less and thus to get better value of 'b' we must do this.. btw at 1000 iterations we get w = 3 and b = 6
training_iterations = 5000 
w = 0
b = 0

split = int(0.8*len(x))

x_train = x[:split]
x_test = x[split:]

y_train = y[:split]
y_test = y[split:]

def get_mse_loss(y,y_p):
    return np.mean((y-y_p)**2)

def get_dw(x,y,yp):
    return -2*np.mean(x*(y-yp))

def get_db(y,yp):
    return -2*np.mean(y-yp)

for i in range(training_iterations):

    y_pred = w*x_train + b

    dw = get_dw(x_train,y_train,y_pred)
    db = get_db(y_train,y_pred)

    w -= lr*dw
    b -= lr*db

    if i % 100 == 0:
        print("train mse:", get_mse_loss(y_train,y_pred))

print("w:", w, "b:", b)

test_vals = x_test*w + b
print("test loss:", get_mse_loss(y_test,test_vals))