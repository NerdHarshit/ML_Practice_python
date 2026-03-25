from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()

x = data.data

y = data.target

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2)

model = DecisionTreeClassifier(max_depth=3)

model.fit(xtrain,ytrain)

accuracy = model.score(xtest,ytest)

print(accuracy)