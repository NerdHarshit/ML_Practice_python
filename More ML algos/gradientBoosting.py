from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris

data = load_iris()

x = data.data
y = data.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size =0.2)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

model.fit(xtrain,ytrain)

score = model.score(xtest,ytest)

print(score)