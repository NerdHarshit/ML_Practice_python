import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()

x = data.data
y = data.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

model = xgb.XGBClassifier(
    n_estimators = 200,
    learning_rate=0.05,
    max_depth = 5
)

model.fit(xtrain,ytrain)

print(model.score(xtest,ytest))