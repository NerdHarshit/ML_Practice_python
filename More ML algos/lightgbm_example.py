import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = load_iris()
X = data.data
y = data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)

# train
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# evaluate
print("Accuracy:", accuracy_score(y_test, pred))

import matplotlib.pyplot as plt

lgb.plot_importance(model)
plt.show()