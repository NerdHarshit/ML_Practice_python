from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = svm.SVC(kernel='rbf')

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))