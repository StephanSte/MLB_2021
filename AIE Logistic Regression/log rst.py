from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np

X, y = load_iris(return_X_y=True)

kf = KFold(n_splits=10)
kf.get_n_splits(X)

logRegRes = []
dtcRes = []
knnRes = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    logReg = LogisticRegression(random_state=0).fit(X_train, y_train)
    logRegRes.append(logReg.score(X_test, y_test))

    dtc = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    dtcRes.append(dtc.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    knnRes.append(knn.score(X_test, y_test))

print("Logistic Regression Average Result: ", np.average(logRegRes))
print("Decision Tree Average Result: ", np.average(dtcRes))
print("KNN Average Result: ", np.average(knnRes))
