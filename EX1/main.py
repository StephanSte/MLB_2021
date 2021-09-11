import math
from collections import Counter

import numpy as np
import pandas as pd


def euclidean_distance(x1, x2):
    # print(x1)
    # print(x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.type = "classifier"
        self.k = k

    def fit(self, X, y, type):
        self.type = type
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # find the likely labels for the data in X (data to have labels predicted) and return it in a np array
        y_pred = [self._predict(x) for x in X_test.values]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train.values]
        # Sort by distance and return indices of the k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train.values[i] for i in k_idx]
        # return the most common class label
        if self.type == 'classifier':
            most_common = Counter(k_neighbor_labels).most_common(1)
            return most_common[0][0]
        elif self.type == 'regressor':
            print("lalala")
            sum = 0
            for i in range(self.k):
                print(distances[i][1])
                sum += distances[i][1]
            # return mean
            return sum / self.k
        else:
            return []


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    #from sklearn import datasets
    #from sklearn.model_selection import train_test_split
    #from sklearn.neighbors import KNeighborsClassifier


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true.values == y_pred) / len(y_true)
        return accuracy


    def getFolds(givenData, _folds, _i):
        setsize = math.floor(data.shape[0] / _folds)
        start = _i * setsize
        end = start + setsize

        train = givenData.drop(range(start, end))
        test = givenData.iloc[start:end]

        X_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]

        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]

        return [X_train, y_train, X_test, y_test]


    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target

    # X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=7777
    # )

    # data_headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    # data = pd.read_csv("EX1/iris.data", names=data_headers)
    # data = pd.read_csv("EX1/iris.data")
    data = pd.read_csv(r"EX1/iris.data")

    folds = 5
    k = 7
    results = list()
    # do it with 1,3,5,7,9 or smth k values
    for i in range(0, folds):
        folders = getFolds(data, folds, i)
        X_train = folders[0]
        y_train = folders[1]
        X_test = folders[2]
        y_test = folders[3]

        clf = KNN(k=k)
        # classifier_sklearn = KNeighborsClassifier(k)
        # classifier_sklearn.fit(X_train, y_train)

        regressor = 'regressor'
        classifier = 'classifier'
        clf.fit(X_train, y_train, classifier)
        classifierKnn = clf.predict(X_test)
        results.append(accuracy(y_test, classifierKnn))
    print(results)
