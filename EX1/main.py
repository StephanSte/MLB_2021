import math
import numpy
from collections import Counter

import numpy as np
import pandas as pd
import sklearn.neighbors
from sklearn import preprocessing


def euclidean_distance(x1, x2):
    # print(x1)
    # print(x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))


def min_max_norm(data):
    # iterate over each column (but class)
    for col in range(0, data.shape[1] - 1):
        min_val = data.iloc[:, col].min()  # min val of current col
        max_val = data.iloc[:, col].max()  # max val
        # iterate over rows
        for row in range(0, data.shape[0]):
            # normalize as per min max norm
            data.iat[row, col] = (data.iat[row, col] - min_val) / (max_val - min_val)
    return data


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
            # sum += k_neighbor_labels[i] for i in range(self.k)
            sum = 0
            for i in range(self.k):
                sum += k_neighbor_labels[i]
            # return mean
            return sum / self.k
        else:
            return []


if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true.values == y_pred) / len(y_true)
        return accuracy


    def getFolds(data, _folds, _i):
        setsize = math.floor(data.shape[0] / _folds)
        start = _i * setsize
        end = start + setsize

        train = data.drop(range(start, end))
        test = data.iloc[start:end]

        X_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]

        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]

        return [X_train, y_train, X_test, y_test]


    def classification(k):
        data = pd.read_csv("EX1/iris.data")

        data = min_max_norm(data)

        folds = 5
        k = k

        results = list()
        resultsSk = list()
        # do it with 1,3,5,7,9 or smth k values
        for i in range(0, folds):
            folders = getFolds(data, folds, i)
            X_train = folders[0]
            y_train = folders[1]
            X_test = folders[2]
            y_test = folders[3]

            clf = KNN(k=k)
            clfSK = sklearn.neighbors.KNeighborsClassifier(k)

            clf.fit(X_train, y_train, 'classifier')
            clfSK.fit(X_train, y_train)

            classifierKnn = clf.predict(X_test)
            classifierSklearn = clfSK.predict(X_test)

            results.append(accuracy(y_test, classifierKnn))
            resultsSk.append(accuracy(y_test, classifierSklearn))

        avg = 0
        for i in range(len(results)):
            avg += results[i]
        avg = avg / len(results)

        avg2 = 0
        for i in range(len(resultsSk)):
            avg2 += resultsSk[i]
        avg2 = avg2 / len(resultsSk)

        print("Avg:    ", avg)
        print("Avg SK: ", avg2)

        # print(resultsSk)
        return [avg, avg2]


    # classification(5)

    def regression(k):
        data = pd.read_csv("EX1/housing.csv")
        folds = 5
        k = k
        mean = list()
        meanSK = list()
        results = list()
        resultsSk = list()
        for i in range(0, folds):
            folders = getFolds(data, folds, i)
            X_train = folders[0]
            y_train = folders[1]
            X_test = folders[2]
            y_test = folders[3]

            clf = KNN(k=k)
            clfSK = sklearn.neighbors.KNeighborsRegressor(k)

            clf.fit(X_train, y_train, 'regressor')
            clfSK.fit(X_train, y_train)

            classifierKnn = clf.predict(X_test)
            classifierSklearn = clfSK.predict(X_test)

            results.append(classifierKnn)
            resultsSk.append(classifierSklearn)

            mean.append(sklearn.metrics.mean_squared_error(y_test, results[i]))
            meanSK.append(sklearn.metrics.mean_squared_error(y_test, resultsSk[i]))

        avg = 0
        for i in range(len(mean)):
            avg += mean[i]
        avg = avg / len(mean)

        avg2 = 0
        for i in range(len(meanSK)):
            avg2 += meanSK[i]
        avg2 = avg2 / len(meanSK)

        print("Avg:    ", avg)
        print("Avg sk: ", avg2)
        return [avg, avg2]


    # regression()

    def hypertrain():
        res = list()
        for i in range(1, 10, 2):
            res.append(classification(i))
        print("=================================================")
    hypertrain()


    def hypertrain2():
        res = list()
        for i in range(1, 10, 2):
            res.append(regression(i))

    hypertrain2()
