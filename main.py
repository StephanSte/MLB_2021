# k-nearest neighbors on the Iris Flowers Dataset
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import randint


def euclidean_distance(x_test, x_train):
    distance = 0.0
    for i in range(len(x_test) - 1):
        distance += (x_test[i] - x_train[i]) ** 2
    return sqrt(distance)


def knn(row, x_train, y_train, k):
    label = []
    neigbour_dist = []
    # Calc dist for all data
    for i in range(0, len(x_train)):
        dist = euclidean_distance(row, x_train[i])
        neigbour_dist.append(dist)

    # find the k nearest points
    ndist = np.array(neigbour_dist)
    knn = ndist.argsort()[:k]

    # find labels for nearest points
    for i in knn:
        label.append(y_train[i])
    # return most common label
    return max(set(label), key=label.count)


def predict(x_test, x_train, y_train, k):
    predictions = []
    for row in x_test:
        label = knn(row, x_train, y_train, k)
        predictions.append(label)
    return predictions


data_headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data = pd.read_csv("iris.data", names=data_headers)

#split data into k arrays and get train and test xy
folds = 5
for i in range(folds):
    #split das zeug ab
    #in die 4 train und test

    #mach knn mit train und speichers

    #dann mach knn nochmal aber in predict method
    #predictions = predict(X_train, y_train, X_test, 3)
    #vergelcihe das predictete mit dem echten??






#[plt.scatter(X[i][0], X[i][1], color='y') for i in range(len(y))]
#[plt.scatter(X[i][0], X[i][1], color='b' if y[i] == 0 else 'r') for i in range(len(y))]
#plt.show()

#split into folds => 150 und man nimmt immer 15 weg zb und hat dadurch 10 verschiedene sets an daten
# diese daten mit knn berechnen und dann am schluss mit acutal wert vergleichen