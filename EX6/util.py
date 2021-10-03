import math as mt
import matplotlib.pyplot as plt
import numpy as np


def is_centroid(sample, clusters):
    for centroid in list(map(lambda x: x.centroid, clusters)):
        if np.array_equal(sample, centroid):
            return True
    return False


def distance_euclid(p, q, dim):
    sum_squared = 0
    for i in range(dim):
        sum_squared += (p[i] - q[i]) ** 2
    return mt.sqrt(sum_squared)


def nearest_cluster(x, clusters):
    min_distance = distance_euclid(x, clusters[0].centroid, 2)
    min_index = 0

    for i, cluster in enumerate(clusters):
        cur = distance_euclid(x, cluster.centroid, 2)
        if cur < min_distance:
            min_distance = cur
            min_index = i

    return min_index


def plot_clusters(clusters):
    X1 = []
    X2 = []
    cluster_indices = []
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster.centroid[0],
                    cluster.centroid[1],
                    c='r',
                    marker='o',
                    s=50,
                    edgecolor='k',
                    zorder=1)
        for member in cluster.members:
            X1.append(member[0])
            X2.append(member[1])
            cluster_indices.append(i)

    plt.scatter(X1,
                X2,
                c=cluster_indices,
                marker='o',
                s=25,
                edgecolor='k',
                zorder=0)

    plt.show()