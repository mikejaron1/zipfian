import numpy as np
import pandas as pd
from collections import Counter
from itertools import izip
from sklearn.datasets import make_classification

def euclidean_distance(a, b):
    return np.sqrt(np.dot(a - b, a - b))

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

class KNearestNeighbors(object):
    def __init__(self, k=5, distance=euclidean_distance):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = np.zeros((self.X_train.shape[0], X.shape[0]))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = self.distance(x_train, x)
        top_k = y[distances.argsort()[:,:self.k]]  #sort and take top k
        result = np.zeros(X.shape[0])
        for i, values in enumerate(top_k):
            result[i] = Counter(values).most_common(1)[0][0]
        return result

    def classify(self, x):
        distances = []
        for row in self.X:
            distances.append(self.distance(row, x))
        sorted_distances = sorted(izip(distances, self.y), key=lambda a: a[0])
        return Counter(a[1] for a in sorted_distances[:self.k]).most_common(1)[0][0]


if __name__ == '__main__':
    X, y = make_classification(n_features=4, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, class_sep=5,
                               random_state=5)
    knn = KNearestNeighbors(3, cosine_distance)
    knn.fit(X, y)
    print "\tactual\tpredict\tcorrect?"
    for i, (actual, predicted) in enumerate(izip(y, knn.predict(X))):
        print "%d\t%d\t%d\t%d" % (i, actual, predicted, int(actual == predicted))
