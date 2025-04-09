import datetime
import numpy as np
import random
import math
import pandas as pd
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import k_means
from sklearn.preprocessing import MinMaxScaler

def findClosestCentroids(X, w, centroids):
    K = np.size(centroids, 0)
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]
    for i in range(n):
        subs = centroids - X[i, :]
        w_dimension2 = np.power(subs, 2)
        w_dimension2 = np.multiply(w, w_dimension2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(K)
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    return idx

def computeWeight(X, centroid, idx, K, belta, w):
    n, m = X.shape
    weight = np.zeros((1, m), dtype=float)
    weight = np.array(w)
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.power((temp - centroid[k, :]), 2) * w
        D = D + np.sum(distance2, axis=0)
    belta = 10
    e = 1 / float(belta - 1)
    D = D + 0.0000001
    for j in range(m):
        temp = D[0][j] / D[0]
        temp_e = np.power(temp, e)
        temp_sum = np.sum(temp_e, axis=0)
        if temp_sum == 0:
            weight[0][j] = 0
        else:
            weight[0][j] = 1 / temp_sum
    return weight

def costFunction(X, K, centroids, idx, w, belta):
    n, m = X.shape
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.power((temp - centroids[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0)
    cost = np.sum(w ** belta * D)
    return cost


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index - 1):
        if costF[i] < costF[i + 1]:
            return False
    if index >= max_iter:
        return True
    elif costF[index - 1] == costF[index - 2] == costF[index - 3]:
        return True
    elif abs(costF[index - 1] - costF[index - 2]) < 0.001:
        return True
    return 'continue'

def wkmeans(X, K, centroids, belta, max_iter):
    n, m = X.shape
    costF = []
    centroids = np.array(centroids)
    r = np.ones((1, m))
    w = np.divide(r, r.sum())
    belta = 10
    if max_iter != 1:
        for i in range(max_iter):
            idx = findClosestCentroids(X, w, centroids)
            w = computeWeight(X, centroids, idx, K, belta, w)
            c = costFunction(X, K, centroids, idx, w, belta)
            costF.append(round(c, 4))
            if i < 2:
                continue
            flag = isConvergence(costF, max_iter)
            if flag == 'continue':
                continue
            elif flag:
                best_labels = idx
                best_centers = centroids
                isConverge = True
                return isConverge, best_labels, best_centers, costF, w
            else:
                isConverge = False
                return isConverge, None, None, costF, w
    else:
        idx = findClosestCentroids(X, w, centroids)
        w = computeWeight(X, centroids, idx, K, belta, w)
        best_labels = idx
        best_centers = centroids
        isConverge = True
        return isConverge, best_labels, best_centers, costF, w

class WKMeans:
    def __init__(self, n_clusters=3, max_iter=10, belta=7.0,centers=[], w=[]):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.belta = belta
        self.centers = centers
        self.w = w

    def fit(self, X):
        self.isConverge, self.best_labels, self.best_centers, self.cost, self.w = wkmeans(
            X=X, K=self.n_clusters, centroids=self.centers, max_iter=self.max_iter, belta=self.belta
        )
        return self

    def fit_predict(self, X, y=None):
        if self.fit(X).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'

    def get_params(self):
        return self.isConverge, self.n_clusters, self.belta, 'WKME'

    def get_cost(self):
        return self.cost


def load_data():
    data_path = "./synthetic/test data/"
    df = pd.read_csv(data_path + "wine.csv", header=None)
    data = df.values
    n, m = data.shape
    y = data[:, m-1] - 1
    data = np.delete(data, -1, axis=1)
    x = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    return x, y

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    # print(D)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind=np.asarray(ind)
    ind=np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

if __name__ == '__main__':
    x, y = load_data()
    n, m = x.shape
    k = 3
    centers=k_means(X=x, n_clusters=k, n_init=2, init='k-means++', random_state=5)[0]
    model = WKMeans(n_clusters=k, belta=2, centers=centers)
    indicators = []
    for i in range(1):
        while True:
            y_pred = model.fit_predict(x)
            w = model.w
            if model.isConverge == True:
                nmi = normalized_mutual_info_score(y, y_pred)
                end_time = datetime.datetime.now()
                score = metrics.cluster.adjusted_rand_score(y, y_pred)
                ACC = acc(y, y_pred)
                indicators.append([score, nmi, ACC])
                break
            else:
                break
    indicators = np.array(indicators)
    n = indicators.shape[0]
    average = indicators.sum(axis=0) / n