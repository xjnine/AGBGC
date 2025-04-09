import warnings
import numpy as np
from sklearn.cluster import k_means
warnings.filterwarnings('ignore')

def vec_euc_dist(x1, x2, w):
    p = (x1 - x2) ** 2
    p = p * w
    return p

def wt_euc_dist(x1, x2, w):
    p = (x1 - x2) ** 2
    p = sum(w * p)
    return p

def lwkmeans(X, K, alpha = None, beta = None, tmax = None):

    n, m = X.shape
    weight = np.ones((1, m)) / m
    weight = weight.flatten()
    w = np.zeros((1, m))

    label = k_means(X, K, init="k-means++",n_init=10 )[1]
    lambda1 = 0.0005
    dist = np.zeros(K)
    D = np.zeros((1, m))
    x_mean = np.mean(X, axis = 0 )
    nargin = 0
    M = np.zeros((K,m))

    if nargin < 5 or alpha is None:
        alpha = 1
    if nargin < 6 or beta is None:
        beta = 4
    if nargin < 7 or tmax is None:
        tmax = 30

    on = np.ones((1,m))
    for i in range(n):
        X[i, :] = X[i, :] - x_mean
    lambda1 = lambda1/(m*m)
    for i in range(m):
        X[:, i] = X[:, i] / np.std(X[:, i])
    for i in range(K):
        I = np.where(label == i)
        M[i] = np.mean(X[I], axis=0)
    D = np.zeros((1,m))
    for i in range(K):
        I = np.array(np.where(label == i)).flatten()
        for j in I:
            D += vec_euc_dist(X[j,:], M[i,:], on)
    D = 1.0 / D
    D **= 1 / (beta - 1)
    alpha = D.sum()
    alpha = 1 / alpha
    alpha **= beta - 1
    for iter in range(tmax):
        for i in range(K):
            I = np.array(np.where(label == i)).flatten()
            M[i] = X[I,:].mean(axis=0)
        D = np.zeros((1,m))
        for i in range(K):
            I = np.array(np.where(label == i)).flatten()
            for j in I:
                D += vec_euc_dist(X[j,:], M[i,:], on)
        D = D.flatten()
        for i in range(m):
            if alpha > lambda1 * D[i]:
                weight[i] = alpha / D[i] - lambda1
                weight[i] **= 1 / (beta - 1)
            else:
                weight[i] = 0
    w =  weight ** beta + lambda1 * weight
    for i in range(n):
        for j in range(K):
            dist[j] = wt_euc_dist(X[i,:], M[j,:], w)
        label[i] = dist.argmin()

    L = sum(weight > 0)
    
    return label,weight,L,alpha
