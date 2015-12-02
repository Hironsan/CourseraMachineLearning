# -*- coding: utf-8 -*-

import sys

import scipy.spatial.distance
from numpy import *
from matplotlib import pyplot

"""
def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = zeros((m, 1))
    for i in range(m):
        lowest = sys.maxint
        lowest_index = 0

        for j in range(K):
            cost = X[i] - centroids[j]
            cost = cost.T.dot(cost)
            if cost < lowest:
                lowest_index = j
                lowest = cost

        idx[i] = lowest_index

    return idx
"""


def find_closest_centroids(X, centroids):
    sqdists = scipy.spatial.distance.cdist(centroids, X, 'sqeuclidean')
    idx = argmin(sqdists, axis=0)

    return idx

"""
def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = zeros((K, n))

    for k in range(K):
        temp = X[idx == k]     # クラスタkに割り当てられたデータを取り出す
        count = temp.shape[0]  # クラスタkに割り当てられたデータ数

        for j in range(n):
            centroids[k, j] = sum(temp[:, j]) / count

    return centroids
"""

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = zeros((K, n))

    for k in range(K):
        centroids[k] = mean(X[idx == k], axis=0)

    return centroids


def run_kmeans(X, initial_centroids, max_iters, plot=False):
    K = initial_centroids.shape[0]
    centroids = copy(initial_centroids)
    idx = None

    for iteration in range(max_iters):
        pre = copy(centroids)
        idx = find_closest_centroids(X, centroids)

        if plot is True:
            data = c_[X, idx]

            for ci, color in enumerate(('r', 'g', 'b')):
                data_tmp = data[data[:, 2] == ci]
                pyplot.plot(data_tmp[:, 0], data_tmp[:, 1], color+'o', markersize=5)
                pyplot.plot(centroids[ci, 0], centroids[ci, 1], color+'*', markersize=17)

            #pyplot.plot(centroids[:, 0], centroids[:, 1], 'k*', markersize=17)
            pyplot.show(block=True)


        centroids = compute_centroids(X, idx, K)

        #for i in range(K):
         #   ls1 = [pre[i][0], centroids[i][0]]
          #  ls2 = [pre[i][1], centroids[i][1]]
           # pyplot.plot(ls1, ls2, color='k', linestyle='-', linewidth=2)

    #pyplot.savefig("../img/result.png")

    return centroids, idx


def kmeans_init_centroids(X, K):
    return random.permutation(X)[:K]
