# -*- coding: utf-8 -*-
from numpy import *
import scipy.misc
from matplotlib import pyplot


def find_closest_centroids(X, centroids):
    K = shape(centroids)[0]
    m = shape(X)[0]
    idx = zeros((m, 1))
    for i in range(m):
        lowest = 999
        lowest_index = 0

        for k in range(K):
            cost = X[i] - centroids[k]
            cost = cost.T.dot(cost)
            if cost < lowest:
                lowest_index = k
                lowest = cost

        idx[i] = lowest_index

    return idx + 1  # add 1, since python's index starts at 0


def compute_centroids(X, idx, K):
    m, n = shape(X)
    centroids = zeros((K, n))
    data = c_[X, idx]  # append the cluster index to the X

    for k in range(1, K+1):
        temp = data[data[:, n] == k]  # クラスタkに割り当てられたデータを取り出す
        count = shape(temp)[0]        # クラスタkに割り当てられたデータ数

        for j in range(n):
            centroids[k-1, j] = sum(temp[:, j]) / count

    return centroids


def run_kmeans(X, initial_centroids, max_iters, plot=False):
    K = shape(initial_centroids)[0]
    centroids = copy(initial_centroids)
    idx = None

    for iteration in range(max_iters):
        idx = find_closest_centroids(X, centroids)

        if plot is True:
            data = c_[X, idx]

            data_1 = data[data[:, 2] == 1]
            pyplot.plot(data_1[:, 0], data_1[:, 1], 'ro', markersize=5)

            data_2 = data[data[:, 2] == 2]
            pyplot.plot(data_2[:, 0], data_2[:, 1], 'go', markersize=5)

            data_3 = data[data[:, 2] == 3]
            pyplot.plot(data_3[:, 0], data_3[:, 1], 'bo', markersize=5)

            pyplot.plot(centroids[:, 0], centroids[:, 1], 'k*', markersize=17)

            pyplot.show(block=True)

        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def kmeans_init_centroids(X, K):
    return random.permutation(X)[:K]


def part1_4():
    A = scipy.misc.imread("./bird_small.png")  # (128, 128, 3) dimension

    pyplot.imshow(A)
    pyplot.show(block=True)

    A = A / 255.0
    img_size = shape(A)

    X = A.reshape(img_size[0] * img_size[1], 3)
    K = 16
    max_iters = 10

    initial_centroids = kmeans_init_centroids(X, K)
    centroids, idx = run_kmeans(X, initial_centroids, max_iters)

    # mapping the centroids back to compressed image,
    # e.g. all pixels in that cluster shares the same color as the centroid
    m = shape(X)[0]
    X_recovered = zeros(shape(X))

    for i in range(m):
        k = int(idx[i]) - 1
        X_recovered[i] = centroids[k]

    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)
    pyplot.imshow(X_recovered)
    pyplot.show(block=True)


def main():
    set_printoptions(precision=6, linewidth=200)
    part1_4()


if __name__ == '__main__':
    main()