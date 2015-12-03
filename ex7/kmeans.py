# -*- coding: utf-8 -*-

import sys

import scipy.misc
import scipy.spatial.distance
from numpy import *
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def find_closest_centroids(X, centroids):
    sqdists = scipy.spatial.distance.cdist(centroids, X, 'sqeuclidean')
    idx = argmin(sqdists, axis=0)

    return idx


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
        idx = find_closest_centroids(X, centroids)

        if plot is True:
            data = c_[X, idx]

            for ci, color in enumerate(('r', 'g', 'b')):
                data_tmp = data[data[:, 2] == ci]
                pyplot.plot(data_tmp[:, 0], data_tmp[:, 1], color+'o', markersize=5)
                pyplot.plot(centroids[ci, 0], centroids[ci, 1], color+'*', markersize=17)

            pyplot.show(block=True)

        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def kmeans_init_centroids(X, K):
    return random.permutation(X)[:K]


def compress_image(img_path=''):
    A = scipy.misc.imread(img_path)  # (128, 128, 3) dimension
    pyplot.imshow(A)
    pyplot.show(block=True)

    A = A / 255.0
    img_size = A.shape

    X = A.reshape(img_size[0] * img_size[1], 3)
    K = 16
    max_iters = 10

    initial_centroids = kmeans_init_centroids(X, K)
    centroids, idx = run_kmeans1(X, initial_centroids, max_iters, True)

    # K色に変換
    m = X.shape[0]
    X_recovered = zeros(X.shape)

    for i in range(m):
        k = int(idx[i])
        X_recovered[i] = centroids[k]

    # 元の画像表現に戻す。3次元
    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)
    pyplot.imshow(X_recovered)
    pyplot.show(block=True)


def plot_image_compression(img_path=''):

    A = scipy.misc.imread(img_path)  # (128, 128, 3) dimension

    A = A / 255.0
    img_size = A.shape

    X = A.reshape(img_size[0] * img_size[1], 3)

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("RED",)
    ax.xaxis.label.set_color('red')
    ax.set_ylabel("GREEN")
    ax.yaxis.label.set_color('green')
    ax.set_zlabel("BLUE")
    ax.zaxis.label.set_color('blue')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o', color="red", ms=4, mew=0.5)
    pyplot.show(block=True)


def run_kmeans1(X, initial_centroids, max_iters, plot=False):
    K = initial_centroids.shape[0]
    centroids = copy(initial_centroids)
    idx = None

    for iteration in range(max_iters):
        idx = find_closest_centroids(X, centroids)

        if plot is True:
            colors = ('#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff',
                      '#800000', '#808000', '#008000', '#008080', '#000080', '#800080',
                      '#000000', '#808080', '#c0c0c0', '#ffa500')
            fig = pyplot.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("RED",)
            ax.xaxis.label.set_color('red')
            ax.set_ylabel("GREEN")
            ax.yaxis.label.set_color('green')
            ax.set_zlabel("BLUE")
            ax.zaxis.label.set_color('blue')

            for ci, color in enumerate(colors):
                data_tmp = X[idx == ci]
                ax.plot(data_tmp[:, 0], data_tmp[:, 1], data_tmp[:, 2], 'o', color=color, markersize=4)
                ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], '*', color='#000000', markersize=20)

            pyplot.show(block=True)

        centroids = compute_centroids(X, idx, K)

    return centroids, idx