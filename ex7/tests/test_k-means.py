# -*- coding: utf-8 -*-

from numpy import *
import numpy.testing
import scipy.io
import scipy.misc
from matplotlib import pyplot

from ex7.kmeans import *


class TestKMeans(numpy.testing.TestCase):

    def setUp(self):
        mat = scipy.io.loadmat("../ex7data2.mat")
        self.X = mat['X']
        self.K = 3

    def test_find_closest_centroids(self):
        initial_centroids = array([[3, 3], [6, 2], [8, 5]])
        idx = find_closest_centroids(self.X, initial_centroids)
        ans = array([[0.], [2.], [1.]])
        numpy.testing.assert_array_equal(idx[0: 3], ans)

    def test_compute_centroids(self):
        initial_centroids = array([[3, 3], [6, 2], [8, 5]])
        idx = find_closest_centroids(self.X, initial_centroids)
        centroids = compute_centroids(self.X, idx, self.K)
        centroids = around(centroids, 6)
        ans = array([[2.428301, 3.157924],
                     [5.813503, 2.633656],
                     [7.119387, 3.616684]])
        numpy.testing.assert_array_equal(centroids, ans)

    def test_init_centroids(self):
        centroids = kmeans_init_centroids(self.X, self.K)
        for el in centroids:
            numpy.testing.assert_equal(el in self.X, True)

    def test_run_kmeans(self):
        max_iters = 10
        centroids = array([[3, 3], [6, 2], [8, 5]])
        run_kmeans(self.X, centroids, max_iters, plot=True)

    def test_image_compression(self):
        A = scipy.misc.imread("../bird_small.png")  # (128, 128, 3) dimension
        pyplot.imshow(A)
        pyplot.show(block=True)

        A = A / 255.0
        img_size = A.shape

        X = A.reshape(img_size[0] * img_size[1], 3)
        K = 16
        max_iters = 10

        initial_centroids = kmeans_init_centroids(X, K)
        centroids, idx = run_kmeans(X, initial_centroids, max_iters)

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