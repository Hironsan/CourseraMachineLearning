from numpy import *
import numpy.testing
import scipy.misc, scipy.io, scipy.optimize, scipy.cluster.vq

from ex7.kmeans import *


class TestKMeans(numpy.testing.TestCase):

    def setUp(self):
        mat = scipy.io.loadmat("../ex7data2.mat" )
        self.X = mat['X']
        self.K = 3

    def test_find_closest_centroids(self):
        initial_centroids = array([[3, 3], [6, 2], [8, 5]])
        idx = find_closest_centroids(self.X, initial_centroids)
        ans = array([[1.], [3.], [2.]])
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