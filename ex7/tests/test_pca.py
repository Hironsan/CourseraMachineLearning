# -*- coding: utf-8 -*-

from numpy import *
import numpy.testing
import scipy.io

from ex7.kmeans import *


class TestPCA(numpy.testing.TestCase):

    def setUp(self):
        mat = scipy.io.loadmat("../data/ex7data1.mat")
        self.X = mat['X']
        self.K = 3

    def test_find_closest_centroids(self):
        pass