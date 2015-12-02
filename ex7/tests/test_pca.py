# -*- coding: utf-8 -*-

from numpy import *
import numpy.testing
import scipy.io
import scipy.misc
from matplotlib import pyplot

from ex7.kmeans import *


class TestKMeans(numpy.testing.TestCase):

    def setUp(self):
        mat = scipy.io.loadmat("../ex7data1.mat")
        self.X = mat['X']
        self.K = 3