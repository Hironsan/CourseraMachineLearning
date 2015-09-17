# -*- coding: utf-8 -*-
import scipy.io
import scipy.misc
import scipy.optimize
import scipy.special
from numpy import *
from matplotlib import pyplot


digits_data_path = './data/ex3data1.mat'
weights_data_path = './data/ex3weights.mat'


def displayData(X):
    width = 20
    rows, cols = 10, 10
    out = zeros((width * rows, width * cols))
    m = X.shape[0]  # データ数

    # 5000個のデータセットから適当に100個選ぶ
    rand_indices = random.permutation(m)[0: rows * cols]

    counter = 0
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x * width
            start_y = y * width
            out[start_x: start_x + width, start_y: start_y + width] = X[rand_indices[counter]].reshape(width, width).T
            counter += 1

    img = scipy.misc.toimage(out)
    figure = pyplot.figure()
    pyplot.tick_params(labelbottom="off")
    pyplot.tick_params(labelleft="off")
    pyplot.set_cmap(pyplot.gray())
    axes = figure.add_subplot(111)
    axes.imshow(img)
    pyplot.savefig("digits.png")


def sigmoid(z):
    return scipy.special.expit(z)


def computeCost(theta, X, y, lamda):
    m = X.shape[0]
    hypo = sigmoid(X.dot(theta))
    term1 = log(hypo).dot(-y)
    term2 = log(1.0 - hypo).dot(1 - y)
    cost_term = (term1 - term2) / m
    reg_term = theta[1:].T.dot(theta[1:]) * lamda / (2 * m)
    return cost_term + reg_term


def gradientCost(theta, X, y, lamda):
    m = X.shape[0]
    grad = X.T.dot(sigmoid(X.dot(theta)) - y) / m
    grad[1:] = grad[1:] + ((theta[1:] * lamda) / m)
    return grad


def oneVsAll(X, y, num_classes, lamda):
    m, n = X.shape
    X = c_[ones((m, 1)), X]
    all_theta = zeros((n+1, num_classes))

    # 各クラスに対して最適なthetaをもとめる。
    for k in range(0, num_classes):
        theta = zeros((n+1, 1)).reshape(-1)
        # 0, 1のベクトルに変換
        temp_y = ((y == (k+1)) + 0).reshape(-1)
        # コストが小さいシータを求める
        result = scipy.optimize.fmin_cg(computeCost, fprime=gradientCost, x0=theta, args=(X, temp_y, lamda), maxiter=50, disp=False)
        all_theta[:, k] = result

    return all_theta


def predictOneVsAll(theta, X, y):
    m, n = X.shape
    X = c_[ones((m, 1)), X]

    correct = 0
    for i in range(0, m):
        prediction = argmax(theta.T.dot(X[i])) + 1
        actual = y[i]
        if actual == prediction:
            correct += 1
    print('Accuracy: %.2f%%' % (correct * 100.0 / m))


def loadmat(file_path, *names):
    mat = scipy.io.loadmat(file_path)
    return [mat[name] for name in names]


def part1_2():
    X, y = loadmat(digits_data_path, 'X', 'y')
    displayData(X)


def part1_3():
    X, y = loadmat(digits_data_path, 'X', 'y')
    num_labels = 10
    lamda = 0.1
    theta = oneVsAll(X, y, num_labels, lamda)


def part1_4():
    X, y = loadmat(digits_data_path, 'X', 'y')
    num_labels = 10
    lamda = 0.1
    theta = oneVsAll(X, y, num_labels, lamda)
    predictOneVsAll(theta, X, y)


def part2_1():
    X, y = loadmat(digits_data_path, 'X', 'y')
    theta1, theta2 = loadmat(weights_data_path, 'Theta1', 'Theta2')
    m, n = X.shape
    X = c_[ones((m, 1)), X]
    A = c_[ones((m, 1)), sigmoid(theta1.dot(X.T)).T]
    out = theta2.dot(A.T).T
    correct = 0
    for i in range(0, m):
        prediction = argmax(out[i]) + 1
        correct += prediction == y[i]
    print('Accuracy: %.2f%%' % (correct * 100.0 / m))


def main():
    part1_2()
    part1_3()
    print('Logistic Regression')
    part1_4()
    print('')
    print('Neural Network')
    part2_1()

if __name__ == '__main__':
    main()