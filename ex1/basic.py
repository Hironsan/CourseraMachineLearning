import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from computeCost import compute_cost
from gradientDescent import gradient_descent, update_theta

def part1():
    print('Identity Matrix')
    print np.eye(5)


def part2_1():
    plt.xlim([4, 24])
    plt.ylim([-5, 25])
    plt.xlabel('Poppulation of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'rx')
    plt.savefig('2-1.png')


def part2_2():
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    m = len(y)
    X = np.c_[np.ones((m, 1)), x]
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    cost = compute_cost(X, y, theta)
    theta = gradient_descent(X, y, theta, alpha, iterations)
    print('cost: {0}'.format(cost))
    print('theta: {0}'.format(theta))

    predict1 = np.array([1, 3.5]).dot(theta)
    predict2 = np.array([1, 7]).dot(theta)
    print('predict1: {0}'.format(predict1))
    print('predict2: {0}'.format(predict2))

    x = np.arange(5, 22, 0.1)
    y = [theta[0] + theta[1] * xi for xi in x]
    plt.plot(x, y)
    plt.savefig('2-2.png')


def part2_4():
    data = np.loadtxt('./data/ex1data1.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    m = len(y)
    y = y.reshape(m, 1)
    X = np.c_[np.ones((m, 1)), x]
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, v0 in enumerate(theta0_vals):
        for j, v1 in enumerate(theta1_vals):
            t = np.array((v0, v1))
            J_vals[i, j] = compute_cost(X, y, t)[0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    R, P = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(R, P, J_vals)
    plt.savefig('2-4_surface.png')

    fig = plt.figure()
    plt.contour(R, P, J_vals.T, np.logspace(-2, 3, 20))
    plt.xlabel(r'${\Theta}_0$')
    plt.ylabel(r'${\Theta}_1$')
    plt.savefig('2-4_contour.png')

if __name__ == '__main__':
    part1()
    part2_1()
    part2_2()
    part2_4()