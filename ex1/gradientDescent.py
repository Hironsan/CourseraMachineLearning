import numpy as np
from computeCost import hypothesis


def update_theta(X, y, theta, alpha):
    new_theta = []
    m = len(y)
    for j, theta_i in enumerate(theta):
        new_theta.append((theta_i - alpha * sum([(hypothesis(X[i], theta) - y[i]) * X[i][j] for i in range(m)]) / float(m)))
    return np.array(new_theta)


def gradient_descent(X, y, theta, alpha, iterations):
    for i in range(iterations):
        theta = update_theta(X, y, theta, alpha)
    return theta