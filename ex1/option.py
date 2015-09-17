import numpy as np


file_path = './data/ex1data2.txt'

def load_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    x = data[:, 0:2]
    y = data[:, 2]
    return x, y


def feature_normalize(x):
    mu = []
    sigma = []
    new_x = []
    for i in range(x.shape[1]):
        mu.append(np.average(x[:, i]))
        sigma.append(np.std(x[:, i]))
        new_x.append((x[:, i] - mu[i]) / sigma[i])
    return np.array(new_x).T, np.array(mu), np.array(sigma)


def compute_cost(x, y, theta):
    m = len(y)
    #diff = x * theta - y
    diff = x.dot(theta) - y
    J_t = diff.T.dot(diff) / (2 * m)
    return J_t


def part1():
    x, y = load_data(file_path)
    normalized_x, mu, sigma = feature_normalize(x)
    print(normalized_x)
    print(mu)
    print(sigma)


def part2():
    x, y = load_data(file_path)
    m = len(y)
    theta = np.zeros((2, 1))
    print(compute_cost(x, y, theta))


def part3():
    x, y = load_data(file_path)


if __name__ == '__main__':
    part1()
    part2()
    part3()
