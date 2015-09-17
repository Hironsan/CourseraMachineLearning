def hypothesis(x, theta):
    return x.dot(theta)


def compute_cost(X, y, theta):
    m = len(y)
    return sum([pow(hypothesis(X[i], theta) - y[i], 2) for i in range(m)]) / (2 * m)



