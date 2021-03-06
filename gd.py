import numpy as np


# http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def stochastic_gradient_descent(X, Y, cost_function, init_param=None, mu=1e-4, decay=1e-2, max_iter=1e4, \
                                min_weight_dist=1e-7, seed=42, intercept=False, verbose=False):
    """
    :param X: our data
    :param Y: our target
    :param init_param: initial params. it's a point where we star SGD
    :param mu: learning rate
    :param decay: learning rate decay
    :param max_iter: max iteration
    :param min_weight_dist: max distance between the weights
    :param seed: random seed if you need
    :param intercept: ask if your X have 1 column
    :param verbose: if you want some comment's while running
    :return:
    """

    assert (X.shape[0] == len(Y))
    if not intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    if init_param is None:
        init_param = np.zeros(X.shape[1])

    weight_dist = np.inf
    param = np.array(init_param)
    hist_params = []
    iter_num = 0
    epoch = 0
    np.random.seed(seed)
    costs = []

    indexes = np.arange(X.shape[0])
    while True:
        learning_rate = mu / (1 + decay * epoch)
        np.random.shuffle(indexes)
        for random_ind in indexes:
            if weight_dist < min_weight_dist or iter_num > max_iter:
                return param, costs

            pred = np.dot(X[random_ind, :], param)
            error = pred - Y[random_ind]
            gradient = X[random_ind, :].T.dot(error) * 2
            param = param - learning_rate * gradient
            weight_dist = np.sqrt(np.sum((param - init_param) ** 2))
            init_param = param
            hist_params.append(param)
            iter_num += 1
        epoch += 1
        cost = cost_function(X, Y, param)
        costs.append(cost)
        if (verbose):
            print 'Epoch {}, iteration {}, weight distance {}, costs {}'.format(epoch, iter_num, weight_dist, cost)


def gradient_descent(X, Y, cost_function, init_param=None, max_iter=1e4, mu=1e-2, intercept=False):
    """
    param:  x, y, init_param, iters, mu
    :param x: data
    :param y: target
    :param init_param: initial weights of your model
    :param max_iter: maximum number of iterations
    :param mu: learning rate
    :param intercept: ask if your X have 1 column
    :param verbose: if you want some comment's while running
    """

    try:
        if not intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        if init_param is None:
            init_param = np.zeros(X.shape[1])
        # let's makr param's type numpy array for our cost function
        param = np.array(init_param)
        # we'll colect history value of param fo visualization
        hist_param = []
        # it's history of cost function
        costs = []

        for i in range(max_iter):
            hist_param.append(param)
            cost = cost_function(X, Y, param)
            costs.append(cost)
            pred = np.dot(X, param)
            error = pred - Y
            gradient = X.T.dot(error) * 2
            param -= mu * gradient
        return param, costs

    except Exception as inst:
        print type(inst)  # the exception instance
        print inst.args  # arguments stored in .args
        print inst


def square_distance_cost_function(X, Y, param):
    param = param.T
    Y = Y.T
    return np.sum((X.dot(param) - Y) ** 2, axis=0) / X.shape[0]
