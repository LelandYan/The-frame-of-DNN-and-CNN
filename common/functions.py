import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, np.int)


def plot_step_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def plot_sigmoid_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def ReLU_function(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax_function(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def cross_entropy_one_hot(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size
