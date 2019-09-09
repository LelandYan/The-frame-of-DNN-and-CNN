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
    if a.ndim == 2:
        a = a.T
        a = a - np.max(a, axis=0)
        y = np.exp(a) / np.sum(np.exp(a), axis=0)
        return y.T
    a = a - np.max(a)
    return np.exp(a) / np.sum(np.exp(a))


def softmax_loss(X, t):
    y = softmax_function(X)
    return cross_entropy_error(y, t)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
