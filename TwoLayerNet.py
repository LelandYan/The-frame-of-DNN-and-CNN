import numpy as np
from common.gradient import numerical_gradient
from common.layers import Affine, Sigmoid, SoftmaxWithLoss, ReLU
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        # self.params = {"W1": np.random.randn(input_size, hidden_size) / np.sqrt(input_size),
        #                "b1": np.zeros(hidden_size),
        #                "W2": np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size),
        #                "b2": np.zeros(output_size)}
        self.params = {"W1": weight_init_std * np.random.randn(input_size, hidden_size),
                       "b1": np.zeros(hidden_size),
                       "W2": weight_init_std * np.random.randn(hidden_size, output_size),
                       "b2": np.zeros(output_size)}
        # 生成层
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {'W1': self.layers["Affine1"].dW, 'b1': self.layers["Affine1"].db, 'W2': self.layers["Affine2"].dW,
                 'b2': self.layers["Affine2"].db}

        return grads


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["b1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b2"].shape)

