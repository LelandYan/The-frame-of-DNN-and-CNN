import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss, ReLU, BatchNormalization, Dropout
from collections import OrderedDict
from common.gradient import numerical_gradient


class MultiLayerNexExtend:
    """
    具有weight Decay、Dropout、Batch Normalization功能的全连接多层神经网络
    """
    def __init__(self, input_size, hidden_size_list, output_size, activation="relu",
                 weight_init_std="relu", weight_decay_lambda=0,
                 use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        """
        :param input_size: 输入的大小
        :param hidden_size_list: 隐藏层的神经元数量列表
        :param output_size: 输出的大小
        :param activation: "relu" or "sigmoid"
        :param weight_init_std: 指定权重的标准差，
        指定"relu" 或者 "he" 是定为"He"的初始值
        指定"sigmoid" 或者 "xavier" 是定为"Xauver"的初始值
        :param weight_decay_lambda: Weight Decay(L2范数)的强度
        :param use_dropout: 是否使用Dropout
        :param dropout_ratio: Dropout比例
        :param use_batchnorm: 是否只用Batch Normalization
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 初始化权值
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {"sigmoid": Sigmoid, "relu": ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)],
                                                      self.params["b" + str(idx)])
            if self.use_batchnorm:
                self.params["gamma" + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params["beta" + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                         self.params['beta' + str(idx)])

            self.layers["Activation_function" + str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers["Dropout" + str(idx)] = Dropout(dropout_ratio)
        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)], self.params["b" + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        设定权重的初始值
        :param weight_init_std:
        :return:
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        """
        求损失函数
        :param x:输入数据
        :param t: 真是标签
        :param train_flg:是否为模型训练
        :return:
        """

        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T, train_flg=True)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = numerical_gradient(loss_W, self.params["W" + str(idx)])
            grads["b" + str(idx)] = numerical_gradient(loss_W, self.params["b" + str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x, t):

        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = self.layers["Affine" + str(idx)].dW + self.weight_decay_lambda * self.params[
                "W" + str(idx)]
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads
