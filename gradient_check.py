import os
import sys
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
from TwoLayerNet import TwoLayerNet

# 为了引入父目录中的文件而进行设定
sys.path.append(os.pardir)

# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#
#
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# #
# # # print(x_train.shape)
# # # print(t_train.shape)
# # # print(x_test.shape)
# # # print(t_test.shape)
# #
# img = x_test[20]
# label = t_test[20]
# print(label)
#
# img = img.reshape(28, 28)
# # print(img.shape)
#
# img_show(img)

if __name__ == '__main__':
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    count = 0
    theta1 = 0
    theta2 = 0
    for key in grad_numerical.keys():
        print(key+": ", np.average(np.abs(grad_backprop[key] - grad_numerical[key])))
        # diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        new_vector1 = np.reshape(grad_numerical[key], (1, -1))
        new_vector2 = np.reshape(grad_backprop[key], (1, -1))
        if count == 0:
            theta1 = new_vector1
            theta2 = new_vector2
        else:
            theta1 = np.concatenate((theta1, new_vector1), axis=1)
            theta2 = np.concatenate((theta2, new_vector2), axis=1)
        count += 1
    diff = np.linalg.norm(theta1 - theta2) / (
            np.linalg.norm(theta1) + np.linalg.norm(theta2))
    print(diff)
    # # print(np.linalg.norm(theta1 - theta2))
    # # print(np.linalg.norm(theta1) + np.linalg.norm(theta2))
    if diff < 1e-6:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
