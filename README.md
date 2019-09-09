# DNN : The frame of Deep Neural NetWork

code for the DNN with python and numpy etc.

### Get Started

### code Requirements

- Python3.6
- numpy>=1.16.4
- matplotlib>=3.1.0

### Downloading the code and model

mdoel: TwoLayerNet 、MultiLayerNetExtend etc.

### Runing the Demo Script

After downloading the models, you should be able to use model for your build network. We provide a demo script `demo.py` to test if the repo is installed correctly.

```python
python example.py
```

The script will run the classify model for mnist dataset and output the train and test set acc.

```python
import numpy as np
from dataset.mnist import load_mnist # 自己编写mnist文件下载脚本，可能速度会有点慢
import matplotlib.pyplot as plt
from common.trainer import Trainer 
from MultiLayerNetExtend import MultiLayerNexExtend

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 为了再现过拟合，减少学习数据
x_train = x_train[:300]
t_train = t_train[:300]

# 设定是否使用Dropuout、BatchNorm，以及比例 ========================
use_dropout = True  # 不使用Dropout的情况下为False
use_batchnorm = True # 不使用BatchNorm的情况下为False
dropout_ratio = 0.2 # Dropout节点保存率
# ====================================================

# 构建一个输入为input_size=784，中间层的各层隐藏节点数目为hidden_size_list=[100, 100, 100, 100, 100, 100]，输出为output_size=10
network = MultiLayerNexExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ratio=dropout_ratio, use_batchnorm=True)

# 构建训练器
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)

trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 绘制图形==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

```

### Using DNN in Your Project

```python
│  example.py 
│  gradient_check.py 
│  list.txt
│  MultiLayerNetExtend.py
│  neuralnet_mnist.py
│  README.md
│  TwoLayerNet.py
│  
│          
├─common
│  │  functions.py
│  │  gradient.py
│  │  layers.py
│  │  optimizer.py
│  │  trainer.py
│  └─ __init__.py
│          
│          
├─dataset
│  │  mnist.pkl
│  │  mnist.py
│  │  t10k-images-idx3-ubyte.gz
│  │  t10k-labels-idx1-ubyte.gz
│  │  train-images-idx3-ubyte.gz
│  │  train-labels-idx1-ubyte.gz
│  └─ __init__.py
└─ 
        
```



