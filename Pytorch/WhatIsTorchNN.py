"""什么是 torch.nn？

原教程：What is torch.nn really?
https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
作者：Jeremy Howard（fast.ai）；感谢 Rachel Thomas 和 Francisco Ingham。
创建于 2018-12-26；最后更新于 2025-01-24；最后核验于 2024-11-05。

以下讲解依据用户提供的原教程翻译，并与代码保持原有顺序。
原教程建议以 notebook 而非脚本运行；本文件将 notebook 中用于说明的
非执行代码片段保留为注释，使它可以作为独立 Python 脚本从头运行。

运行环境：Conda 环境 MLLearn，Python 3.9.25，PyTorch 2.7.1+cu118。
从项目根目录运行：
    conda activate MLLearn
    python .\Pytorch\WhatIsTorchNN.py

首次运行会按原教程下载 MNIST pickle 文件到当前工作目录的 data/mnist。
"""

"""
我们建议将本教程作为 notebook 而不是脚本运行。要下载 notebook（.ipynb）文件，
请点击原教程页面顶部的链接。

PyTorch 提供了设计精巧的模块和类：torch.nn、torch.optim、Dataset 和
DataLoader，以帮助你创建和训练神经网络。要充分利用它们，并针对自己的问题进行
定制，就需要真正理解它们具体在做什么。为了建立这种理解，我们首先只用最基本的
PyTorch Tensor 功能，而不使用这些模块提供的功能，在 MNIST 数据集上训练一个
基本的神经网络。然后，我们每次逐步加入 torch.nn、torch.optim、Dataset 或
DataLoader 的一个功能，展示每个部分具体做什么，以及它如何让代码更简洁或更灵活。

本教程假定你已经安装 PyTorch，并熟悉 Tensor 运算的基础知识。如果你熟悉 NumPy
数组运算，会发现这里使用的 PyTorch Tensor 运算与它几乎一样。


MNIST 数据准备
===============

我们将使用经典的 MNIST 数据集，它由手写数字（0 到 9）的黑白图像组成。

我们用 Python 3 标准库中的 pathlib 处理路径，使用 requests 下载数据集。
各模块只在需要用到时导入，这样你能清楚看到每一步实际用了什么。
"""

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)


"""
这个数据集采用 NumPy 数组格式，并通过 pickle 保存。pickle 是 Python 特有的
数据序列化格式。
"""

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


"""
每张图像的大小是 28×28，保存时被展平为长度为 784（=28×28）的一行。
先把其中一张图像重新变形成二维，再查看它。
"""

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# 如果不在 Colab 中，则显示图像。
try:
    import google.colab
except ImportError:
    pyplot.show()
print(x_train.shape)


"""
PyTorch 使用 torch.tensor 而不是 NumPy 数组，因此需要转换数据。
"""

import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())


"""
从零开始构建神经网络（不使用 torch.nn）
==========================================

首先，只用 PyTorch Tensor 运算创建模型。这里假定你已经熟悉神经网络的基础知识。
如果还不熟悉，可以在 course.fast.ai 学习。

PyTorch 提供创建随机 Tensor 或全零 Tensor 的方法。我们将用它们为一个简单线性
模型创建权重和偏置。这些都是普通 Tensor，但有一个非常特殊的补充：我们告诉
PyTorch 它们需要计算梯度。这会使 PyTorch 记录对 Tensor 执行的所有运算，从而
在反向传播时自动计算梯度。

对于权重，我们在初始化之后才设置 requires_grad，因为不希望把初始化这一步纳入
梯度计算。注意，PyTorch 中名称末尾的下划线 _ 表示该运算是原地执行的。

注意：这里用 Xavier 初始化权重，即乘以 1/sqrt(n)。
"""

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


"""
由于 PyTorch 能自动计算梯度，我们可以把任何标准 Python 函数（或可调用对象）
当作模型。因此，直接写一个矩阵乘法和支持广播的加法，创建简单的线性模型。
模型还需要激活函数，所以我们自己编写 log_softmax 并使用它。请记住，虽然
PyTorch 提供了许多现成的损失函数、激活函数等，你也可以用普通 Python 轻松编写
自己的函数。PyTorch 甚至会自动为这些函数生成快速的加速器代码或向量化 CPU 代码。
"""


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)


"""
上面的 @ 表示矩阵乘法。我们将对一个 batch 的数据调用这个函数，这里是 64 张图像。
这就是一次前向传播。由于权重一开始是随机的，这时的预测不会比随机猜测更好。
"""

bs = 64  # batch 大小。

xb = x_train[0:bs]  # 从 x 中取一个小批次。
preds = model(xb)  # 预测值。
preds[0], preds.shape
print(preds[0], preds.shape)


"""
可以看到，preds Tensor 不仅包含数值，还带有梯度函数。稍后我们会用它进行反向传播。

现在自己实现负对数似然损失函数；同样只需使用标准 Python。
"""


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll


"""
先检查随机初始化模型的损失，以便稍后比较执行反向传播后有没有改进。
"""

yb = y_train[0:bs]
print(loss_func(preds, yb))


"""
再实现一个计算模型准确率的函数。对每个预测，如果最大值所在位置的索引与目标值
相同，那么这次预测就是正确的。
"""


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


"""
检查随机模型的准确率，这样可以观察损失改善时准确率是否也提高。
"""

print(accuracy(preds, yb))


"""
现在可以运行训练循环。每次迭代都会：

* 选择大小为 bs 的一个小批次；
* 用模型进行预测；
* 计算损失；
* 通过 loss.backward() 更新模型（这里是 weights 和 bias）的梯度。

然后，我们用这些梯度更新权重和偏置。更新放在 torch.no_grad() 上下文中，因为
我们不希望这些动作被记录到下一次梯度计算中。关于 PyTorch 的 Autograd 如何记录
运算，可以阅读原教程链接的进一步说明。

之后将梯度清零，为下一轮循环做好准备。否则，梯度会持续累加此前所有运算的结果：
loss.backward() 是把新梯度加到已有梯度上，而不是直接替换它们。

提示：可以使用标准 Python 调试器逐步运行 PyTorch 代码，在每一步检查变量的值。
要试用它，请取消注释下面的 set_trace()。
"""

from IPython.core.debugger import set_trace

lr = 0.5  # 学习率。
epochs = 2  # 训练的 epoch 数。

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()


"""
就是这样：我们完全从零创建并训练了一个最小神经网络。由于没有隐藏层，这个模型
实际上是逻辑回归。

现在检查损失和准确率，与先前的结果比较。我们期望损失下降、准确率提高；
运行结果也确实如此。
"""

print(loss_func(model(xb), yb), accuracy(model(xb), yb))


"""
使用 torch.nn.functional
=========================

接下来重构代码，让它完成与之前相同的工作，同时开始利用 PyTorch 的 nn 类，
使代码更简洁、更灵活。此后的每一步都应使代码至少在以下一个方面有所改进：
更短、更容易理解，或者更灵活。

第一步最简单：用 torch.nn.functional 中的函数（通常按惯例导入为 F），代替
手写的激活函数和损失函数，从而缩短代码。这个模块包含 torch.nn 库中的函数；
库中的其他部分则包含类。除了大量损失函数和激活函数，还能在这里找到一些方便的
神经网络构建函数，例如池化函数。这里也有卷积、线性层等对应的函数，但如后文所示，
这些功能通常更适合由库的其他部分处理。

如果使用负对数似然损失和 log_softmax 激活，PyTorch 提供了把两者合在一起的
F.cross_entropy。因此，甚至可以从模型中移除激活函数。
"""

import torch.nn.functional as F

loss_func = F.cross_entropy


def model(xb):
    return xb @ weights + bias


"""
注意，现在 model 函数不再调用 log_softmax。确认损失和准确率与此前相同：
"""

print(loss_func(model(xb), yb), accuracy(model(xb), yb))


"""
使用 nn.Module 重构
====================

接下来使用 nn.Module 和 nn.Parameter，让训练循环更清晰、更简洁。我们继承
nn.Module；它本身是一个类，可以跟踪状态。这里需要创建一个类，保存权重和偏置，
并定义前向传播的方法。nn.Module 提供了许多属性和方法，例如接下来会用到的
.parameters() 和 .zero_grad()。
注意：nn.Module（大写 M）是 PyTorch 特有的概念，本教程中会大量使用它。
不要把它与 Python 中的 module（小写 m）概念混淆；后者是可导入的 Python 代码文件。
"""

from torch import nn


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


"""
现在使用的是对象，而不只是函数，因此首先需要实例化模型：
"""

model = Mnist_Logistic()


"""
现在可以和之前一样计算损失。注意，nn.Module 对象可以像函数一样调用；在背后，
PyTorch 会自动调用我们的 forward 方法。
"""

print(loss_func(model(xb), yb))

"""
此前在训练循环中，我们必须按名称逐个更新参数的值，并手动清零每个参数的梯度：
"""

# 以下是原教程用于对照的旧写法，不在这里重复执行：
# with torch.no_grad():
#     weights -= weights.grad * lr
#     bias -= bias.grad * lr
#     weights.grad.zero_()
#     bias.grad.zero_()


"""
现在可以利用 model.parameters() 和 model.zero_grad()；它们都是 PyTorch
为 nn.Module 定义的方法。这使操作更简洁，也降低了漏掉某个参数的风险，尤其是
当模型更加复杂时：
"""

# 以下是原教程用于对照的新写法；此时尚未对新模型反向传播，故不单独执行：
# with torch.no_grad():
#     for p in model.parameters(): p -= p.grad * lr
#     model.zero_grad()


"""
将这个小训练循环放入 fit 函数，方便稍后再次运行。
"""


def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()


fit()


"""
再次确认损失已经下降：
"""

print(loss_func(model(xb), yb))


"""
使用 nn.Linear 重构
====================

继续重构代码。不再手动定义和初始化 self.weights、self.bias，也不再手动计算
xb @ self.weights + self.bias，而是改用 PyTorch 的 nn.Linear 类表示线性层；
它会替我们完成这些事情。PyTorch 预定义了许多种层，可以大幅简化代码，通常还会
让它运行得更快。
"""


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


"""
像以前一样实例化模型并计算损失：
"""

model = Mnist_Logistic()
print(loss_func(model(xb), yb))


"""
仍然可以使用此前定义的 fit 方法。
"""

fit()

print(loss_func(model(xb), yb))


"""
使用 torch.optim 重构
======================

PyTorch 还提供了包含多种优化算法的 torch.optim 包。可以调用优化器的 step
方法来完成参数更新，而不必手动逐个更新参数。

这样就可以将先前手写的优化步骤：
"""

# 以下为原教程的旧优化步骤，仅用于对照：
# with torch.no_grad():
#     for p in model.parameters(): p -= p.grad * lr
#     model.zero_grad()


"""
替换为：
"""

# 以下为原教程的新优化步骤；opt 在下文才创建：
# opt.step()
# opt.zero_grad()


"""
optim.zero_grad() 会把梯度重置为 0。必须在计算下一个小批次的梯度之前调用它。
"""

from torch import optim


"""
定义一个创建模型和优化器的小函数，方便今后复用。
"""


def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))


"""
使用 Dataset 重构
==================

PyTorch 有一个抽象的 Dataset 类。只要一个对象有 __len__ 函数（由 Python 标准
len 函数调用），以及可用于索引的 __getitem__ 函数，它就可以是 Dataset。
原教程链接的另一篇教程详细演示了如何继承 Dataset，创建自己的
FacialLandmarkDataset 类。

PyTorch 的 TensorDataset 是包装 Tensor 的 Dataset。通过定义长度和索引方式，
它还使我们能沿 Tensor 的第一个维度迭代、索引和切片。训练时，读取自变量和因变量
就可以写在同一行中。
"""

from torch.utils.data import TensorDataset


"""
可以将 x_train 和 y_train 合并到同一个 TensorDataset 中，这样更方便迭代和切片。
"""

train_ds = TensorDataset(x_train, y_train)


"""
此前必须分别遍历 x 和 y 的小批次：
"""

# xb = x_train[start_i:end_i]
# yb = y_train[start_i:end_i]


"""
现在可以在同一步中完成这两件事：
"""

# xb, yb = train_ds[i * bs : i * bs + bs]

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs : i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))


"""
使用 DataLoader 重构
=====================

PyTorch 的 DataLoader 负责管理批次。它可以基于任意 Dataset 创建，使批次迭代
更容易。我们不再需要手动使用 train_ds[i*bs : i*bs+bs]，DataLoader 会自动提供
每一个小批次。
"""

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)


"""
此前的循环像这样遍历批次 (xb, yb)：
"""

# for i in range((n - 1) // bs + 1):
#     xb, yb = train_ds[i * bs : i * bs + bs]
#     pred = model(xb)


"""
现在循环简洁得多，(xb, yb) 会从 DataLoader 自动取出：
"""

# for xb, yb in train_dl:
#     pred = model(xb)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))


"""
借助 PyTorch 的 nn.Module、nn.Parameter、Dataset 和 DataLoader，训练循环
现在明显更短，也更容易理解。接下来加入实际构建有效模型所需的基本功能。


加入验证
========

第一部分只是为了得到能在训练数据上运行的合理训练循环。实际训练时，始终还应该
使用验证集，以便识别是否出现过拟合。

打乱训练数据很重要：它可以防止批次之间产生相关性，并减轻过拟合。另一方面，
无论是否打乱验证集，验证损失都相同。既然打乱需要额外时间，就没有必要打乱
验证数据。

验证集使用的 batch 大小将是训练集的两倍。因为验证不需要反向传播，所占内存更少
（不需要保存梯度）；因此可以利用更大的 batch 更快地计算损失。
"""

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


"""
每个 epoch 结束时，计算并打印验证损失。

注意：训练前始终调用 model.train()，推理前调用 model.eval()。这样，
nn.BatchNorm2d 和 nn.Dropout 等层才能在这两个阶段分别使用适当的行为。
"""

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))


"""
创建 fit() 和 get_data()
========================

训练集与验证集都要经历计算损失的类似过程。现在进行一点重构：将计算一个批次
损失的过程提取到 loss_batch 函数中。

对于训练集，我们传入优化器，使函数执行反向传播；对于验证集，不传入优化器，
所以函数不会执行反向传播。
"""


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


"""
fit 执行训练模型所需的操作，并计算每个 epoch 的训练和验证损失。
"""

import numpy as np


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


"""
get_data 返回训练集和验证集的 DataLoader。
"""


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


"""
现在，取得 DataLoader 并拟合模型的整个过程可以用三行代码完成：
"""

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


"""
这三行基本代码可以训练各种各样的模型。看看能否用它们训练卷积神经网络（CNN）！
"""


"""
切换到 CNN
==========

现在使用三个卷积层构建神经网络。由于上一节的函数都没有假定模型必须具有某种形式，
因此无需修改它们，就能用来训练 CNN。

使用 PyTorch 预定义的 Conv2d 类作为卷积层。我们定义一个含有三个卷积层的 CNN；
每个卷积层后面都接一个 ReLU，最后执行平均池化。注意，view 是 PyTorch 中与
NumPy 的 reshape 对应的操作。
"""


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


lr = 0.1


"""
动量是随机梯度下降的一种变体，它还会考虑之前的更新，通常能使训练更快。
"""

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)


"""
使用 nn.Sequential
==================

torch.nn 还有一个可以简化代码的便利类：Sequential。Sequential 对象会按顺序
运行其中包含的各个模块。这是一种更简单的神经网络写法。

为利用这个类，我们需要能够方便地从给定函数定义自定义层。例如，PyTorch 没有
view 层，而我们的网络需要一个。Lambda 将创建这样的层，以便在通过 Sequential
定义网络时使用。
"""


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)


"""
通过 Sequential 创建的模型很简单：
"""

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)


"""
包装 DataLoader
===============

我们的 CNN 已经相当简洁，但它只能用于 MNIST，因为：

* 它假定输入是长度为 28×28 的向量；
* 它假定 CNN 最终的网格大小为 4×4，因为我们选用了这个大小的平均池化核。

现在去掉这两个假定，让模型能处理任意二维单通道图像。首先，把数据预处理移入
生成器，以便移除最初的 Lambda 层：
"""


def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)


"""
接下来，将 nn.AvgPool2d 替换为 nn.AdaptiveAvgPool2d。后者让我们指定想要的
输出 Tensor 大小，而不是要求特定的输入 Tensor 大小。因此模型可以用于任意
大小的输入。
"""

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


"""
试运行一下：
"""

fit(epochs, model, loss_func, opt, train_dl, valid_dl)


"""
使用加速器
==========

如果你有幸能够使用 CUDA 等加速器（多数云服务提供商的租用价格约为每小时
0.50 美元），就可以用它加速代码。首先检查 PyTorch 中的加速器是否正常工作：
"""

# 如果当前加速器可用，就使用它；否则使用 CPU。
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


"""
更新 preprocess，将批次数据移至加速器：
"""


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(device), y.to(device)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)


"""
最后，将模型移至加速器。
"""

model.to(device)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


"""
现在应该会发现它运行得更快：
"""

fit(epochs, model, loss_func, opt, train_dl, valid_dl)


"""
结束语
======

现在我们有了一个通用的数据流水线和训练循环，可以用于训练许多类型的 PyTorch
模型。要了解训练模型如今可以有多简单，请查看 mnist_sample notebook。

当然，还可以加入许多功能，例如数据增强、超参数调优、训练监控和迁移学习。
fastai 库提供了这些功能。它采用了本教程所展示的相同设计方法，对于想进一步
发展模型的实践者而言，是自然的下一步。

本教程一开始承诺通过示例解释 torch.nn、torch.optim、Dataset 和 DataLoader。
总结一下已经看到的内容：

torch.nn：

* Module：创建一个可以像函数一样调用、同时还能包含状态（例如神经网络层权重）
  的对象。它知道自己包含哪些 Parameter，能清空这些参数的梯度，并能遍历参数
  以更新权重等。
* Parameter：Tensor 的包装器，告诉 Module 它拥有需要在反向传播期间更新的权重。
  只有设置了 requires_grad 属性的 Tensor 才会被更新。
* functional：一个通常按惯例导入为 F 的模块，包含激活函数、损失函数等，
  以及卷积层和线性层等层的无状态版本。

torch.optim：包含 SGD 等优化器，它们在反向传播步骤中更新 Parameter 的权重。

Dataset：具有 __len__ 和 __getitem__ 的对象的抽象接口；TensorDataset
是 PyTorch 提供的其中一个类。

DataLoader：接收任意 Dataset，创建按批次返回数据的迭代器。
"""

