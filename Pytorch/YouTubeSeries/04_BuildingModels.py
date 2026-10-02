"""第 4 课：使用 PyTorch 构建模型。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
配套视频：https://www.youtube.com/watch?v=OSqIP-mOWOI

本文件的文字讲解依据用户提供的 Building Models 教程原文翻译并排版，
章节顺序与代码顺序均与所提供的教程内容保持一致。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\04_BuildingModels.py
"""


"""
torch.nn.Module 和 torch.nn.Parameter
=======================================

在本视频中，我们将讨论 PyTorch 为构建深度学习网络所提供的一些工具。

除 Parameter 外，本视频中讨论的类都是 torch.nn.Module 的子类。
torch.nn.Module 是 PyTorch 的基类，用于封装 PyTorch 模型及其组件所特有的行为。

torch.nn.Module 的一个重要行为是注册参数。如果某个 Module 子类具有学习权重，
这些权重会表示为 torch.nn.Parameter 的实例。Parameter 类是 torch.Tensor 的
子类；它有一种特殊行为：当 Parameter 被赋值为 Module 的属性时，它会被加入
该 Module 的参数列表。可以通过 Module 类的 parameters() 方法访问这些参数。

下面是一个简单示例：这是一个非常简单的模型，包含两个线性层和一个激活函数。
我们将创建该模型的一个实例，并让它报告自己的参数：
"""

import torch


class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


tinymodel = TinyModel()

print("模型：")
print(tinymodel)

print("\n\n其中一个层：")
print(tinymodel.linear2)

print("\n\n模型参数：")
for param in tinymodel.parameters():
    print(param)

print("\n\n该层的参数：")
for param in tinymodel.linear2.parameters():
    print(param)


"""
以上内容展示了 PyTorch 模型的基本结构：模型中有一个 __init__() 方法，
用于定义模型的层和其他组件；还有一个 forward() 方法，用于执行计算。
请注意，我们可以打印整个模型或它的任意子模块，以了解其结构。


常见的层类型
============

线性层
------

最基本的神经网络层是线性层，也称为全连接层。在这种层中，每一个输入都会影响
该层的每一个输出，而影响程度由该层的权重决定。如果一个模型有 m 个输入和 n 个
输出，那么权重将是一个 m×n 矩阵。例如：
"""

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print("输入：")
print(x)

print("\n\n权重和偏置参数：")
for param in lin.parameters():
    print(param)

y = lin(x)
print("\n\n输出：")
print(y)


"""
如果用 x 乘以线性层的权重矩阵，再加上偏置，就会得到输出向量 y。

还需要注意另一个重要特性：当我们通过 lin.weight 查看该层的权重时，它将自己
报告为一个 Parameter（它是 Tensor 的子类），同时还会告诉我们它正在使用
autograd 跟踪梯度。这是 Parameter 的默认行为，与 Tensor 不同。

线性层广泛用于深度学习模型。最常见的使用位置之一是分类器模型。分类器通常会在
末端设置一个或多个线性层，其中最后一层会有 n 个输出，n 是分类器所处理的类别数。


卷积层
------

卷积层用于处理具有高度空间相关性的数据。它们非常常见于计算机视觉领域，在这里，
卷积层会检测相互邻近的特征组合，然后将它们组成更高层次的特征。卷积层也会出现
在其他场景中——例如自然语言处理应用，在这类应用中，一个单词的直接上下文
（也就是序列中靠近它的其他单词）会影响句子的含义。

我们在此前的视频中已经看到卷积层在 LeNet5 中的应用：
"""

import torch.nn.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 1 个输入图像通道（黑白）、6 个输出通道、5×5 的方形卷积核。
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # 仿射运算：y = Wx + b。
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6×6 来自图像尺寸。
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # 在一个 2×2 窗口上执行最大池化。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果窗口是正方形，只需要指定一个数字。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除 batch 维度外的所有维度。
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


"""
下面分解一下该模型的卷积层中发生了什么。先从 conv1 开始：

LeNet5 的输入应当是一张 1×32×32 的黑白图像。卷积层构造函数的第一个参数是
输入通道数，这里是 1。如果我们构建这个模型是为了查看具有 3 个颜色通道的图像，
那么该参数就会是 3。

卷积层就像一个在图像上进行扫描的窗口，它会寻找自己能够识别的模式。这些模式称为
特征；卷积层的参数之一，就是我们希望它学习的特征数量。构造函数的第二个参数是
输出特征数量。在这里，我们要求该层学习 6 个特征。

刚才我们把卷积层比作一个窗口——那么这个窗口有多大？第三个参数是窗口或卷积核的
尺寸。这里的“5”表示我们选择了一个 5×5 卷积核。如果想使用高度和宽度不同的
卷积核，可以为该参数指定一个元组，例如用 (3, 5) 获得一个 3×5 卷积核。

卷积层的输出是一个激活图，也就是输入 Tensor 中是否存在相应特征的空间表示。
conv1 会输出一个 6×28×28 的 Tensor；6 是特征数量，28 是激活图的高度和宽度。
之所以是 28，是因为用一个宽度为 5 像素的窗口扫描一行 32 像素时，只有 28 个
有效位置。

然后，我们把卷积输出传入 ReLU 激活函数（稍后会进一步介绍激活函数），再传入一个
最大池化层。最大池化层会把激活图中彼此邻近的特征组合起来。它通过缩小 Tensor
实现这一点：把输出中每一组 2×2 单元合并为一个单元，并把组成该单元的 4 个单元
中的最大值赋给它。这样会得到一个较低分辨率的激活图，其尺寸为 6×14×14。

下一个卷积层 conv2 期望 6 个输入通道（对应第一个卷积层所寻找的 6 个特征），
具有 16 个输出通道，并使用 3×3 卷积核。它会输出一个 16×12×12 的激活图，
这个激活图再次经过最大池化层缩小为 16×6×6。在把该输出传递给线性层之前，
它会被重新变形为一个包含 16×6×6=576 个元素的向量，供下一层使用。

PyTorch 提供用于处理一维、二维和三维 Tensor 的卷积层。卷积层的构造函数还有
许多其他可选参数，包括输入上的步长（例如每隔两个或三个位置才扫描一次）、
padding（填充，使窗口可以一直扫描到输入边缘）等。更多信息请参阅相关文档。


循环层
------

循环神经网络（Recurrent Neural Network，RNN）用于处理序列数据——包括科学
仪器的时间序列测量数据、自然语言句子和 DNA 核苷酸序列等。RNN 通过维护一个
隐藏状态来实现这一点；这个隐藏状态就像一种记忆，用来记录到目前为止它在序列中
已经看到的内容。

RNN 层及其变体 LSTM（长短期记忆）和 GRU（门控循环单元）的内部结构较为复杂，
超出了本视频的讨论范围。不过，我们将通过一个基于 LSTM 的词性标注器展示它的
实际形式。词性标注器是一种分类器，用于判断一个单词是名词、动词还是其他词性：
"""


class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # LSTM 以词嵌入为输入，并输出维度为 hidden_dim 的隐藏状态。
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # 该线性层将隐藏状态空间映射到标签空间。
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


"""
该构造函数有四个参数：

vocab_size 是输入词表中的单词数量。每个单词都是 vocab_size 维空间中的一个
one-hot 向量（或单位向量）。

tagset_size 是输出标签集合中的标签数量。

embedding_dim 是词表嵌入空间的大小。嵌入会把一个词表映射到一个低维空间，
在这个空间中，含义相近的单词彼此靠近。

hidden_dim 是 LSTM 记忆的大小。

输入将是一个句子，其中的单词表示为 one-hot 向量的索引。Embedding 层随后会把
这些单词映射到一个 embedding_dim 维空间。LSTM 接收这个嵌入序列并对其进行
迭代，同时生成一个长度为 hidden_dim 的输出向量。最后的线性层充当分类器；
对最后一层的输出应用 log_softmax()，会把输出转换为一组经过归一化的估计概率，
表示给定单词映射到给定标签的可能性。

如果想观察这个网络的实际运行，请查看 pytorch.org 上的“序列模型和 LSTM 网络”
教程。


Transformer
-----------

Transformer 是多用途网络。BERT 等模型使 Transformer 在自然语言处理领域取得了
当前最先进的成果。对 Transformer 架构的讨论超出了本视频的范围，不过 PyTorch
提供了 Transformer 类，可以用它定义 Transformer 模型的整体参数——包括注意力头
的数量、编码器和解码器层的数量、Dropout 和激活函数等。使用正确的参数，甚至可以
只通过这个类构建 BERT 模型。torch.nn.Transformer 还包含用于封装各个组件的类
（TransformerEncoder、TransformerDecoder），以及用于封装子组件的类
（TransformerEncoderLayer、TransformerDecoderLayer）。有关详细信息，
请查看 Transformer 类的相关文档。


其他层与函数
============

数据操作层
----------

还有一些层类型会在模型中执行重要功能，但它们本身不参与学习过程。

最大池化及其对应的最小池化会通过合并单元来缩小 Tensor，并把输入单元中的
最大值赋给输出单元。前面已经见过这种操作。例如：
"""

my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


"""
如果仔细查看上面的数值，就会发现最大池化输出中的每个值，都是 6×6 输入中
对应象限里的最大值。

归一化层会在把某一层的输出传递给下一层之前，重新对其进行中心化和归一化。
对中间 Tensor 进行中心化和缩放会带来许多好处，例如可以使用更高的学习率，
同时避免梯度爆炸或梯度消失。
"""

my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())


"""
运行上面的代码单元时，我们为输入 Tensor 添加了一个较大的缩放因子和偏移量；
你应该会看到输入 Tensor 的 mean() 位于 15 左右。让它通过归一化层后，可以看到
数值变得更小，并且集中在 0 附近——实际上，平均值应该非常小（> 1e-8）。

这很有帮助，因为许多激活函数（将在下文讨论）在接近 0 时具有最强的梯度；
但当输入使它们远离 0 时，有时会出现梯度消失或梯度爆炸。让数据集中在梯度最陡峭
的区域附近，往往意味着更快、更好的学习过程，以及可以使用更高的学习率。

Dropout 层是一种用于鼓励模型形成稀疏表示的工具，也就是推动模型使用更少的数据
进行推理。

Dropout 层通过在训练期间随机把输入 Tensor 的一部分设为 0 来工作；在推理期间，
Dropout 层始终会被关闭。这会迫使模型针对经过遮蔽或缩减的数据集进行学习。例如：
"""

my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))


"""
在上面的示例中，可以看到 Dropout 对一个示例 Tensor 产生的效果。可以使用可选的
p 参数设置单个权重被丢弃的概率；如果不指定该参数，其默认值为 0.5。


激活函数
--------

激活函数使深度学习成为可能。神经网络实际上是一个带有许多参数、用于模拟数学函数
的程序。如果我们所做的只是反复将 Tensor 与各层权重相乘，那么只能模拟线性函数；
而且也没有必要设置许多层，因为整个网络都可以简化为一次矩阵乘法。在各层之间插入
非线性激活函数，使深度学习模型能够模拟任意函数，而不只是线性函数。

torch.nn.Module 提供了封装所有主要激活函数的对象，包括 ReLU 及其多种变体、
Tanh、Hardtanh、Sigmoid 等。它还包括其他函数，例如 Softmax；这些函数在模型的
输出阶段最为有用。


损失函数
--------

损失函数告诉我们，模型的预测距离正确答案有多远。PyTorch 包含多种损失函数，
包括常见的 MSE（均方误差，即 L2 范数）、交叉熵损失和负对数似然损失
（适用于分类器），以及其他损失函数。
"""
