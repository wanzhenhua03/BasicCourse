"""第 5 课：PyTorch 对 TensorBoard 的支持。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html
配套视频：https://www.youtube.com/watch?v=6CEld3hZgqc

以下文字讲解依据用户提供的教程全文翻译、排版，并保持原文顺序。
本地运行适配：数据和日志保存到本文件所在目录；Windows 下使用单进程加载数据。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\05_TensorBoardSupport.py
"""

from pathlib import Path


"""
开始之前
========

要运行本教程，需要安装 PyTorch、TorchVision、Matplotlib 和 TensorBoard。

使用 pip 安装：

    pip install torch torchvision matplotlib tensorboard

安装依赖后，请在安装了这些依赖的 Python 环境中重新启动本 notebook。


简介
====

在本 notebook 中，我们将使用 Fashion-MNIST 数据集训练一个 LeNet-5 的变体。
Fashion-MNIST 由描绘各种服饰的图像块组成；它有十个类别标签，表示图像中服饰的类型。
"""

# PyTorch 模型和训练所需的模块。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 图像数据集和图像处理。
import torchvision
from torchvision.transforms import v2

# 显示图像。
import matplotlib.pyplot as plt
import numpy as np

# PyTorch 对 TensorBoard 的支持。
from torch.utils.tensorboard import SummaryWriter


# 本地运行适配：无论从哪个工作目录启动，数据与日志都保存在本文件所在目录。
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RUNS_DIR = SCRIPT_DIR / "runs"


"""
在 TensorBoard 中显示图像
==========================

首先，我们把数据集中的示例图像添加到 TensorBoard：
"""

# 获取数据集，并为使用数据做好准备。
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))])

# 将训练集和验证集分别保存在数据目录中。
training_set = torchvision.datasets.FashionMNIST(
    DATA_DIR,
    download=True,
    train=True,
    transform=transform)
validation_set = torchvision.datasets.FashionMNIST(
    DATA_DIR,
    download=True,
    train=False,
    transform=transform)

# 本地运行适配：Windows 下使用单进程，避免子进程再次执行脚本顶层代码。
training_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0)

validation_loader = torch.utils.data.DataLoader(validation_set,
                                                 batch_size=64,
                                                 shuffle=False,
                                                 num_workers=0)

# 类别标签。
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# 在 notebook 中显示图像的辅助函数。
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # 还原归一化。
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 取出一批，共 4 张图像。
dataiter = iter(training_loader)
images, labels = next(dataiter)

# 用这些图像创建网格并显示。
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
plt.show()  # 本地运行适配：在独立 Python 脚本中显示 notebook 单元的图像。


"""
上面，我们使用 TorchVision 和 Matplotlib，为输入数据的一个小批次创建了可视化
图像网格。下面，我们调用 SummaryWriter 的 add_image()，把图像写入日志供
TensorBoard 使用；还会调用 flush()，确保图像立即写入磁盘。
"""

# log_dir 参数的默认值是 "runs"，不过明确指定它更合适。
# torch.utils.tensorboard.SummaryWriter 已在上文导入。
writer = SummaryWriter(str(RUNS_DIR / 'fashion_mnist_experiment_1'))

# 将图像数据写入 TensorBoard 日志目录。
writer.add_image('Four Fashion-MNIST Images', img_grid)
writer.flush()

# 查看日志时，在命令行中启动 TensorBoard：
#   tensorboard --logdir=Pytorch/YouTubeSeries/runs
# 然后在浏览器中打开 http://localhost:6006/ 。


"""
如果从命令行启动 TensorBoard，并在新的浏览器标签页中打开它（通常是
localhost:6006），你应该会在 IMAGES 标签页下看到这个图像网格。


绘制标量，观察训练过程
======================

TensorBoard 可用于跟踪训练的进展和效果。下面，我们将运行一个训练循环，跟踪
一些指标，并保存数据供 TensorBoard 使用。

首先，定义一个为图像分类的模型，以及训练所需的优化器和损失函数：
"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if torch.accelerator.is_available():
    print("\n检测到可用加速器。")
else:
    print("\n当前只能使用 CPU。")

if torch.accelerator.is_available():
    accelerator_random = torch.rand(
        2,
        2,
        device=torch.accelerator.current_accelerator(),
    )
    print("加速器上的随机 Tensor：")
    print(accelerator_random)
else:
    print("跳过加速器 Tensor 示例。")

my_device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
print("实际使用的设备：", my_device)


net = Net()
net.to(my_device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


"""
现在训练一个 epoch，并且每经过 1000 个 batch，就比较训练集与验证集上的损失：
"""

# 本地实验配置：这里实际训练 10 个 epoch，batch size 为 64；
# 每个 epoch 只有 938 个 batch，因此每 100 个 batch 和每轮结束时比较损失。
log_interval = 100
print(len(validation_loader))
for epoch in range(10):  # 多次遍历数据集。
    running_loss = 0.0

    for i, data in enumerate(training_loader, 0):
        # 基本训练循环。
        inputs, labels = data
        inputs = inputs.to(my_device)
        labels = labels.to(my_device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % log_interval == 0 or i + 1 == len(training_loader):
            print(f'Batch {i + 1}')
            # 在验证集上检查。
            running_vloss = 0.0

            # 评估模式可省略某些模型特有的操作，例如 Dropout 层。
            net.eval()  # 切换为评估模式，例如关闭正则化。
            with torch.no_grad():
                for j, vdata in enumerate(validation_loader, 0):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(my_device)
                    vlabels = vlabels.to(my_device)
                    voutputs = net(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    running_vloss += vloss.item()
            net.train()  # 切换回训练模式，例如重新启用正则化。

            # 最后一个不足 100 个 batch 的区间，也按实际 batch 数求平均。
            batches_in_window = i % log_interval + 1
            avg_loss = running_loss / batches_in_window
            avg_vloss = running_vloss / len(validation_loader)

            # 记录每个 batch 的平均运行损失。
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch * len(training_loader) + i)

            running_loss = 0.0
print('Finished Training')

writer.flush()


"""
切换到已经打开的 TensorBoard，查看 SCALARS 标签页。


可视化模型
==========

TensorBoard 也可以用来查看模型内部的数据流。为此，调用 add_graph() 方法，
并传入模型和一个示例输入：
"""

# 再次取出一个图像小批次。
dataiter = iter(training_loader)
images, labels = next(dataiter)

# add_graph() 会跟踪示例输入在模型中的流动，并将其呈现为图。
# DataLoader 返回 CPU Tensor；模型位于 GPU 时，示例输入也要迁移到同一设备。
writer.add_graph(net, images.to(my_device))
writer.flush()


"""
切换到 TensorBoard 后，应该能看到 GRAPHS 标签页。双击“NET”节点，可以查看
模型中的层和数据流。


使用 Embedding 可视化数据集
===========================

这里使用的 28×28 图像块可以表示为 784 维向量（28×28=784）。把它们投影到
低维表示可能很有启发性。add_embedding() 方法会将一组数据投影到方差最大的
三个维度，并以交互式三维图表显示。add_embedding() 方法通过投影到方差最大的
三个维度，自动完成这项工作。

下面，我们会从数据中取出一个样本集合，并生成这样的 Embedding：
"""


# 随机选取一个数据子集及其对应标签。
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


# 提取随机数据子集。
images, labels = select_n_random(training_set.data, training_set.targets)

# 取得每张图像的类别标签。
class_labels = [classes[label] for label in labels]

# 记录 Embedding。
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                     metadata=class_labels,
                     label_img=images.unsqueeze(1))
writer.flush()
writer.close()


"""
现在切换到 TensorBoard 并选择 PROJECTOR 标签页，应该能看到该投影的三维表示。
你可以旋转和缩放模型。在较大和较小的尺度下观察它，看看是否能在投影数据和标签
聚类中发现规律。

为了看得更清楚，建议：

1. 在左侧的“Color by”下拉菜单中选择“label”。
2. 切换顶部的夜间模式图标，让浅色图像显示在深色背景上。


其他资源
========

要了解更多信息，请参阅：

* PyTorch 关于 torch.utils.tensorboard.SummaryWriter 的文档。
* PyTorch.org Tutorials 中的 TensorBoard 教程内容。
* TensorBoard 文档中的更多相关信息。
"""
