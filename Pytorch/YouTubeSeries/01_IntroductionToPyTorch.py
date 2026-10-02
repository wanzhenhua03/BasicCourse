"""第 1 课：PyTorch 简介。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
配套视频：https://www.youtube.com/watch?v=IC0_FRiX-sw

本文件依照官方教程的叙事和代码顺序整理，以原创中文说明串联示例。
它依次介绍 Tensor、PyTorch 模型、Dataset/DataLoader，以及完整训练流程。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\01_IntroductionToPyTorch.py

首次运行会把 CIFAR-10 下载到本文件所在目录的 data 文件夹。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 所有下载内容都相对于脚本目录保存，避免运行工作目录影响文件位置。
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


"""
一、PyTorch Tensor

Tensor 是 PyTorch 表示数据的核心结构。它与 NumPy ndarray 很相似，
但还可以在加速器上计算，并参与自动微分。下面先观察 Tensor 的创建、
数据类型、随机数、逐元素运算和常见数学函数。
"""

# 默认创建一个 5×3 的全零浮点 Tensor。
z = torch.zeros(5, 3)
print("5×3 的全零 Tensor：")
print(z)
print("数据类型：", z.dtype)

# dtype 参数可以显式指定元素类型。
i = torch.ones((5, 3), dtype=torch.int16)
print("\n指定 int16 类型的全一 Tensor：")
print(i)

# 固定随机种子后，伪随机数序列可以复现。
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print("\n第一个随机 Tensor：")
print(r1)

r2 = torch.rand(2, 2)
print("\n随机序列中的下一个 Tensor：")
print(r2)

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print("\n重新设置相同种子后，应与第一个 Tensor 相同：")
print(r3)

"""
形状相同的 Tensor 可以直接进行逐元素运算。不同形状通常不能直接运算，
除非它们满足 broadcasting（广播）规则。
"""

ones = torch.ones(2, 3)
print("\n全一 Tensor：")
print(ones)

twos = torch.ones(2, 3) * 2
print("\n全二 Tensor：")
print(twos)

threes = ones + twos
print("\n逐元素相加：")
print(threes)
print("结果形状：", threes.shape)

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
print("\n两个形状不同、不能直接逐元素相加的 Tensor：", r1.shape, r2.shape)
# 取消下一行注释会触发形状不兼容错误。
# r1 + r2

# 生成取值范围约为 [-1, 1) 的随机矩阵。
r = (torch.rand(2, 2) - 0.5) * 2
print("\n随机矩阵 r：")
print(r)
print("\nr 的绝对值：")
print(torch.abs(r))
print("\nr 的反正弦：")
print(torch.asin(r))
print("\nr 的行列式：")
print(torch.det(r))
print("\nr 的奇异值分解：")
print(torch.svd(r))
print("\nr 的标准差和平均值：")
print(torch.std_mean(r))
print("\nr 的最大值：")
print(torch.max(r))


"""
二、PyTorch 模型

PyTorch 模型通常继承 nn.Module，在 __init__() 中声明层，在 forward()
中说明数据如何经过这些层。这里使用经典的 LeNet-5 结构展示卷积层、
池化层和全连接层如何组合。调用模型对象时，PyTorch 会自动执行 forward()；
实际使用时不应直接调用 forward()。
"""


class LeNet(nn.Module):
    """用于演示典型卷积神经网络结构的 LeNet。"""

    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层接收单通道图像，输出 6 个特征图；第二层输出 16 个特征图。
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 卷积结果展平后，依次经过三个全连接层。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积后使用 ReLU，再用最大池化缩小空间尺寸。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # 将卷积输出展平，送入全连接层。
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """计算除 batch 维以外的元素个数。"""
        size = x.size()[1:]
        num_features = 1
        for dimension_size in size:
            num_features *= dimension_size
        return num_features


net = LeNet()
print("\nLeNet 模型结构：")
print(net)

# 模拟一个 batch，其中只有一张 32×32 的单通道图像。
input_image = torch.rand(5, 1, 32, 32)
print("\n输入图像 batch 的形状：")
print(input_image.shape)

output = net(input_image)
print("\n模型原始输出：")
print(output)
print("输出形状：", output.shape)


"""
三、Dataset 与 DataLoader

Dataset 负责保存和读取单个样本；DataLoader 负责打乱、分批并迭代数据。
这里使用 TorchVision 提供的 CIFAR-10。ToTensor 把图像转换为 Tensor，
Normalize 按通道对像素做标准化。
"""

display_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=display_transform,
)

# Windows 直接执行教学脚本时采用单进程加载，避免多进程重复执行顶层代码。
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def imshow(img):
    """将一个图像 Tensor 转换为 Matplotlib 可显示的排列。"""
    img = img / 2 + 0.5
    numpy_image = img.numpy()
    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))


# 取出一个 batch，显示其中四张图像及其标签。
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
plt.title("CIFAR-10 训练样本")
plt.axis("off")
plt.show()
print("样本标签：", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))


"""
四、训练 PyTorch 模型

下面重新建立适用于三通道 CIFAR-10 图像的数据变换、训练集和测试集，
然后训练一个 LeNet 变体。完整训练步骤是：清空梯度、前向计算、计算损失、
反向传播、更新参数。最后在未参加训练的测试集上计算分类准确率。
"""

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform,
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

testset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transform,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)


def show_normalized_image(img):
    """撤销本节的 [-1, 1] 标准化并显示图像。"""
    img = img / 2 + 0.5
    numpy_image = img.numpy()
    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = next(dataiter)
show_normalized_image(torchvision.utils.make_grid(images))
plt.title("用于训练的 CIFAR-10 样本")
plt.axis("off")
plt.show()
print("训练样本标签：", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))


class Net(nn.Module):
    """用于 CIFAR-10 十分类的卷积神经网络。"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# CrossEntropyLoss 适合多类别分类；SGD 根据梯度更新模型参数。
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0

    for batch_index, data in enumerate(trainloader, 0):
        inputs, labels = data

        # PyTorch 默认累加梯度，因此每个 batch 开始前必须清空旧梯度。
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 2000 == 1999:
            print(
                "[%d, %5d] 平均损失：%.3f"
                % (epoch + 1, batch_index + 1, running_loss / 2000)
            )
            running_loss = 0.0

print("训练完成")

# 测试阶段只做前向推理，不需要构建梯度计算图。
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("模型在 10000 张测试图像上的准确率：%d %%" % (100 * correct / total))

