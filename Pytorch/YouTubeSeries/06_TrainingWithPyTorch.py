"""第 6 课：使用 PyTorch 训练模型。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
配套视频：https://www.youtube.com/watch?v=jF43_wj_DCQ

以下文字讲解依据用户提供的教程原文翻译、排版，并保持原文顺序。
本地运行适配：数据、TensorBoard 日志和模型文件保存到本文件所在目录。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\06_TrainingWithPyTorch.py

本课按原教程示例训练 5 个 epoch。
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RUNS_DIR = SCRIPT_DIR / "runs"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


"""
简介
====

在此前的视频中，我们讨论并演示了：

* 使用 torch.nn 模块中的神经网络层和函数构建模型；
* 自动计算梯度的工作机制，这是基于梯度训练模型的核心；
* 使用 TensorBoard 可视化训练进展和其他活动。

在本视频中，我们会为你的工具箱增添一些新工具：

* 熟悉 Dataset 和 DataLoader 抽象，以及它们如何让训练循环更容易向模型提供数据；
* 讨论具体的损失函数，以及何时使用它们；
* 了解 PyTorch 优化器：它们根据损失函数的结果，使用相应算法调整模型权重。

最后，我们会把这些内容组合起来，看看一个完整的 PyTorch 训练循环如何运行。


Dataset 和 DataLoader
======================

Dataset 和 DataLoader 类封装了从存储介质中取出数据，并按批次将数据交给训练循环
的过程。

Dataset 负责访问并处理单个数据实例。

DataLoader 从 Dataset 中取出数据实例（可以自动取出，也可以使用你定义的
sampler），将它们组成批次，再返回给训练循环使用。无论数据集包含哪种类型的
数据，DataLoader 都能与之配合使用。

在本教程中，我们将使用 TorchVision 提供的 Fashion-MNIST 数据集。我们用
torchvision.transforms.v2.Normalize() 对图像块内容的分布进行零中心化和
归一化，并下载训练和验证两个数据划分。
"""

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,)),
    ]
)

training_set = torchvision.datasets.FashionMNIST(
    DATA_DIR,
    train=True,
    transform=transform,
    download=True,
)
validation_set = torchvision.datasets.FashionMNIST(
    DATA_DIR,
    train=False,
    transform=transform,
    download=True,
)

training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)
validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)

print(f"训练集包含 {len(training_set)} 个样本")
print(f"验证集包含 {len(validation_set)} 个样本")


"""
像往常一样，先把数据可视化，做一次基本检查：
"""


# 用于在 notebook 中直接显示图像的辅助函数。
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    numpy_image = img.numpy()
    if one_channel:
        plt.imshow(numpy_image, cmap="Greys")
    else:
        plt.imshow(np.transpose(numpy_image, (1, 2, 0)))


dataiter = iter(training_loader)
images, labels = next(dataiter)
# 从图像创建网格并显示。
image_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(image_grid, one_channel=True)
plt.show()  # 本地运行适配：在独立 Python 脚本中显示 notebook 单元的图像。
print("  ".join(classes[labels[j]] for j in range(4)))


"""
模型
====

本例使用的模型是 LeNet-5 的一个变体。如果你看过本系列之前的视频，应该会对它
感到熟悉。
"""


# PyTorch 模型继承自 torch.nn.Module。
class GarmentClassifier(nn.Module):

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


model = GarmentClassifier()


"""
损失函数
========

本例使用交叉熵损失。为了演示，我们会创建一批虚构的输出值和标签值，把它们传给
损失函数，并查看结果。
"""

loss_fn = torch.nn.CrossEntropyLoss()

# 注意：损失函数期望接收批次数据，所以这里创建包含 4 个样本的批次。
# 每个输入对应的 10 个值，表示模型对 10 个类别的置信程度。
dummy_outputs = torch.rand(4, 10)
# 表示所测试的 10 个类别中，哪一个是正确类别。
dummy_labels = torch.tensor([1, 5, 3, 7])
print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print(f"这个 batch 的总损失：{loss.item()}")


"""
优化器
======

本例使用带动量的简单随机梯度下降法。

尝试改变这种优化方案中的一些设置，会很有启发性：

学习率决定了优化器每一步的步长。改变学习率后，训练结果在准确率和收敛时间上会
发生什么变化？

动量会在连续多步中推动优化器朝最强梯度的方向移动。改变它的数值会如何影响结果？

尝试其他优化算法，例如 averaged SGD、Adagrad 或 Adam。它们得到的结果有何不同？
"""

# 优化器定义在 torch.optim 包中。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


"""
训练循环
========

下面的函数负责执行一个训练 epoch。它会从 DataLoader 中依次取出数据，每次循环
执行以下操作：

* 从 DataLoader 获取一批训练数据；
* 将优化器中的梯度清零；
* 执行一次推理，也就是根据一批输入获得模型预测；
* 计算这批预测与数据集标签之间的损失；
* 计算学习权重的反向梯度；
* 让优化器执行一次学习步骤，也就是根据这批数据观察到的梯度，按照选定的优化
  算法调整模型的学习权重。

它每经过 1000 个 batch 就报告一次损失。

最后，它会报告最近 1000 个 batch 的平均每批次损失，以便与验证运行比较。
"""


def train_one_epoch(epoch_index, tensorboard_writer):
    running_loss = 0.0
    last_loss = 0.0

    # 这里用 enumerate(training_loader) 而不是 iter(training_loader)，
    # 这样可以跟踪 batch 索引，并在一个 epoch 内报告训练情况。
    for batch_index, data in enumerate(training_loader):
        # 每个数据实例由输入和标签组成。
        inputs, labels = data

        # 每个 batch 都要清零梯度！
        optimizer.zero_grad()
        # 对这个 batch 进行预测。
        outputs = model(inputs)
        # 计算损失及其梯度。
        loss = loss_fn(outputs, labels)
        loss.backward()
        # 调整学习权重。
        optimizer.step()

        # 收集数据并报告。
        running_loss += loss.item()
        if batch_index % 1000 == 999:
            last_loss = running_loss / 1000  # 每个 batch 的平均损失。
            print(f"  batch {batch_index + 1} 平均损失：{last_loss}")
            tensorboard_step = epoch_index * len(training_loader) + batch_index + 1
            tensorboard_writer.add_scalar(
                "Loss/train",
                last_loss,
                tensorboard_step,
            )
            running_loss = 0.0

    return last_loss


"""
每个 epoch 要做的事
===================

每个 epoch 结束后，我们还需要做几件事：

* 在未用于训练的一组数据上检查相对损失，以此进行验证，并报告结果；
* 保存一份模型副本。

这里，我们将在 TensorBoard 中报告结果。这需要在命令行中启动 TensorBoard，
并在另一个浏览器标签页中打开它。
"""

# 原教程将初始化写在单独的 notebook 单元中，便于给同一次运行追加更多 epoch。
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = RUNS_DIR / f"fashion_trainer_{timestamp}"
writer = SummaryWriter(log_dir=str(log_dir))
epoch_number = 0

EPOCHS = 5
best_validation_loss = 1_000_000.0

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch_number + 1}：")

    # 确保开启梯度跟踪，并遍历一遍数据。
    model.train(True)
    average_loss = train_one_epoch(epoch_number, writer)

    running_validation_loss = 0.0
    # 切换至评估模式：关闭 Dropout，并让 BatchNorm 使用总体统计量。
    model.eval()

    # 关闭梯度计算，以减少内存消耗。
    with torch.no_grad():
        for validation_data in validation_loader:
            validation_inputs, validation_labels = validation_data
            validation_outputs = model(validation_inputs)
            validation_loss = loss_fn(validation_outputs, validation_labels)
            running_validation_loss += validation_loss.item()

    average_validation_loss = running_validation_loss / len(validation_loader)
    print(
        f"LOSS train {average_loss} "
        f"valid {average_validation_loss}"
    )

    # 记录训练集和验证集的平均每批次损失。
    writer.add_scalars(
        "训练与验证损失",
        {
            "训练": average_loss,
            "验证": average_validation_loss,
        },
        epoch_number + 1,
    )
    writer.flush()

    # 跟踪最佳表现，并保存模型的状态。
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        model_path = CHECKPOINT_DIR / f"model_{timestamp}_{epoch_number}.pth"
        torch.save(model.state_dict(), model_path)
        print("保存新的最佳模型：", model_path)

    epoch_number += 1

writer.close()


"""
加载保存的模型版本：

    saved_model = GarmentClassifier()
    saved_model.load_state_dict(torch.load(PATH))

加载模型后，就可以根据需要继续训练、进行推理或分析。

请注意：如果模型的构造函数中有影响模型结构的参数，就需要提供这些参数，并将
模型配置得与保存时的状态完全相同。


其他资源
========

* pytorch.org 上关于数据工具（包括 Dataset 和 DataLoader）的文档；
* 关于在 GPU 训练中使用固定内存（pinned memory）的说明；
* TorchVision、TorchText 和 TorchAudio 中可用数据集的文档；
* PyTorch 中可用损失函数的文档；
* torch.optim 包的文档，其中包括优化器和学习率调度等相关工具；
* 关于保存和加载模型的详细教程；
* pytorch.org 的 Tutorials 版块，其中包含各种训练任务的教程，例如不同领域的
  分类、生成对抗网络、强化学习等。
"""

