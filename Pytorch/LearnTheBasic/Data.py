"""Dataset 与 DataLoader 基础教程。

参考：https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

本文件按官方教程的知识顺序编写为原创中文讲解：
1. 加载 FashionMNIST。
2. 通过索引访问并可视化数据。
3. 了解自定义 Dataset 需要实现的方法。
4. 用 DataLoader 生成批次数据。

运行：
    conda activate MLLearn
    python .\Pytorch\Data.py

脚本会显示两个图像窗口，关闭当前窗口后才会继续执行。
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import v2


# Dataset 负责保存样本及标签，DataLoader 负责按批次遍历 Dataset。
# 将数据处理与模型代码分开，可以让训练代码更清晰。

# 使用基于脚本位置的绝对路径，保证从不同工作目录运行时都能找到数据。
project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"

# FashionMNIST 中每个样本是一张 28×28 的灰度图像和一个 0～9 的类别标签。
# ToImage 将 PIL 图像转为 Tensor 形式；ToDtype 转为 float32 并把像素缩放到 [0, 1]。
image_transform = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
)

# root 指定数据存储目录；train=True 选择训练集。
# download=True 表示本地没有数据时才下载。
training_data = datasets.FashionMNIST(
    root=data_root,
    train=True,
    download=True,
    transform=image_transform,
)

# train=False 选择测试集。
test_data = datasets.FashionMNIST(
    root=data_root,
    train=False,
    download=True,
    transform=image_transform,
)

print(f"训练集样本数：{len(training_data)}")
print(f"测试集样本数：{len(test_data)}")


# Dataset 可以像 Python 序列一样通过 training_data[index] 访问。
# 字典将数字标签映射为易读的类别名称。
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 随机选择 9 个样本，观察图像与标签的对应关系。
figure = plt.figure(figsize=(8, 8))
columns, rows = 4, 3
for position in range(1, columns * rows + 1):
    sample_index = torch.randint(len(training_data), size=(1,)).item()
    image, label = training_data[sample_index]
    figure.add_subplot(rows, columns, position)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap="gray")
plt.show()


# 当数据不在 TorchVision 预置数据集中时，可以通过继承 Dataset 自定义读取逻辑。
# 自定义 Dataset 至少需要实现 __init__、__len__ 和 __getitem__。
class CustomImageDataset(Dataset):
    """演示“CSV 保存文件名与标签，目录保存图像”的自定义数据集。"""

    def __init__(
        self,
        annotations_file,
        image_directory,
        transform=None,
        target_transform=None,
    ):
        # __init__ 在创建 Dataset 对象时执行一次。
        # img_labels 保存 CSV 中的图像文件名和标签。
        self.img_labels = pd.read_csv(annotations_file)
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # __len__ 返回数据集中的样本总数。
        return len(self.img_labels)

    def __getitem__(self, index):
        # __getitem__ 根据索引加载并返回一个“图像、标签”对。
        image_path = os.path.join(
            self.image_directory,
            self.img_labels.iloc[index, 0],
        )
        image = decode_image(image_path)
        label = self.img_labels.iloc[index, 1]

        # 仅在调用者传入了转换逻辑时执行转换。
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# DataLoader 把逐个样本组成小批次。
# batch_size=64 表示每次返回 64 个样本；shuffle=True 会在每轮重新打乱顺序。
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# iter 创建迭代器，next 取出第一个批次。
train_features, train_labels = next(iter(train_dataloader))
print(f"\n一个图像批次的形状：{train_features.size()}")
print(f"一个标签批次的形状：{train_labels.size()}")

# 显示批次中的第一张图像及其标签。
batch_image = train_features[0].squeeze()
batch_label = train_labels[0]
plt.imshow(batch_image, cmap="gray")
plt.axis("off")
plt.show()
print(f"该图像的数字标签：{batch_label.item()}")
print(f"该图像的类别名称：{labels_map[batch_label.item()]}")
