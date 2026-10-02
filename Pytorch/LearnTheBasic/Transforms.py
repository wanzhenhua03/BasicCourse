"""TorchVision 数据转换基础教程。

参考：https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

运行：
    conda activate MLLearn
    python .\Pytorch\Transforms.py

本文件按官方教程的顺序，介绍 transform、target_transform、
ToImage、ToDtype 和 Lambda 转换。说明为原创中文转述。
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2


# 机器学习数据常常不是模型可以直接使用的形式。
# transform 用于处理特征，target_transform 用于处理标签。

project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"

# FashionMNIST 原始图像是 PIL Image。
# v2.ToImage() 先将它转换成 torchvision Image Tensor。
# v2.ToDtype(..., scale=True) 再转换为 float32，并将像素缩放到 [0, 1]。
feature_transform = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
)

# Lambda 转换允许将自定义的可调用逻辑包装成 transform。
# F.one_hot 把 0～9 的整数标签转换为长度为 10 的独热向量。
label_transform = v2.Lambda(
    lambda label: F.one_hot(
        torch.tensor(label),
        num_classes=10,
    ).float()
)

# 数据集在 __getitem__ 返回样本前，会分别应用上述两种转换。
dataset = datasets.FashionMNIST(
    root=data_root,
    train=True,
    download=True,
    transform=feature_transform,
    target_transform=label_transform,
)

# 取出一个样本，直观检查转换结果。
image, label = dataset[0]
print(f"图像类型：{type(image)}")
print(f"图像形状：{image.shape}")
print(f"图像数据类型：{image.dtype}")
print(f"图像像素范围：[{image.min().item():.1f}, {image.max().item():.1f}]")
print(f"独热标签：{label}")
print(f"独热标签形状：{label.shape}")
print(f"原始类别索引：{label.argmax().item()}")
