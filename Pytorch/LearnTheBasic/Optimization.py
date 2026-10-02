"""PyTorch 模型参数优化基础教程。

参考：https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

运行：
    conda activate MLLearn
    python .\Pytorch\Optimization.py

内容依次包含前置数据与模型、超参数、损失函数、优化器、
训练循环、测试循环和多轮训练。只封装教程本身需要的网络与循环。
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# 前置代码：复用前面 Dataset/DataLoader 与构建模型两节的核心内容。
project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
image_transform = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
)

training_data = datasets.FashionMNIST(
    root=data_root,
    train=True,
    download=True,
    transform=image_transform,
)
test_data = datasets.FashionMNIST(
    root=data_root,
    train=False,
    download=True,
    transform=image_transform,
)


# 超参数由开发者指定，而不是由模型自动学习。
# learning_rate 控制每次更新的步长。
# batch_size 是一次参数更新前处理的样本数。
# epochs 是完整遍历训练集的次数。
learning_rate = 1e-3
batch_size = 64
epochs = 10

train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True,
)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    """FashionMNIST 全连接分类模型。"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, input_tensor):
        flattened = self.flatten(input_tensor)
        logits = self.linear_relu_stack(flattened)
        return logits


# 当前 MLLearn 环境可用 CUDA，所以此处会选择 GTX 1050 Ti。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
print(f"当前计算设备：{device}")
if device.type == "cuda":
    print(f"GPU：{torch.cuda.get_device_name(device)}")


# CrossEntropyLoss 适用于多分类任务，直接接收原始 logits 和整数类别标签。
# 它内部组合了 LogSoftmax 与 NLLLoss，所以训练时不要先手工调用 Softmax。
loss_fn = nn.CrossEntropyLoss()

# SGD 使用损失梯度更新 model.parameters() 中的可训练参数。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    """完整遍历一次训练集并更新模型参数。"""
    dataset_size = len(dataloader.dataset)

    # train() 将模型切换到训练模式，对 Dropout 和 BatchNorm 尤其重要。
    model.train()

    for batch_index, (features, labels) in enumerate(dataloader):
        # 模型、特征与标签必须位于同一设备。
        features = features.to(device)
        labels = labels.to(device)

        # 前向传播并计算预测误差。
        predictions = model(features)
        loss = loss_fn(predictions, labels)

        # backward 计算梯度，step 更新参数，zero_grad 清除已累加的梯度。
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_index % 100 == 0:
            loss_value = loss.item()
            processed = batch_index * batch_size + len(features)
            print(
                f"损失：{loss_value:>7f}  "
                f"[已处理 {processed:>5d}/{dataset_size:>5d}]"
            )


def test_loop(dataloader, model, loss_fn):
    """在测试集上计算平均损失和准确率，不更新参数。"""
    # eval() 将模型切换到评估模式。
    model.eval()

    dataset_size = len(dataloader.dataset)
    number_of_batches = len(dataloader)
    test_loss = 0.0
    correct_count = 0.0

    # no_grad 避免创建无用的反向计算图，降低显存和计算开销。
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            test_loss += loss_fn(predictions, labels).item()
            correct_count += (
                (predictions.argmax(dim=1) == labels)
                .type(torch.float)
                .sum()
                .item()
            )

    average_loss = test_loss / number_of_batches
    accuracy = correct_count / dataset_size
    print(
        "测试结果：\n"
        f" 准确率：{100 * accuracy:>0.1f}%，"
        f"平均损失：{average_loss:>8f}\n"
    )


# 每个 epoch 先训练，再在测试集上观察性能。
for epoch_index in range(epochs):
    print(f"训练轮次 {epoch_index + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("训练完成。")
