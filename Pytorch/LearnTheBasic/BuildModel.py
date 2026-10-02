"""PyTorch 神经网络构建基础教程。

参考：https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

运行：
    conda activate MLLearn
    python .\Pytorch\BuildModel.py

内容按官方教程顺序组织：选择设备、定义 nn.Module、运行模型、
逐层解释 Flatten/Linear/ReLU/Sequential/Softmax，最后查看模型参数。
"""

import torch
from torch import nn


# 优先使用可用的加速器，否则使用 CPU。
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"当前计算设备：{device}")
if device == "cuda":
    print(f"GPU：{torch.cuda.get_device_name(torch.cuda.current_device())}")


# PyTorch 中的神经网络由层（module）构成。
# 自定义网络需要继承 nn.Module，并实现 __init__ 和 forward。
class NeuralNetwork(nn.Module):
    """用于 FashionMNIST 10 类图像分类的全连接网络。"""

    def __init__(self):
        super().__init__()

        # Flatten 保留批次维，将每张 28×28 图像展平为 784 个像素。
        self.flatten = nn.Flatten()

        # Sequential 使数据按声明顺序通过各层。
        # 最后一层输出 10 个 logits，分别对应 10 个类别。
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, input_tensor):
        # forward 定义数据从输入到输出的计算流程。
        flattened = self.flatten(input_tensor)
        logits = self.linear_relu_stack(flattened)
        return logits


# 创建模型后将其所有参数移动到目标设备。
model = NeuralNetwork().to(device)
print("\n模型结构：")
print(model)

# 直接使用 model(input) 会自动调用 forward 并执行 PyTorch 的必要钩子。
# 不应手工调用 model.forward(input)。
input_batch = torch.rand(1, 28, 28, device=device)
logits = model(input_batch)

# logits 是未归一化分数。Softmax 沿类别维把它转换为和为 1 的概率。
predicted_probabilities = nn.Softmax(dim=1)(logits)
predicted_class = predicted_probabilities.argmax(dim=1)
print(f"\n模型输出形状：{logits.shape}")
print(f"概率之和：{predicted_probabilities.sum(dim=1)}")
print(f"预测类别：{predicted_class}")


# 下面使用 3 张 28×28 的随机图像，分步观察主要网络层。
input_images = torch.rand(3, 28, 28)
print(f"\n原始批次形状：{input_images.size()}")

# nn.Flatten 把每张图像展平，但不改变批次数 3。
flatten = nn.Flatten()
flat_images = flatten(input_images)
print(f"Flatten 后的形状：{flat_images.size()}")

# nn.Linear 使用内部保存的权重和偏置执行线性变换。
first_linear_layer = nn.Linear(in_features=28 * 28, out_features=20)
hidden_values = first_linear_layer(flat_images)
print(f"Linear 后的形状：{hidden_values.size()}")

# ReLU 将负数设为 0，通过引入非线性帮助网络学习复杂关系。
print(f"\nReLU 前两个样本的前 5 个值：\n{hidden_values[:2, :5]}")
hidden_values = nn.ReLU()(hidden_values)
print(f"ReLU 后两个样本的前 5 个值：\n{hidden_values[:2, :5]}")

# Sequential 是有序容器，会依次调用其中的层。
sequential_modules = nn.Sequential(
    flatten,
    first_linear_layer,
    nn.ReLU(),
    nn.Linear(20, 10),
)
sequential_logits = sequential_modules(input_images)
print(f"\nSequential 输出形状：{sequential_logits.shape}")

# Softmax 中 dim=1 表示对每个样本的 10 个类别分数归一化。
softmax = nn.Softmax(dim=1)
sequential_probabilities = softmax(sequential_logits)
print(f"Softmax 后每个样本的概率和：{sequential_probabilities.sum(dim=1)}")


# nn.Module 会自动跟踪在对象中声明的子模块与参数。
# named_parameters() 同时返回参数名称和参数 Tensor。
print("\n模型参数摘要：")
for parameter_name, parameter in model.named_parameters():
    preview = parameter.detach().flatten()[:5].cpu()
    print(
        f"参数：{parameter_name} | "
        f"形状：{tuple(parameter.shape)} | "
        f"前 5 个值：{preview}"
    )
