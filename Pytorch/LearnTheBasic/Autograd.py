"""torch.autograd 自动微分基础教程。

参考：https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

运行：
    conda activate MLLearn
    python .\Pytorch\Autograd.py

内容顺序：构建计算图、查看 grad_fn、计算梯度、关闭梯度跟踪、
理解动态计算图以及观察向量-雅可比乘积和梯度累加。
"""

import torch


# 反向传播通过损失对模型参数的梯度，指导参数调整。
# torch.autograd 会在前向计算时记录操作，并在 backward 时自动应用链式法则。

# x 是长度为 5 的输入，y 是长度为 3 的目标。
input_tensor = torch.ones(5)
target = torch.zeros(3)

# weight 和 bias 是需要优化的参数，所以设置 requires_grad=True。
weight = torch.randn(5, 3, requires_grad=True)
bias = torch.randn(3, requires_grad=True)

# 一层线性网络的前向计算：z = xW + b。
logits = torch.matmul(input_tensor, weight) + bias

# 二元交叉熵函数同时完成 sigmoid 与损失计算。
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
print(f"当前损失：{loss.item():.6f}")


# 参与计算图构建的运算由 Function 对象表示。
# 输出 Tensor 的 grad_fn 引用了反向传播时应执行的梯度函数。
print(f"logits 的梯度函数：{logits.grad_fn}")
print(f"loss 的梯度函数：{loss.grad_fn}")


# loss 是标量，直接调用 backward() 即可开始反向传播。
loss.backward()

# 叶子 Tensor weight 和 bias 的梯度分别保存在 .grad 中。
print(f"\nloss 对 weight 的梯度：\n{weight.grad}")
print(f"loss 对 bias 的梯度：\n{bias.grad}")

# 默认情况下，同一张计算图在 backward 后会被释放。
# 若必须在同一张图上多次反向传播，需要使用 retain_graph=True。


# 推理时只需要前向计算，通常不需要记录梯度。
# 正常计算仍会跟踪梯度。
tracked_logits = torch.matmul(input_tensor, weight) + bias
print(f"\n正常计算是否跟踪梯度：{tracked_logits.requires_grad}")

# torch.no_grad() 作用域内的运算不会加入计算图。
with torch.no_grad():
    untracked_logits = torch.matmul(input_tensor, weight) + bias
print(f"no_grad 中是否跟踪梯度：{untracked_logits.requires_grad}")

# detach() 会返回与原 Tensor 分离的视图，新 Tensor 不继续跟踪梯度。
detached_logits = tracked_logits.detach()
print(f"detach 后是否跟踪梯度：{detached_logits.requires_grad}")

# 关闭梯度跟踪可用于冻结部分模型参数，也可以降低纯前向计算的资源消耗。


# Autograd 在前向传播中计算结果并记录运算，
# 在反向传播中从输出向叶子节点逐步传播梯度。
# PyTorch 的计算图是动态的：每次前向计算都会重新构建，因此可以自然使用 Python 控制流。


# 当输出不是标量时，backward 需要接收一个与输出同形状的向量。
# PyTorch 计算的是向量-雅可比乘积，而不是显式构建完整雅可比矩阵。
input_matrix = torch.eye(4, 5, requires_grad=True)
output_matrix = (input_matrix + 1).pow(2).t()
gradient_vector = torch.ones_like(output_matrix)

output_matrix.backward(gradient_vector, retain_graph=True)
print(f"\n第一次 backward 后的梯度：\n{input_matrix.grad}")

# PyTorch 默认会累加梯度，所以第二次结果是第一次的两倍。
output_matrix.backward(gradient_vector, retain_graph=True)
print(f"\n第二次 backward 后累加的梯度：\n{input_matrix.grad}")

# 清零后再计算，得到与第一次相同的梯度。
input_matrix.grad.zero_()
output_matrix.backward(gradient_vector, retain_graph=True)
print(f"\n梯度清零后再次 backward：\n{input_matrix.grad}")
