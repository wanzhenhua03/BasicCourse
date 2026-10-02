"""第 3 课：Autograd 自动微分基础。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
配套视频：https://www.youtube.com/watch?v=M0fX15_-xrY

本文件按照官方课程顺序介绍动态计算图、反向传播、梯度累加、梯度跟踪控制、
原地操作的限制、Autograd Profiler，以及 Jacobian 与向量-Jacobian 乘积。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\03_Autograd.py
"""

import math

import matplotlib.pyplot as plt
import torch


"""
一、为什么需要 Autograd

训练神经网络的核心任务之一，是计算损失对每个可学习参数的偏导数。
复杂网络包含大量相互连接的运算，手工展开链式法则既困难又容易出错。
Autograd 会在程序运行时动态记录 Tensor 运算，形成计算图，并利用每个运算
已知的局部导数高效完成反向传播。
"""


"""
二、一个简单示例

requires_grad=True 告诉 PyTorch：从这个 Tensor 开始跟踪后续运算，以便稍后
计算梯度。这里从等间距的输入 a 出发，依次计算正弦、缩放、平移和求和。
"""

a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)
print("输入 a：")
print(a)

b = torch.sin(a)
plt.figure("sin(a)")
plt.plot(a.detach(), b.detach())
plt.title("b = sin(a)")
plt.xlabel("a")
plt.ylabel("b")
plt.show()
print("\nb = sin(a)：")
print(b)

c = 2 * b
print("\nc = 2b：")
print(c)

d = c + 1
print("\nd = c + 1：")
print(d)

# 将向量归约为标量，标量可以直接调用 backward()。
out = d.sum()
print("\nout = sum(d)：")
print(out)


"""
三、观察计算图

由被跟踪运算得到的 Tensor 带有 grad_fn。next_functions 可以继续向计算图
的上游查看。a 是用户直接创建的叶子 Tensor，没有创建它的运算，因此
a.grad_fn 为 None；反向传播后的梯度会存放在 a.grad 中。
"""

print("\nd 的梯度函数：")
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(
    d.grad_fn.next_functions[0][0]
    .next_functions[0][0]
    .next_functions[0][0]
    .next_functions
)
print("\nc 的梯度函数：", c.grad_fn)
print("b 的梯度函数：", b.grad_fn)
print("a 的梯度函数：", a.grad_fn)

# out 是标量，相当于隐式提供上游梯度 1。
out.backward()
print("\na 的梯度：")
print(a.grad)

plt.figure("gradient")
plt.plot(a.detach(), a.grad.detach())
plt.title("d(sum(2sin(a)+1)) / da")
plt.xlabel("a")
plt.ylabel("gradient")
plt.show()


"""
四、训练中的 Autograd

下面建立一个两层小网络。模型参数默认 requires_grad=True，输入和目标数据
不需要梯度。损失反向传播后，各参数的 grad 保存损失对该参数的梯度；
优化器再据此更新参数。
"""

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10


class TinyModel(torch.nn.Module):
    """用于观察参数梯度的两层全连接网络。"""

    def __init__(self):
        super(TinyModel, self).__init__()
        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)
model = TinyModel()

print("\n更新前 layer2 的部分权重：")
print(model.layer2.weight[0][0:10])
print("反向传播前的梯度：", model.layer2.weight.grad)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
prediction = model(some_input)
loss = (ideal_output - prediction).pow(2).sum()
print("当前损失：", loss)

loss.backward()
print("\n反向传播后，权重尚未改变：")
print(model.layer2.weight[0][0:10])
print("对应梯度：")
print(model.layer2.weight.grad[0][0:10])

optimizer.step()
print("\noptimizer.step() 后的权重：")
print(model.layer2.weight[0][0:10])
print("grad 中仍保留刚才的梯度：")
print(model.layer2.weight.grad[0][0:10])


"""
五、梯度会累加

backward() 不会自动覆盖旧梯度，而是把新梯度加到叶子节点的 grad 上。
这允许某些算法累积多个 batch，但普通训练循环必须在下一轮前清空梯度。
"""

print("\n再次反向传播前的梯度：")
print(model.layer2.weight.grad[0][0:10])

for _ in range(5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print("连续反向传播五次后的累积梯度：")
print(model.layer2.weight.grad[0][0:10])

optimizer.zero_grad(set_to_none=False)
print("清零后的梯度：")
print(model.layer2.weight.grad[0][0:10])


"""
六、关闭与恢复梯度跟踪

可以直接修改叶子 Tensor 的 requires_grad，也可用 torch.no_grad() 临时关闭
某段代码的跟踪。装饰器 @torch.no_grad() 适用于整个函数。detach() 则返回
一个与原 Tensor 共享数据、但脱离当前计算图的新 Tensor。
"""

a = torch.ones(2, 3, requires_grad=True)
print("\nrequires_grad=True 的 a：")
print(a)

b1 = 2 * a
print("由 a 计算出的 b1：")
print(b1)

a.requires_grad = False
b2 = 2 * a
print("关闭跟踪后计算的 b2：")
print(b2)

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print("\n正常跟踪的 c1：")
print(c1)

with torch.no_grad():
    c2 = a + b
print("no_grad 块中的 c2：")
print(c2)

c3 = a * b
print("离开 no_grad 后重新跟踪的 c3：")
print(c3)


def add_tensors1(x, y):
    """普通函数，其运算会在需要时跟踪梯度。"""
    return x + y


@torch.no_grad()
def add_tensors2(x, y):
    """整个函数均不构建梯度计算图。"""
    return x + y


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3
c1 = add_tensors1(a, b)
print("\n普通函数输出：")
print(c1)
c2 = add_tensors2(a, b)
print("no_grad 装饰函数输出：")
print(c2)

x = torch.rand(5, requires_grad=True)
y = x.detach()
print("\n原 Tensor x：")
print(x)
print("脱离计算图的 y：")
print(y)


"""
七、Autograd 与原地操作

反向传播可能需要前向计算保存的中间值。原地操作若覆盖这些值，就无法得到
正确梯度。因此，对 requires_grad=True 的叶子 Tensor 执行原地修改通常会报错。
下面保留官方错误示例，但不执行，以保证本文件可以完整运行。
"""

# a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)
# torch.sin_(a)  # RuntimeError：不能原地修改需要梯度的叶子 Tensor。


"""
八、Autograd Profiler

Profiler 能记录各个运算耗时，帮助定位训练或推理瓶颈。这里重复执行一组简单
Tensor 运算，再按 CPU 自身耗时输出统计表。
"""

run_on_gpu = torch.cuda.is_available()
x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(
    use_device="cuda" if run_on_gpu else None
) as profiler_result:
    for _ in range(1000):
        z = (z / x) * y

print("\nProfiler 统计：")
print(profiler_result.key_averages().table(sort_by="self_cpu_time_total"))


"""
九、进一步理解 Jacobian 与向量-Jacobian 乘积

向量值函数的导数是 Jacobian 矩阵。神经网络通常不显式构造完整 Jacobian，
而是让 backward(v) 计算向量 v 与 Jacobian 的乘积。输出不是标量时，必须
提供与输出形状相同的上游梯度向量。
"""

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print("\n向量输出 y：")
print(y)

upstream_gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(upstream_gradients)
print("向量-Jacobian 乘积，保存在 x.grad 中：")
print(x.grad)


"""
十、Autograd 高层函数式 API

torch.autograd.functional 提供直接计算 Jacobian 和向量-Jacobian 乘积的接口。
函数式 API 接收函数及其输入，不需要手动调用 backward()。
"""


def exp_adder(x, y):
    """分别对两个输入执行指数与线性变换。"""
    return 2 * x.exp() + 3 * y


inputs = (torch.rand(1), torch.rand(1))
print("\n单元素输入：", inputs)
print("对应 Jacobian：")
print(torch.autograd.functional.jacobian(exp_adder, inputs))

inputs = (torch.rand(3), torch.rand(3))
print("\n三元素输入：", inputs)
print("对应 Jacobian：")
print(torch.autograd.functional.jacobian(exp_adder, inputs))


def do_some_doubling(x):
    """反复把输入加倍，直到向量范数达到阈值。"""
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y


inputs = torch.randn(3)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
print("\n函数式 API 计算向量-Jacobian 乘积：")
print(torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients))

