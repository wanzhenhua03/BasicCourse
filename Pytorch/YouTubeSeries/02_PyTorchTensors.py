"""第 2 课：深入理解 PyTorch Tensor。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html
配套视频：https://www.youtube.com/watch?v=r7QDUPb2dCM

本文件按照官方课程顺序介绍 Tensor 的创建、随机数、形状与类型、数学运算、
广播、原地操作、复制、设备、形状变换，以及与 NumPy 的互操作。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\02_PyTorchTensors.py
"""

import math

import numpy as np
import torch


"""
一、创建 Tensor

torch.empty() 只分配内存，不初始化其中的数值，因此打印内容取决于这块内存
原有的数据。torch.zeros()、torch.ones() 和 torch.rand() 则会明确填充数值。
"""

x = torch.empty(3, 4)
print("torch.empty 创建的对象类型：", type(x))
print(x)

zeros = torch.zeros(2, 3)
print("\n全零 Tensor：")
print(zeros)

ones = torch.ones(2, 3)
print("\n全一 Tensor：")
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print("\n均匀分布随机 Tensor：")
print(random)


"""
二、随机 Tensor 与随机种子

伪随机数生成器每调用一次就沿随机序列前进。重新设置同一个种子后，后续生成
的数值序列可以复现，这对于调试和比较实验很重要。
"""

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print("\nrandom1：")
print(random1)

random2 = torch.rand(2, 3)
print("\nrandom2：")
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print("\n重新设种子后的 random3，应等于 random1：")
print(random3)

random4 = torch.rand(2, 3)
print("\nrandom4，应等于 random2：")
print(random4)


"""
三、Tensor 形状

shape 描述各维度长度。以已有 Tensor 为模板的 *_like() 系列函数，会保留
模板的形状与数据类型，但按函数名称决定新 Tensor 的内容。
"""

x = torch.empty(2, 2, 3)
print("\n三维 Tensor 的形状：", x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print("\nempty_like 的形状：", empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print("\nzeros_like 的形状：", zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print("\nones_like 的形状：", ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print("\nrand_like 的形状：", rand_like_x.shape)
print(rand_like_x)

# 也可以直接从 Python 列表或元组提供具体数据。
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print("\n从嵌套列表创建：")
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print("\n从元组创建：")
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print("\n嵌套集合产生多维 Tensor：")
print(more_integers)


"""
四、Tensor 数据类型

创建 Tensor 时可通过 dtype 指定底层类型，也可使用 to() 转换现有 Tensor。
数据类型会影响存储空间、数值范围、计算精度以及可用运算。
"""

a = torch.ones((2, 3), dtype=torch.int16)
print("\nint16 Tensor：")
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.0
print("\nfloat64 Tensor：")
print(b)

c = b.to(torch.int32)
print("\n转换为 int32 后，小数部分被截断：")
print(c)


"""
五、Tensor 的数学与逻辑运算

Tensor 可以与标量运算，也可以在形状相容时彼此逐元素运算。普通 Python
运算符会返回新的 Tensor，并遵循通常的运算优先级。
"""

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos**2
sqrt2s = twos**0.5

print("\n全一：\n", ones)
print("全二：\n", twos)
print("全三：\n", threes)
print("全四：\n", fours)
print("2 的平方根：\n", sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print("\n逐元素幂运算：")
print(powers2)

fives = ones + fours
print("\n逐元素加法：")
print(fives)

dozens = threes * fours
print("\n逐元素乘法：")
print(dozens)

t1 = torch.ones(3,4)
t2 = torch.ones(4,6)*2
t3 = t1.matmul(t2)
print(t3)


"""
六、Tensor 广播

当两个 Tensor 形状不完全相同时，PyTorch 会从最后一个维度向前比较：
两个维度必须相等、其中一个为 1，或者其中一个维度不存在。满足这些条件时，
较小的 Tensor 会在逻辑上扩展，而无需实际复制数据。
"""

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)
print("\n原 Tensor：")
print(rand)
print("广播后逐元素乘 2：")
print(doubled)

a = torch.ones(4, 3, 2)
b = a * torch.rand(3, 2)
print("\n(4, 3, 2) 与 (3, 2) 广播后的形状：", b.shape)

c = a * torch.rand(3, 1)
print("(4, 3, 2) 与 (3, 1) 广播后的形状：", c.shape)

d = a * torch.rand(1, 2)
print("(4, 3, 2) 与 (1, 2) 广播后的形状：", d.shape)

# 以下形状不满足广播规则，因此官方教程把它们作为错误示例而不实际执行。
# a * torch.rand(4, 3)      # 最后两个维度为 (3, 2) 与 (4, 3)，不相容。
# a * torch.rand(2, 3)      # 最后维度 2 与 3 不相容。
# a * torch.rand((0,))      # 长度为 0 的维度不能扩展为非零维度。


"""
七、更多数学运算

PyTorch 提供常用数学函数、三角函数、位运算、比较、归约，以及向量和线性代数
运算。这里按官方示例观察其中一小部分。
"""

a = torch.rand(2, 4) * 2 - 1
print("\n常用函数：")
print("绝对值：\n", torch.abs(a))
print("向上取整：\n", torch.ceil(a))
print("向下取整：\n", torch.floor(a))
print("截断到 [-0.5, 0.5]：\n", torch.clamp(a, -0.5, 0.5))

angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print("\n角度、正弦和反正弦：")
print(angles)
print(sines)
print(inverses)

print("\n按位异或：")
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

print("\n支持广播的逐元素相等比较：")
d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
e = torch.ones(1, 2)
print(torch.eq(d, e))

print("\n归约运算：")
print("最大值 Tensor：", torch.max(d))
print("提取出的 Python 数值：", torch.max(d).item())
print("平均值：", torch.mean(d))
print("标准差：", torch.std(d))
print("所有元素之积：", torch.prod(d))
print("去重：", torch.unique(torch.tensor([1, 2, 1, 2, 1, 2])))

v1 = torch.tensor([1.0, 0.0, 0.0])
v2 = torch.tensor([0.0, 1.0, 0.0])
m1 = torch.rand(2, 2)
m2 = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
print("\n向量与矩阵运算：")
print("v2 × v1：", torch.linalg.cross(v2, v1))
print("随机矩阵 m1：\n", m1)
m3 = torch.matmul(m1, m2)
print("m1 与 3 倍单位矩阵相乘：\n", m3)
print("奇异值分解：\n", torch.linalg.svd(m3))


"""
八、原地修改 Tensor

名称以下划线结尾的方法通常执行原地操作，即直接改变调用它的 Tensor。
原地操作可以减少分配，但可能破坏自动微分所需的历史值，因此应谨慎使用。
"""

a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print("\na：")
print(a)
print("torch.sin(a) 返回新 Tensor：")
print(torch.sin(a))
print("a 本身不变：")
print(a)

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print("\nb：")
print(b)
print("b.sin_() 的结果：")
print(b.sin_())
print("b 已被改变：")
print(b)

a = torch.ones(2, 2)
b = torch.rand(2, 2)
print("\n原地加法之前：")
print(a)
print(b)
print("a.add_(b)：")
print(a.add_(b))
print("a：\n", a)
print("b：\n", b)
print("b.mul_(b)：")
print(b.mul_(b))

"""
带 out 参数的函数把结果写入指定 Tensor。下面验证 c 与返回值 d 是同一个对象，
而且写入前后 c 的对象标识不变。
"""

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)
print("\nout 写入前的 c：\n", c)
d = torch.matmul(a, b, out=c)
print("out 写入后的 c：\n", c)
assert c is d
assert id(c) == old_id

torch.rand(2, 2, out=c)
print("再次使用 out 改写 c：\n", c)
assert id(c) == old_id


"""
九、复制 Tensor

普通赋值只创建另一个引用；clone() 才会复制数据。若源 Tensor 正在跟踪梯度，
clone() 得到的副本也会保留计算关系；detach().clone() 可得到不跟踪梯度的副本。
"""

a = torch.ones(2, 2)
b = a
a[0][1] = 561
print("\n普通赋值后，修改 a 也会影响 b：")
print(b)

a = torch.ones(2, 2)
b = a.clone()
assert b is not a
print("\nclone 后内容相等：")
print(torch.eq(a, b))
a[0][1] = 561
print("修改 a 后，b 仍保持不变：")
print(b)

a = torch.rand(2, 2, requires_grad=True)
print("\n启用自动微分的 a：")
print(a)
b = a.clone()
print("clone 后的 b 仍跟踪梯度：")
print(b)
c = a.detach().clone()
print("detach().clone() 后的 c 不跟踪梯度：")
print(c)


"""
十、在可用设备上计算

Tensor 默认在 CPU 上创建。若存在加速器，可以在创建时用 device 指定设备，
也可以对现有 Tensor 调用 to(device)。参与同一次运算的 Tensor 必须位于同一设备。
"""

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

x = torch.rand(2, 2, device=my_device)
print(x)

y = torch.rand(2, 2)
y = y.to(my_device)
print("移动后的 y 所在设备：", y.device)

# 下列写法会因设备不同而失败，所以不实际运行。
# cpu_tensor = torch.rand(2, 2)
# cuda_tensor = torch.rand(2, 2, device="cuda")
# cpu_tensor + cuda_tensor


"""
十一、操作 Tensor 形状

unsqueeze() 在指定位置加入长度为 1 的维度；squeeze() 删除长度为 1 的维度。
它们常用于补充 batch 维或让 Tensor 满足广播要求。
"""

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)
print("\n加入 batch 维之前：", a.shape)
print("加入 batch 维之后：", b.shape)

c = torch.rand(1, 1, 1, 1, 1)
print("多个长度为 1 的维度：", c.shape)

a = torch.rand(1, 20)
print("\nsqueeze 之前：", a.shape)
b = a.squeeze(0)
print("squeeze(0) 之后：", b.shape)

c = torch.rand(2, 2)
print("\n不能删除长度大于 1 的维度：", c.shape)
d = c.squeeze(0)
print("squeeze(0) 后形状不变：", d.shape)

a = torch.ones(4, 3, 2)
b = torch.rand(3)
c = b.unsqueeze(1)
print("\n将 (3,) 调整为可广播的形状：", c.shape)
print(a * c)

batch_me = torch.rand(3, 226, 226)
print("\n原地图像形状：", batch_me.shape)
batch_me.unsqueeze_(0)
print("原地加入 batch 维：", batch_me.shape)

"""
reshape() 在元素总数不变的前提下改变形状。它可能返回共享底层存储的视图，
也可能创建副本，因此不要依赖修改其中一个一定会影响另一个。
"""

output3d = torch.rand(6, 20, 20)
print("\n三维输出形状：", output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print("展平后的形状：", input1d.shape)
print("torch.reshape 的等价写法：", torch.reshape(output3d, (6 * 20 * 20,)).shape)


"""
十二、NumPy Bridge

torch.from_numpy() 可从 NumPy ndarray 建立 Tensor，Tensor.numpy() 则执行反向
转换。在 CPU 上，这两种转换通常共享底层内存，所以一方的修改会反映到另一方。
"""

numpy_array = np.ones((2, 3))
print("\nNumPy 数组：")
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print("转换得到的 PyTorch Tensor：")
print(pytorch_tensor)

pytorch_rand = torch.rand(2, 3)
print("\nPyTorch 随机 Tensor：")
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print("转换得到的 NumPy 数组：")
print(numpy_rand)

numpy_array[1, 1] = 23
print("\n修改 NumPy 数组后，共享内存的 Tensor：")
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print("修改 Tensor 后，共享内存的 NumPy 数组：")
print(numpy_rand)

