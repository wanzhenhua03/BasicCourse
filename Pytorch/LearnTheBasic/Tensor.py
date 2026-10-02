"""PyTorch Tensor 基础学习脚本。

参考资料：
https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

本文件是基于 PyTorch 官方 Tensor 基础教程整理的原创中文讲解，
不是对官方网页的逐句翻译。可以直接阅读注释，也可以运行脚本观察结果。

本节学习目标：
1. 理解 Tensor 在 PyTorch 中的作用。
2. 掌握创建 Tensor 的常用方法。
3. 会查看 Tensor 的形状、数据类型和所在设备。
4. 掌握索引、切片、拼接、矩阵乘法和逐元素运算。
5. 理解原地操作的特点和风险。
6. 理解 CPU Tensor 与 NumPy 数组共享内存的现象。

项目运行环境：Conda 环境 MLLearn。
PowerShell 运行命令：
    conda activate MLLearn
    python .\Pytorch\Tensor.py
"""

import numpy as np
import torch


def print_section(title: str) -> None:
    """输出统一格式的章节标题，便于阅读脚本运行结果。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demonstrate_tensor_creation() -> None:
    """演示从 Python 数据、NumPy 数组和其他 Tensor 创建 Tensor。"""
    print_section("1. 创建 Tensor")

    # Tensor 是 PyTorch 中表示数据的基本结构。
    # 模型的输入、输出和参数通常都由 Tensor 表示。

    # 从 Python 列表直接创建 Tensor。
    # torch.tensor 会根据输入值自动推断数据类型。
    source_data = [[1, 2], [3, 4]]
    tensor_from_data = torch.tensor(source_data)
    print("从 Python 列表创建：")
    print(tensor_from_data)
    print(f"自动推断的类型：{tensor_from_data.dtype}")

    # torch.from_numpy 从 NumPy 数组创建 Tensor。
    # 默认情况下，两者共享底层 CPU 内存。
    numpy_array = np.array(source_data)
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print("\n从 NumPy 数组创建：")
    print(tensor_from_numpy)

    # *_like 函数使用已有 Tensor 的形状和其他属性创建新 Tensor。
    # ones_like 保留原 Tensor 的形状和数据类型。
    ones_like_data = torch.ones_like(tensor_from_data)
    print("\n与已有 Tensor 形状、类型相同的全 1 Tensor：")
    print(ones_like_data)

    # rand_like 要生成随机小数，因此显式把数据类型改为 float32。
    random_like_data = torch.rand_like(tensor_from_data, dtype=torch.float32)
    print("\n与已有 Tensor 形状相同的随机浮点 Tensor：")
    print(random_like_data)

    # shape 是表示各个维度长度的元组。
    # 此处要创建一个 2 行 3 列的 Tensor。
    target_shape = (2, 3)
    random_tensor = torch.rand(target_shape)
    ones_tensor = torch.ones(target_shape)
    zeros_tensor = torch.zeros(target_shape)

    print(f"\n形状为 {target_shape} 的随机 Tensor：")
    print(random_tensor)
    print(f"\n形状为 {target_shape} 的全 1 Tensor：")
    print(ones_tensor)
    print(f"\n形状为 {target_shape} 的全 0 Tensor：")
    print(zeros_tensor)


def demonstrate_tensor_attributes_and_devices() -> None:
    """演示 Tensor 的形状、类型、设备属性和设备迁移。"""
    print_section("2. Tensor 属性与计算设备")

    # torch.rand 默认创建 float32 类型的 CPU Tensor。
    attribute_tensor = torch.rand(3, 4)
    print(f"形状 shape：{attribute_tensor.shape}")
    print(f"数据类型 dtype：{attribute_tensor.dtype}")
    print(f"存储设备 device：{attribute_tensor.device}")

    # 如果 PyTorch 检测到可用加速器，就使用 .to(device) 移动 Tensor。
    # 在本项目当前的 MLLearn 环境中，该加速器是 CUDA GPU。
    if torch.accelerator.is_available():
        accelerator_device = torch.accelerator.current_accelerator()
        accelerated_tensor = attribute_tensor.to(accelerator_device)
        print(f"\n已将 Tensor 移动到：{accelerated_tensor.device}")

        # CUDA 设备可以进一步查询实际的 GPU 名称。
        if accelerated_tensor.device.type == "cuda":
            cuda_index = accelerated_tensor.device.index or 0
            print(f"GPU 名称：{torch.cuda.get_device_name(cuda_index)}")
    else:
        print("\n未检测到可用加速器，继续使用 CPU。")

    # 设备之间复制大量数据有时会成为性能瓶颈，
    # 因此通常应避免在训练循环中反复往返移动同一份数据。


def demonstrate_indexing_and_joining() -> None:
    """演示 Tensor 的索引、切片、修改、拼接和堆叠。"""
    print_section("3. 索引、切片与拼接")

    # 创建 4 行 4 列的全 1 Tensor，用来演示索引。
    matrix = torch.ones(4, 4)
    print("原始 Tensor：")
    print(matrix)

    # matrix[0] 选择第一行。
    print(f"\n第一行 matrix[0]：{matrix[0]}")

    # matrix[:, 0] 表示所有行的第一列。
    print(f"第一列 matrix[:, 0]：{matrix[:, 0]}")

    # 省略号 ... 代表补齐其他维度，-1 表示最后一列。
    print(f"最后一列 matrix[..., -1]：{matrix[..., -1]}")

    # 通过索引赋值，把第二列全部改为 0。
    matrix[:, 1] = 0
    print("\n将第二列设为 0 后：")
    print(matrix)

    # torch.cat 在已有维度上连接 Tensor。
    # dim=1 表示沿列方向连接，结果形状为 [4, 12]。
    concatenated = torch.cat([matrix, matrix, matrix], dim=1)
    print(f"\ncat 沿列方向拼接后的形状：{concatenated.shape}")
    print(concatenated)

    # torch.stack 与 cat 不同：stack 会新增一个维度。
    # 三个 [4, 4] Tensor 沿新维度堆叠后的形状为 [3, 4, 4]。
    stacked = torch.stack([matrix, matrix, matrix], dim=0)
    print(f"\nstack 新增维度后的形状：{stacked.shape}")


def demonstrate_arithmetic_operations() -> None:
    """演示矩阵乘法、逐元素乘法、聚合和标量提取。"""
    print_section("4. Tensor 运算")

    # 固定随机种子，使本节每次运行时都得到相同的随机数。
    torch.manual_seed(42)
    left_matrix = torch.rand(2, 3)
    right_matrix = torch.rand(3, 2)
    print("左矩阵：")
    print(left_matrix)
    print("\n右矩阵：")
    print(right_matrix)

    # @、Tensor.matmul 和 torch.matmul 都能执行矩阵乘法。
    matrix_product_operator = left_matrix @ right_matrix
    matrix_product_method = left_matrix.matmul(right_matrix)
    matrix_product_function = torch.empty_like(matrix_product_operator)
    torch.matmul(left_matrix, right_matrix, out=matrix_product_function)

    print("\n矩阵乘法 left_matrix @ right_matrix：")
    print(matrix_product_operator)
    print(
        "三种矩阵乘法写法的结果一致：",
        torch.allclose(matrix_product_operator, matrix_product_method)
        and torch.allclose(matrix_product_operator, matrix_product_function),
    )

    # *、Tensor.mul 和 torch.mul 执行逐元素乘法，而不是矩阵乘法。
    elementwise_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    elementwise_right = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    elementwise_product = elementwise_left * elementwise_right
    print("\n逐元素乘法 elementwise_left * elementwise_right：")
    print(elementwise_product)

    # sum 将所有元素聚合为只有一个值的 Tensor。
    aggregated_tensor = elementwise_product.sum()

    # item() 把单元素 Tensor 转换为普通 Python 数值。
    # 只有包含一个元素的 Tensor 才能使用 item()。
    aggregated_number = aggregated_tensor.item()
    print(f"\n求和后的单元素 Tensor：{aggregated_tensor}")
    print(f"转换后的 Python 数值：{aggregated_number}")
    print(f"Python 数值类型：{type(aggregated_number)}")


def demonstrate_in_place_operations() -> None:
    """演示会直接修改操作数的原地操作。"""
    print_section("5. 原地操作")

    # PyTorch 中以下划线 _ 结尾的方法通常是原地操作，
    # 即结果直接写回原 Tensor，不另外返回一份独立结果。
    original_tensor = torch.ones(2, 3)
    in_place_tensor = original_tensor.clone()
    print("执行 add_ 前：")
    print(in_place_tensor)

    # add_(5) 直接给 in_place_tensor 的每个元素加 5。
    in_place_tensor.add_(5)
    print("\n执行 add_(5) 后：")
    print(in_place_tensor)

    # 原地操作可能节省部分内存，但会覆盖旧数据。
    # 当 Tensor 参与自动微分时，覆盖历史值可能导致梯度计算出错，
    # 因此初学阶段建议优先使用非原地操作，除非明确知道其影响。


def demonstrate_numpy_bridge() -> None:
    """演示 CPU Tensor 和 NumPy 数组之间的共享内存。"""
    print_section("6. Tensor 与 NumPy 的互操作")

    # CPU Tensor 调用 numpy() 后，返回的 NumPy 数组通常共享同一块内存。
    tensor_values = torch.ones(5)
    numpy_view = tensor_values.numpy()
    print(f"Tensor：{tensor_values}")
    print(f"NumPy 视图：{numpy_view}")

    # 原地修改 Tensor 后，共享内存的 NumPy 数组也会同步变化。
    tensor_values.add_(1)
    print("\nTensor 执行 add_(1) 后：")
    print(f"Tensor：{tensor_values}")
    print(f"NumPy 视图：{numpy_view}")

    # torch.from_numpy 同样通常不复制数据，而是共享 NumPy 数组的内存。
    numpy_values = np.ones(5, dtype=np.float32)
    tensor_view = torch.from_numpy(numpy_values)

    # 原地修改 NumPy 数组后，Tensor 的内容也会变化。
    np.add(numpy_values, 1, out=numpy_values)
    print("\nNumPy 数组原地加 1 后：")
    print(f"NumPy 数组：{numpy_values}")
    print(f"Tensor 视图：{tensor_view}")

    # GPU Tensor 不能直接转换为 NumPy 数组。
    # 如果 Tensor 位于 GPU 且参与梯度计算，常见写法是：
    # numpy_array = gpu_tensor.detach().cpu().numpy()


def main() -> None:
    """按照教程知识顺序执行全部 Tensor 演示。"""
    print("PyTorch Tensor 基础教程")
    print(f"PyTorch 版本：{torch.__version__}")

    demonstrate_tensor_creation()
    demonstrate_tensor_attributes_and_devices()
    demonstrate_indexing_and_joining()
    demonstrate_arithmetic_operations()
    demonstrate_in_place_operations()
    demonstrate_numpy_bridge()

    print_section("7. 学习要点总结")
    print(
        "1. Tensor 是 PyTorch 中表示模型输入、输出和参数的基本数据结构。\n"
        "2. shape、dtype 和 device 是最常用的 Tensor 属性。\n"
        "3. 模型和参与运算的 Tensor 必须位于同一计算设备。\n"
        "4. @ 表示矩阵乘法，* 表示逐元素乘法。\n"
        "5. item() 只能把单元素 Tensor 转换为 Python 数值。\n"
        "6. 以 _ 结尾的原地操作会直接修改原数据，需谨慎使用。\n"
        "7. CPU Tensor 和 NumPy 数组可以共享底层内存。"
    )


if __name__ == "__main__":
    main()
