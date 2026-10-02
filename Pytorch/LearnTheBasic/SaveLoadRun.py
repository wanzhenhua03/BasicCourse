"""PyTorch 模型保存与加载基础教程。

参考：https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

运行：
    conda activate MLLearn
    python .\Pytorch\SaveLoadRun.py

官方页面使用预训练 VGG16，会额外下载约 528 MB 权重并产生较大文件。
为了使本项目可直接、快速运行，这里使用不下载预训练权重的 SqueezeNet，
但保留官方教程的学习顺序与核心 API。
"""

from pathlib import Path

import torch
import torchvision.models as models


# 将运行产生的模型文件统一放在 Pytorch/checkpoints 中。
checkpoint_directory = Path(__file__).resolve().parent / "checkpoints"
checkpoint_directory.mkdir(parents=True, exist_ok=True)
weights_path = checkpoint_directory / "squeezenet_weights.pth"
full_model_path = checkpoint_directory / "squeezenet_model.pth"


# state_dict 是一个字典，保存模型可学习的权重和偏置。
# weights=None 创建随机初始化模型，不会访问网络下载预训练参数。
model = models.squeezenet1_0(weights=None)

# 推荐保存 state_dict，因为它只保存参数，不绑定完整 Python 对象结构。
torch.save(model.state_dict(), weights_path)
print(f"已保存模型权重：{weights_path}")


# 加载权重时，首先必须创建结构完全相同的模型。
loaded_weights_model = models.squeezenet1_0(weights=None)

# weights_only=True 限制反序列化过程只加载权重所需的安全类型。
loaded_state_dict = torch.load(weights_path, weights_only=True)
loaded_weights_model.load_state_dict(loaded_state_dict)

# 推理前调用 eval()，使 Dropout 和 BatchNorm 使用评估模式。
model.eval()
loaded_weights_model.eval()

# 用同一批随机输入检查加载前后的输出是否一致。
sample_input = torch.rand(1, 3, 224, 224)
with torch.no_grad():
    original_output = model(sample_input)
    loaded_output = loaded_weights_model(sample_input)
print(
    "state_dict 加载后输出一致：",
    torch.allclose(original_output, loaded_output),
)


# 也可以直接保存完整模型对象，即同时保存结构和参数。
torch.save(loaded_weights_model, full_model_path)
print(f"已保存完整模型：{full_model_path}")

# 完整模型加载依赖 Python pickle，所以需要 weights_only=False。
# 这种方式要求加载时仍然能导入原模型类，可移植性低于 state_dict。
loaded_full_model = torch.load(full_model_path, weights_only=False)
loaded_full_model.eval()

with torch.no_grad():
    full_model_output = loaded_full_model(sample_input)
print(
    "完整模型加载后输出一致：",
    torch.allclose(original_output, full_model_output),
)

print("\n建议：实际项目中优先保存和加载 state_dict。")
