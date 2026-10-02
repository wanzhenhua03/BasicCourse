"""第 7 课：使用 Captum 理解模型。

官方教程：
https://docs.pytorch.org/tutorials/beginner/introyt/captumyt.html
配套视频：https://www.youtube.com/watch?v=Am2EF9CLu-g
官方素材：https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip

本文件依照官方课程顺序介绍特征归因、层归因、Integrated Gradients、
Occlusion 和 Layer Grad-CAM，并保留相应代码。

运行环境：Conda 环境 MLLearn
PowerShell：
    conda activate MLLearn
    python .\Pytorch\YouTubeSeries\07_ModelUnderstandingWithCaptum.py

当前项目决定不修改 MLLearn 的 NumPy 和 OpenBLAS 配置。因此，本脚本在
Captum 未安装时只输出说明并正常结束，不下载图片或 ResNet18 权重。
"""

import importlib.util


# 必须在导入 Captum、下载素材和创建模型之前检查依赖。
if importlib.util.find_spec("captum") is None:
    print("第 7 课需要 Captum，但当前 MLLearn 环境没有安装该包。")
    print("为保护项目现有 NumPy 2.0.2 与 OpenBLAS 配置，本脚本不会自动安装依赖。")
    print("课程讲解与完整示例代码已保留；获得环境修改许可并安装 Captum 后可继续运行。")
    raise SystemExit(0)


import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
from captum.attr import LayerAttribution
from captum.attr import LayerGradCam
from captum.attr import Occlusion
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR / "img"
ASSET_URL = (
    "https://pytorch-tutorial-assets.s3.amazonaws.com/"
    "youtube-series/video7.zip"
)


def ensure_official_assets():
    """缺少课程图片时，下载官方压缩包并只提取本课所需的两个文件。"""
    cat_path = IMAGE_DIR / "cat.jpg"
    labels_path = IMAGE_DIR / "imagenet_class_index.json"
    if cat_path.exists() and labels_path.exists():
        return cat_path, labels_path

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("正在下载第 7 课官方示例素材……")

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "video7.zip"
        urllib.request.urlretrieve(ASSET_URL, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            asset_members = {
                "video7/img/cat.jpg": cat_path,
                "video7/img/imagenet_class_index.json": labels_path,
            }
            for member_name, target_path in asset_members.items():
                with archive.open(member_name) as source_file:
                    target_path.write_bytes(source_file.read())

    return cat_path, labels_path


"""
一、Captum 与模型归因

Captum 是构建在 PyTorch 之上的模型可解释性库。它把“模型为何产生某个输出”
描述为 attribution（归因）问题：特征归因解释输入特征的贡献；层归因研究隐藏层
响应；神经元归因进一步聚焦单个神经元。本课演示前两种。

归因算法主要包括两类：梯度方法从目标输出向输入或隐藏层反向传播；扰动方法
有计划地改变输入，再观察输出变化。可视化工具把归因值映射回图像空间，帮助人
理解模型关注了哪些区域，但这些结果仍需结合算法假设谨慎解释。
"""


"""
二、第一个示例：预训练 ResNet18

先加载在 ImageNet 上预训练的 ResNet18，再读取官方猫图片。模型期望输入为
224×224 的三通道图像，并使用 ImageNet 训练时的通道均值和标准差归一化。
"""

model = models.resnet18(weights="IMAGENET1K_V1")
model = model.eval()

cat_image_path, labels_path = ensure_official_assets()
test_img = Image.open(cat_image_path).convert("RGB")
test_img_data = np.asarray(test_img)
plt.imshow(test_img_data)
plt.title("原始输入图像")
plt.axis("off")
plt.show()

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0)

with labels_path.open(encoding="utf-8") as labels_file:
    index_to_labels = json.load(labels_file)

output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, predicted_label_index = torch.topk(output, 1)
predicted_label_index.squeeze_()
predicted_label = index_to_labels[str(predicted_label_index.item())][1]
print(
    "预测类别：",
    predicted_label,
    "，置信度：",
    prediction_score.squeeze().item(),
)


"""
三、使用 Integrated Gradients 进行特征归因

Integrated Gradients 从一个基线输入出发，沿通向真实输入的路径积分梯度，
为每个输入特征分配对指定目标输出的贡献分数。下面解释模型当前预测类别，
并把正向贡献以热力图形式叠加到图像空间。
"""

integrated_gradients = IntegratedGradients(model)
attributions_ig = integrated_gradients.attribute(
    input_img,
    target=predicted_label_index,
    n_steps=200,
)

_ = viz.visualize_image_attr(
    None,
    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    method="original_image",
    title="原始图像",
)

default_cmap = LinearSegmentedColormap.from_list(
    "custom blue",
    [(0, "#ffffff"), (0.25, "#0000ff"), (1, "#0000ff")],
    N=256,
)

_ = viz.visualize_image_attr(
    np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    method="heat_map",
    cmap=default_cmap,
    show_colorbar=True,
    sign="positive",
    title="Integrated Gradients",
)
plt.show()


"""
四、使用 Occlusion 进行特征归因

Occlusion 是扰动方法：用基线值遮挡输入的局部窗口，并观察目标输出如何变化。
窗口大小决定一次遮挡的区域，stride 决定窗口移动间隔。对图像逐位置执行模型
推理会比一次反向传播昂贵，但其解释不直接依赖梯度。
"""

occlusion = Occlusion(model)
attributions_occ = occlusion.attribute(
    input_img,
    target=predicted_label_index,
    strides=(3, 8, 8),
    sliding_window_shapes=(3, 15, 15),
    baselines=0,
)

_ = viz.visualize_image_attr_multiple(
    np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    ["original_image", "heat_map", "heat_map", "masked_image"],
    ["all", "positive", "negative", "positive"],
    show_colorbar=True,
    titles=["原图", "正向归因", "负向归因", "掩膜结果"],
    fig_size=(18, 6),
)
plt.show()


"""
五、使用 Layer Grad-CAM 进行层归因

Layer Grad-CAM 计算目标输出对指定卷积层通道的梯度，对空间位置求平均后作为
通道权重，再与激活相乘。卷积层输出尺寸通常小于输入图像，因此可使用
LayerAttribution.interpolate() 把归因结果上采样到输入尺寸后叠加显示。
"""

layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(
    input_img,
    target=predicted_label_index,
)

_ = viz.visualize_image_attr(
    attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
    sign="all",
    title="Layer 3 Block 1 Conv 2",
)
plt.show()

upsampled_attributions_lgc = LayerAttribution.interpolate(
    attributions_lgc,
    input_img.shape[2:],
)

print("原始层归因形状：", attributions_lgc.shape)
print("上采样后归因形状：", upsampled_attributions_lgc.shape)
print("模型输入形状：", input_img.shape)

_ = viz.visualize_image_attr_multiple(
    upsampled_attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
    transformed_img.permute(1, 2, 0).numpy(),
    ["original_image", "blended_heat_map", "masked_image"],
    ["all", "positive", "positive"],
    show_colorbar=True,
    titles=["原图", "正向归因叠加", "掩膜结果"],
    fig_size=(18, 6),
)
plt.show()


"""
归因可视化能够揭示模型用到了哪些输入区域或隐藏层响应，但不能单独证明因果
关系，也不能保证模型在其他样本上使用相同策略。实际分析应比较多种算法、
多种基线与大量样本，并结合领域知识验证解释是否稳定可信。
"""

