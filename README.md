# BasicCourse

本项目用于个人 AI 基础课程学习，包括 PyTorch、LLM、RL 和 VLM。
通过 GitHub 在笔记本与服务器之间同步代码和项目说明。

## 学习课程

- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

## 目录说明

- `Pytorch/`：PyTorch 学习代码。
- `AGENTS.md`：项目约定、环境要求和编码要求。
- `.gitignore`：数据、模型、日志、缓存和本机配置的排除规则。
- `asset/`：本地学习参考资料，目前未纳入首次提交。

## 运行环境

本项目统一使用 Conda 环境 `MLLearn`。当前笔记本已验证的核心环境：

| 组件 | 版本或要求 |
| --- | --- |
| Python | 3.9.25 |
| PyTorch | 2.7.1+cu118 |
| TorchVision | 0.22.1+cu118 |
| CUDA Runtime | 11.8 |
| NumPy | 2.0.2 |
| OpenBLAS | libopenblas 0.3.34，pthreads 后端 |

`libblas`、`libcblas` 和 `liblapack` 必须保持 OpenBLAS 构建，避免
MKL/LLVM OpenMP 与 PyTorch 自带 Intel OpenMP 在同一进程中冲突。
环境创建、切换，以及依赖安装、卸载或升级，均须遵循 `AGENTS.md`，先获得用户明确许可。

运行下面的命令前，需要在对应机器上已有可用的 `MLLearn` 环境。
拉取代码不会自动安装依赖或创建环境；服务器的操作系统、GPU、驱动和依赖需要分别核对、验证。
`AGENTS.md` 中的 Windows 解释器路径适用于当前笔记本，服务器应选择其自身 `MLLearn` 环境的解释器。

## 运行示例

在项目根目录使用 PowerShell 执行：

```powershell
conda activate MLLearn
python .\Pytorch\LearnTheBasic\Tensor.py
```

也可以直接在指定环境中执行，以下写法可用于 PowerShell 或 Linux shell：

```text
conda run --name MLLearn python Pytorch/LearnTheBasic/Tensor.py
```

部分课程脚本会下载数据、显示图像或执行训练，运行前请阅读对应文件的说明。
无图形界面的服务器需要根据脚本需求处理绘图显示。

## 跨机器同步

服务器首次获取代码，在准备存放项目的父目录执行：

```text
git clone https://github.com/wanzhenhua03/BasicCourse.git
cd BasicCourse
```

日常在任一机器开始修改前，检查本地状态，再拉取远端版本：

```text
git status
git pull --ff-only
```

若有未提交修改，先保存和处理；若两端提交历史已经分叉，先处理合并，再继续同步。
修改完成并在 `MLLearn` 中验证后，选择文件、提交并推送。下面以项目说明为例：

```text
git add -- README.md
git commit -m "更新项目说明"
git push
```

修改 Python 文件时，将 `README.md` 替换为对应文件路径；提交前使用
`git diff --cached --name-only` 检查文件清单。
两台机器都通过具有仓库写入权限的 GitHub 认证后，可以按同样的流程双向同步。

数据集、模型权重、训练日志和缓存默认不纳入 Git，需在对应机器分别准备或单独传输。
首次提交包含学习代码和项目说明；本地 `asset/` 与 `.vscode/` 未纳入此次提交。
