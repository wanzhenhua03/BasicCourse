# Course
1. 本项目用于进行个人的AI基础课程学习

# 学习课程
包括但不限于：
1. Pytorch: https://docs.pytorch.org/tutorials/
2. LLM: https://huggingface.co/learn/llm-course/chapter1/1
3. RL: https://github.com/huggingface/trl、https://huggingface.co/learn/deep-rl-course/unit0/introduction
4. vlm: https://github.com/QwenLM/Qwen3-VL

# 项目全局约定
- 尽量使用英文进行推理或思考，用中文回答和书写文档或注释等（专业词汇、概念除外）。
- 必须采用：__调研（基于事实）->思考（理解用户意图/分析问题）->规划/设计方案->讨论/对齐需求->确认行动__ 的工作流程，尤其是在复杂任务/用户描述不清/某些地方存疑时，务必先询问和讨论清楚后再行动。

# 项目依赖环境
- 本项目统一使用 Conda 虚拟环境 `MLLearn`。
- 在 PowerShell 中运行项目前，使用 `conda activate MLLearn` 激活环境；无需激活时，可使用 `conda run --name MLLearn <command>` 在该环境中执行命令。
- VS Code 的 Python 解释器应选择 `D:\Users\14404\anaconda3\envs\MLLearn\python.exe`。
- 当前已验证的核心环境为：Python `3.9.25`、PyTorch `2.7.1+cu118`、TorchVision `0.22.1+cu118`、CUDA Runtime `11.8`。
- NumPy `2.0.2` 使用 `libopenblas 0.3.34` 的 `pthreads` 后端；`libblas`、`libcblas` 和 `liblapack` 必须保持 OpenBLAS 构建，避免 MKL/LLVM OpenMP 与 PyTorch 自带 Intel OpenMP 在同一进程中冲突。
- 该环境已通过 PyTorch 张量运算、自动微分、TorchVision 算子和 NVIDIA GPU 运算测试。
- 未经用户明确许可，不得新建或切换项目环境，不得在 `MLLearn` 中安装、卸载或升级依赖；如需变更，必须先说明原因、影响和具体命令，经用户确认后再执行。


# 编码要求
- 编码时，尽量采用简单的数据结构，代码简单易懂、结构和流程清晰，避免复杂的跳转或调用关系。
- __非必要不要增加过多的文件和代码，尽量复用已有的代码或第三方库，避免过多的冗余；__
- 编码时，每段代码都应该有充足且清晰的注释；对于声明的变量必须写清其用途等。
- 进行代码更新后，同步更新维护其注释和说明文档等。
- __在运行项目时，务必使用本文件中明确声明的项目依赖环境，除非得到用户明确许可，严禁私自增删、修改或切换依赖环境。__
- 务必严格按照用户指令行动，非必要不修改指令以外的代码或数据。
- __项目的所有代码必须明确编译、运行环境，修改项目依赖环境必须先询问用户，不得私自增删或修改环境__
- 所有的报告、文档等，应当保持前后逻辑清晰连贯、易读易懂、用平实的语言解释；除非必要情况，不要自己产生一些生造/晦涩的名词，不要通篇都是专业名词或黑话的文字，不要生成一些云里雾里或务虚的空话。

