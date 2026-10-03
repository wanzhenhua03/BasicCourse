"""Hugging Face 第 11 章学习教程：对 Qwen3-0.6B 做全量 SFT。

一、先明确本课在做什么
======================
目标：用一小批英文日常对话，学习如何继续微调一个已经具备对话能力的小模型。
Qwen/Qwen3-0.6B 是支持 thinking / non-thinking 的对话模型，区别于
Qwen/Qwen3-0.6B-Base。本课采用 non-thinking 模式，不训练思考过程。
数据使用 smoltalk 的 everyday-conversations 子集；代码讲解和注释使用中文。
这是一轮用于理解流程的教学实验，不保证通用能力提高，也不是中文专项微调。

学习顺序：目标与环境 → 数据检查、去重与划分 → Chat Template 与 token →
loss 掩码 → 加载模型并开启全部参数的梯度 → 训练前评估 → SFT → 训练后评估 → 保存 → 重载推理。

SFT（Supervised Fine-Tuning，监督微调）描述训练任务：给定问题和示范答案，
让模型更可能生成示范答案。它仍然使用“预测下一个 token”的交叉熵目标。
全量微调描述更新参数的方法：原模型的全部参数都参与反向传播和优化器更新，
包括词嵌入、注意力层、MLP、归一化层和输出层，不冻结其中某些层。
“全量参数”与“监督全部 token”是两个不同问题：本课更新全部参数，但仍然
只在目标答案 token 上计算 loss，上下文 token 通过注意力影响答案预测。
本课不使用 LoRA、adapter 或量化，也不进行 DPO、RLHF 或强化学习。

二、环境与依赖：先检查，再按项目约定准备
======================================
仅使用项目既有的 Conda 环境 MLLearn；不要新建、切换环境或自动安装依赖。
本机已验证 Python 3.9.25、PyTorch 2.7.1+cu118、NumPy 2.0.2。
本课 API 参考组合为 Transformers 4.52.4、TRL 0.19.1、Accelerate 1.7.0，
另需 datasets >= 3.0。以上版本声明支持 Python 3.9；全量训练不需要 PEFT。
这不是已经跑过完整训练的环境锁文件；实际环境仍须验证间接依赖与 GPU。
本机目前缺少这三个库，GPU 为 GTX 1050 Ti（4GB），无法在当前环境验证训练。

若后续获得用户明确许可，可先审阅以下依赖解析计划，再决定是否执行安装：
    conda run --name MLLearn python -m pip install --dry-run transformers==4.52.4 trl==0.19.1 accelerate==1.7.0
实际安装命令是去掉 --dry-run 后的同一命令；本文件不会执行它。
这些库补齐 Qwen3、SFTTrainer 和设备管理支持；解析时可能更新
huggingface-hub、tokenizers 等间接依赖，须核对影响，保持项目既有
torch / torchvision / numpy 和 OpenBLAS 构建，不要顺手升级它们。

主要面向支持 BF16 的单张 NVIDIA GPU（例如已有课程提到的 RTX 4090，24GB）。
本课始终以 FP32 保存可训练权重和 AdamW 状态；支持时由 Trainer 开启 BF16
混合精度计算，不支持时全程 FP32。BF16 混合精度不等于把所有权重直接存成 BF16。
按 0.6B 参数粗略估算，FP32 权重、梯度、AdamW 的两个状态合计约
0.6×10^9×(4+4+4+4)=9.6×10^9 字节，约 8.9GiB；这还不含激活、
logits、临时计算与 CUDA 缓存，实际峰值需实测。本机 4GB 不适合此全量训练配置。
梯度检查点和较短序列减少激活开销，但不会消除全量梯度与优化器状态的开销。
完整权重和含优化器状态的 checkpoint 也较大，建议预留数十 GB 的输出空间。
CPU 可以学习流程，但全量训练很慢。
不额外依赖 bitsandbytes、FlashAttention、W&B 或 TensorBoard。

三、运行方式：先检查，后试跑，再进行完整教学实验
================================================
在项目根目录运行；Windows 使用 -X utf8，便于显示中文。--no-capture-output
避免 Conda 默认捕获输出时以本机 GBK 解码 UTF-8，训练日志也能直接显示。
    conda run --no-capture-output --name MLLearn python -X utf8 LLM/Qwen3SFTTutorial.py
    conda run --no-capture-output --name MLLearn python -X utf8 LLM/Qwen3SFTTutorial.py --help
默认只检查环境，不下载模型、不训练。依赖准备完成后：
    conda run --no-capture-output --name MLLearn python -X utf8 LLM/Qwen3SFTTutorial.py --stage train --sample-limit 64 --max-steps 2 --output-dir checkpoints/LLM/Qwen3_SFT_Full_smoke
    conda run --no-capture-output --name MLLearn python -X utf8 LLM/Qwen3SFTTutorial.py --stage train
    conda run --no-capture-output --name MLLearn python -X utf8 LLM/Qwen3SFTTutorial.py --stage infer --prompt "How can I start learning Python?"
试跑的两步只检查流程能否走通，不用于判断训练效果；默认完整实验训练 1 epoch。
--sample-limit 控制原始对话数量；拆成问答后，实际训练样本数量会更多。
--device cpu 可显式选择 CPU；不会因此改变 Conda 环境。

首次训练需要访问 Hugging Face，下载约十亿字节量级的模型权重及数据。
缓存遵循 Hugging Face 的 HF_HOME 设置。训练产物默认写入项目
checkpoints/LLM/Qwen3_SFT_Full/；相对 --output-dir 按项目根目录解析。
新训练要求输出目录为空，避免覆盖前一实验；重做实验可换一个输出目录。
infer 要指向已有全量实验的 --output-dir；它直接从 final_model/ 加载完整模型。

四、与原教程的对应关系及适配
==========================
1. 第 11.1、11.6 节：理解 SFT 的作用和完整流程。
2. 第 11.2 节：保留 Qwen3 自带模板，不用 setup_chat_format 覆盖已有模板。
3. 第 11.3 节：用 SFTConfig / SFTTrainer 管理训练，逐项解释关键参数。
4. 第 11.4 节介绍参数高效微调，可作为对照阅读；本课按学习目标采用全量方式。
5. 第 11.5 节：结合留出数据上的 loss、相同提示下的回答和人工评价。
   本课不自动运行 MMLU、GSM8K 或付费 LLM-as-Judge，这些是后续拓展方向。

原课程代码涉及不同时间的 API。本文件按 TRL 0.19.1 使用 max_length 和
processing_class，并显式设置 completion_only_loss。每个训练样本只监督
最后一个 assistant 答案；同一原始对话的所有轮次先放在同一个数据划分中，
再展开，避免把同一对话的不同前缀分散到训练集和留出集。
我们预先渲染并编码模板，便于查看 loss 掩码，也避免对 Qwen3 的 thinking
模式依赖隐式默认值。输入仍然是完整上下文，被掩掉的 prompt 仍参与注意力计算。

五、官方参考资料（编写时核对：2026-10-03）
=========================================
课程：https://huggingface.co/learn/llm-course/en/chapter11/1 （以及 /2 至 /6）
模型：https://huggingface.co/Qwen/Qwen3-0.6B
数据：https://huggingface.co/datasets/HuggingFaceTB/smoltalk
SFT：https://huggingface.co/docs/trl/v0.19.1/en/sft_trainer
模板：https://huggingface.co/docs/transformers/v4.52.4/en/chat_templating
Trainer：https://huggingface.co/docs/transformers/v4.52.4/en/main_classes/trainer
接口源码：https://github.com/huggingface/trl/tree/v0.19.1/trl/trainer
"""

import argparse
import gc
import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


# PROJECT_DIR 是项目根目录；所有相对输出路径以这里为起点。
PROJECT_DIR = Path(__file__).resolve().parent.parent
# MODEL_ID、MODEL_REVISION 指定模型及核对过的提交，避免 main 更新改变实验。
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
# DATASET_ID、DATASET_CONFIG、DATASET_REVISION 指定数据来源、子集和提交。
DATASET_ID = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "everyday-conversations"
DATASET_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
# SEED 固定抽样、划分与初始化；不同设备和计算内核仍可能有细微数值差异。
SEED = 42
# API_VERSIONS 只用于提示接口参考版本，不触发任何安装或升级。
API_VERSIONS = {
    "transformers": "4.52.4", "trl": "0.19.1",
    "accelerate": "1.7.0",
}


def check_environment():
    """只读检查 MLLearn、依赖版本与 GPU；返回是否具备继续运行的基本条件。"""
    # ready 记录是否缺少必需依赖；sys.prefix 是当前 Python 所属环境的目录。
    ready = Path(sys.prefix).name.lower() == "mllearn"
    print("Python：", sys.version.split()[0], "；解释器：", sys.executable)
    if not ready:
        print("请使用 conda run --name MLLearn 运行；本文件不会切换环境。")
    # package_name 是发行包名称；installed_version 是本环境实际安装的版本。
    for package_name in ("torch", "datasets", *API_VERSIONS):
        try:
            installed_version = version(package_name)
            print(f"{package_name}: {installed_version}")
            if package_name in API_VERSIONS and installed_version != API_VERSIONS[package_name]:
                print(f"  与本课接口参考 {API_VERSIONS[package_name]} 不同，请核对兼容性。")
        except PackageNotFoundError:
            ready = False
            print(f"{package_name}: 缺失；须先按 AGENTS.md 确认依赖准备方案。")
    # 仅在正确环境中读取 GPU 状态；这里不分配模型、不进行训练。
    if Path(sys.prefix).name.lower() == "mllearn":
        try:
            import torch
            print("PyTorch CUDA Runtime：", torch.version.cuda)
            if torch.cuda.is_available():
                print("GPU：", torch.cuda.get_device_name(0))
                print("总显存（GiB）：", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
            else:
                print("CUDA 不可用；显式选择 CPU 可以学习流程，但训练会很慢。")
        except ImportError as error:
            ready = False
            print("PyTorch 导入失败：", error)
    return ready


def clean_messages(messages):
    """清洗一条原始对话。messages 应为 role/content 字典列表；不合格时返回 None。"""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    # cleaned 保存清洗后的纯文本对话；expected_role 检查 user/assistant 交替。
    cleaned = []
    expected_role = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        # role 表示说话方；content 是去掉首尾空白的正文，正文内部格式保持原样。
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        content = content.strip()
        # 开头可以有 system；本课仅处理文本对话，不处理工具调用或多模态字段。
        if not (index == 0 and role == "system"):
            if role != expected_role:
                return None
            expected_role = "assistant" if role == "user" else "user"
        # 本课不监督思考或控制标记；若换数据，要根据任务重新制定清洗规则。
        if any(marker in content for marker in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")):
            return None
        cleaned.append({"role": role, "content": content})
    # 有些真实对话以尚未获得答案的 user 消息结尾；去掉这一条，保留前面的完整问答。
    # 只剩 system 或没有任何 assistant 答案的记录，无法作为本课的监督数据。
    if cleaned[-1]["role"] == "user":
        cleaned.pop()
    return cleaned if len(cleaned) >= 2 and cleaned[-1]["role"] == "assistant" else None


def make_sft_examples(conversations, tokenizer, max_length):
    """把已划分的原始对话展开为逐轮 SFT 样本；返回样本列表和超长样本数。

    conversations：每条记录含 conversation_id、messages。
    tokenizer：Qwen3 配套分词器；max_length：完整上下文加答案的 token 上限。
    每条 assistant 消息形成一个目标，前面的历史消息形成该目标的 prompt。
    """
    # examples 保存编码后的样本；too_long 统计整条舍弃的超长问答，避免截掉答案。
    examples, too_long = [], 0
    for conversation in conversations:
        # messages 是完整原始对话；turn_index 定位本次要监督的 assistant 消息。
        messages = conversation["messages"]
        for turn_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            # prompt_messages 是已知上下文，不含这轮参考答案。
            prompt_messages = messages[:turn_index]
            # prompt_text 保留 assistant 起始标记，且关闭 thinking，供训练和生成共用。
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            # full_text 含示范答案和结束标记；完整训练对话不追加新的 assistant 开头。
            full_text = tokenizer.apply_chat_template(
                messages[:turn_index + 1], tokenize=False,
                add_generation_prompt=False, enable_thinking=False,
            )
            # 模板已包含特殊 token；再次编码时不重复添加 BOS/EOS。
            # prompt_ids、input_ids 分别保存上下文和完整问答的词表 ID。
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if input_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("模板或 token 边界不一致：prompt 必须是完整问答的前缀。")
            if len(input_ids) > max_length:
                too_long += 1
                continue
            if len(input_ids) <= len(prompt_ids):
                raise ValueError("答案没有可监督的 token，请检查数据和模板。")
            # completion_mask 的 0 表示只作为上下文，1 表示该位置参与答案监督。
            # 历史 assistant 也被掩掉，但会在它自己对应的另一条样本中接受监督。
            completion_mask = [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))
            examples.append({
                "conversation_id": conversation["conversation_id"],
                "turn_index": turn_index,
                "prompt_messages": prompt_messages,
                "reference": message["content"],
                "input_ids": input_ids,
                "completion_mask": completion_mask,
            })
    return examples, too_long


def generate_reply(model, tokenizer, messages, max_new_tokens=128, use_bf16=False):
    """给定同样格式的上下文生成答案；只返回新增文本，便于前后对比。

    model 是训练前或全量微调后的完整模型；messages 不包含待生成的答案。
    max_new_tokens 控制新增 token 的数量，不包括输入长度。
    use_bf16 控制生成时的混合精度，使训练前、训练后和独立重载的计算设置一致。
    """
    import torch

    # inputs 是 Qwen3 模板编码后的张量；逐条生成，不涉及批量生成的 padding 方向。
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors="pt",
    ).to(model.device)
    model.eval()  # 关闭 dropout，防止前后比较混入训练时的随机扰动。
    # Trainer 可能已为模型包装混合精度 forward；重载后的普通模型也显式使用同样设置。
    # 只保持 FP32 权重相同还不够，比较回答时也要保持实际计算精度相同。
    with torch.inference_mode(), torch.autocast(
        device_type=model.device.type, dtype=torch.bfloat16, enabled=use_bf16,
    ):
        # output_ids 包含原输入与新增 token；do_sample=False 使用贪心解码。
        # 此处为了便于重复比较采用贪心；实际产品可另行评价采样参数。
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()


def main():
    # parser 定义少量常用参数；其余训练配置放在下面逐项讲解，避免隐藏流程。
    parser = argparse.ArgumentParser(description="Qwen3-0.6B 的单文件全量 SFT 学习教程")
    parser.add_argument("--stage", choices=("check", "train", "infer"), default="check",
                        help="check 只检查环境；train 执行全量微调流程；infer 加载已保存的完整模型")
    parser.add_argument("--sample-limit", type=int, default=1024, help="最多读取多少条原始对话")
    parser.add_argument("--max-length", type=int, default=512, help="上下文与答案合计的 token 上限")
    parser.add_argument("--max-steps", type=int, default=-1, help="正数限制更新步数；-1 按 1 epoch 训练")
    parser.add_argument("--output-dir", default="checkpoints/LLM/Qwen3_SFT_Full", help="实验产物目录")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="计算设备")
    parser.add_argument("--prompt", default="How can I start learning Python?", help="infer 阶段的问题")
    # args 保存命令行参数；check 阶段不需要下面的 Hugging Face 库。
    args = parser.parse_args()
    if args.sample_limit < 30 or args.max_length < 64 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("sample-limit 至少 30，max-length 至少 64，max-steps 为 -1 或正整数。")
    if args.stage == "check":
        check_environment()
        print("\n检查结束。阅读本文件开头的环境说明，再使用 --stage train 或 --stage infer。")
        return
    if not check_environment():
        raise SystemExit("环境检查未通过。请先按项目约定处理缺失依赖；没有安装任何依赖。")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("本课只演示单进程、单设备；请使用文档中的 python 命令启动。")

    import torch
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    # ======================================================================
    # 第 1 步：确定设备、精度、随机种子与输出位置
    # ======================================================================
    set_seed(SEED)
    # device 是实际计算设备；use_bf16 控制 Trainer 的混合精度；model_dtype 是权重精度。
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但 PyTorch 无法使用 GPU。")
    # 排除软件模拟 BF16：旧 GPU 即使能分配 BF16 张量，也不代表有原生 BF16 计算支持。
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    model_dtype = torch.float32  # 全量训练保留 FP32 参数，使梯度和 AdamW 状态也采用 FP32。
    print("计算设备：", device, "；权重存储精度：", model_dtype)
    if args.stage == "train":
        print("训练计算精度：", "BF16 混合精度" if use_bf16 else "FP32")
        if device == "cpu" or not use_bf16:
            print("全量 FP32 训练开销较大；CPU 很慢，本机 4GB GPU 不适合，建议使用服务器。")
    # output_dir 保存实验产物；model_dir 保存完整微调模型、模型配置和分词器。
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir = output_dir.resolve()
    model_dir = output_dir / "final_model"

    # ======================================================================
    # 第 2 步：独立推理入口，展示保存后的模型如何再次使用
    # ======================================================================
    if args.stage == "infer":
        if not (model_dir / "config.json").is_file():
            raise SystemExit(f"未找到完整模型：{model_dir}。请先全量训练或指定已有实验目录。")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        # inference_model 直接读取保存的完整权重，无需再次从 Hub 下载原始模型权重。
        inference_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=model_dtype, attn_implementation="sdpa",
        ).to(device)
        print("\n模型回答：", generate_reply(inference_model, tokenizer, [
            {"role": "user", "content": args.prompt},
        ], use_bf16=use_bf16))
        return

    # 新实验保留独立目录；运行 infer 不受这个目录保护检查的限制。
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("输出目录非空。请改用新的 --output-dir，保留已有实验结果。")

    # ======================================================================
    # 第 3 步：读取数据、检查格式、清洗、去重，先划分原始对话
    # ======================================================================
    # tokenizer 是 Qwen3 原配分词器；它保存词表、特殊 token 和聊天模板。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"  # 训练在右侧补齐；本课推理逐条进行。
    if not tokenizer.chat_template or tokenizer.eos_token != "<|im_end|>":
        raise ValueError("未取得预期的 Qwen3 对话模板或 EOS，请检查模型版本。")
    # EOS 是回复结束标记；PAD 是批内填充。保留 Qwen3 自带的不同 token。
    print("EOS：", tokenizer.eos_token, "；PAD：", tokenizer.pad_token)

    # stream 是流式数据集；避免先把课程的完整大数据集全部下载再取少量样本。
    # buffer_size 是近似打乱缓冲区大小，不是训练 batch size；不是全数据随机排列。
    stream = load_dataset(
        DATASET_ID, DATASET_CONFIG, split="train", streaming=True,
        revision=DATASET_REVISION,
    ).shuffle(seed=SEED, buffer_size=1000)
    # cleaned_rows 保存有效原始对话；seen 保存完整对话文本用于精确去重。
    # rejected、duplicate_count 分别统计格式不合格和完全重复的对话数量。
    cleaned_rows, seen = [], set()
    rejected, duplicate_count = 0, 0
    for conversation_id, row in enumerate(stream.take(args.sample_limit)):
        # messages 是本条记录的清洗结果；key 是用于比较重复对话的稳定字符串。
        messages = clean_messages(row.get("messages"))
        if messages is None:
            rejected += 1
            continue
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        cleaned_rows.append({"conversation_id": conversation_id, "messages": messages})
    if len(cleaned_rows) < 30:
        raise ValueError("清洗后不足 30 条对话，请增加 sample-limit 或检查数据。")
    print(f"有效对话 {len(cleaned_rows)}；格式过滤 {rejected}；精确去重 {duplicate_count}")
    print("第一条清洗后的对话：", cleaned_rows[0]["messages"])

    # raw_dataset 保存去重后的对话；first_split 留出约 20%，second_split 将其各分一半。
    # train 用于更新权重，validation 用于选择 checkpoint，test 只用于最终效果对比。
    raw_dataset = Dataset.from_list(cleaned_rows)
    first_split = raw_dataset.train_test_split(test_size=0.2, seed=SEED)
    second_split = first_split["test"].train_test_split(test_size=0.5, seed=SEED)
    # conversation_splits 约为 80% / 10% / 10%，取整后比例会略有差异。
    conversation_splits = {
        "train": first_split["train"],
        "validation": second_split["train"],
        "test": second_split["test"],
    }
    # 本课自建留出集，未使用数据源的官方 test；它不能充当公开基准的正式分数。
    # 精确去重不能解决近似重复；正式实验还应按用户、文档来源等检查数据泄漏。

    # ======================================================================
    # 第 4 步：将每轮答案变成可监督的 token，并检查长度与 loss 掩码
    # ======================================================================
    # prepared 保留问答及其编码用于查看/生成；tokenized_splits 只保留 Trainer 所需列。
    prepared, tokenized_splits = {}, {}
    for split_name, conversations in conversation_splits.items():
        # examples 是此划分逐轮展开的样本；too_long 是被完整舍弃的超长样本数。
        examples, too_long = make_sft_examples(conversations, tokenizer, args.max_length)
        if not examples:
            raise ValueError(f"{split_name} 没有有效问答；请增大 max-length 或 sample-limit。")
        prepared[split_name] = Dataset.from_list(examples)
        tokenized_splits[split_name] = prepared[split_name].select_columns(["input_ids", "completion_mask"])
        print(f"{split_name}: {len(conversations)} 条对话 → {len(examples)} 条问答；超长过滤 {too_long}")
    # sample 展示首个样本；prompt_length 是 loss 不监督的上下文长度。
    sample = prepared["train"][0]
    prompt_length = sample["completion_mask"].index(1)
    print("\n模板渲染后的完整训练文本：\n", tokenizer.decode(sample["input_ids"]))
    print("上下文 token 数：", prompt_length, "；监督 token 数：", len(sample["input_ids"]) - prompt_length)

    # 因果语言模型用第 t 个位置的输出预测第 t+1 个 token：模型内部完成 label shift。
    # collator 先复制 input_ids 为 labels，再把 prompt 和 PAD 位置改为 -100。
    # -100 是交叉熵的 ignore_index；attention_mask=0 则表示 PAD，不是同一用途。
    # 示例：input_ids=[问题..., 答案..., EOS]；labels=[-100..., 答案..., EOS]。
    # 本课不手工 shift，也不把 attention_mask 中的 prompt 改为 0。

    # 只有完成基本数据检查之后才创建产物目录，保存划分便于追溯与复查泄漏。
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_splits.json").write_text(json.dumps(
        {name: split.to_list() for name, split in conversation_splits.items()},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")

    # ======================================================================
    # 第 5 步：加载完整模型，明确开启全部参数的梯度
    # ======================================================================
    # model 是预测下一个 token 的因果语言模型，不是 BERT 句子分类模型。
    # 单设备训练直接放到 device；device_map="auto" 是推理分配方案，不用于本课训练。
    # sdpa 使用 PyTorch 自带注意力实现，不需要另装 FlashAttention。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=model_dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False  # 训练不使用生成时的 KV cache，配合梯度检查点节省显存。

    # requires_grad_(True) 对模型的每个参数开启梯度；优化器会更新全部可训练参数。
    # from_pretrained 通常已开启梯度，这里显式设置，方便新手确认全量微调的含义。
    model.requires_grad_(True)
    # 全量更新可能更容易改变原有能力；采用较小学习率，并结合留出集观察遗忘与过拟合。

    # ======================================================================
    # 第 6 步：配置 SFTTrainer，查看真实 batch 和可训练参数
    # ======================================================================
    # training_args 保存训练、评估、checkpoint 与日志策略，全部显式列在这里。
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,             # 默认只遍历一轮问答，先观察再决定是否加轮数。
        max_steps=args.max_steps,        # 正数优先于 epochs，表示优化器更新次数。
        per_device_train_batch_size=1,  # 每次前向的一条问答，便于控制显存。
        per_device_eval_batch_size=1,   # 验证同样逐条处理。
        gradient_accumulation_steps=8,  # 累积 8 次再更新；单设备有效 batch 约为 8。
        learning_rate=2e-5,             # 全量微调采用较小的起始学习率，需据 validation 调整。
        lr_scheduler_type="cosine",    # warmup 后逐渐衰减，避免一直大步更新。
        warmup_ratio=0.1,               # 总更新步数的约 10% 用于预热；两步试跑无代表性。
        weight_decay=0.01,              # AdamW 的权重衰减；帮助约束权重更新。
        max_grad_norm=1.0,              # 梯度裁剪，限制异常大的更新。
        optim="adamw_torch",           # PyTorch AdamW，不需要量化优化器依赖。
        bf16=use_bf16, fp16=False,      # BF16 只用于混合精度计算；权重保持 FP32。
        use_cpu=device == "cpu",       # 显式选择 CPU 时，Trainer 也使用 CPU。
        gradient_checkpointing=True,    # 少存部分激活，反向时重算，节省显存但增加耗时。
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,     # 本课已过滤超长样本，不在这里再次截断答案。
        packing=False,                 # 不把独立问答拼接成一条，先看清样本边界。
        completion_only_loss=True,      # 使用我们提供的 completion_mask，监督目标答案。
        assistant_only_loss=False,      # 不要求模板提供 {% generation %} 掩码。
        dataset_kwargs={"skip_prepare_dataset": True},  # 已手工编码，避免重复预处理。
        eos_token=tokenizer.eos_token,   # Qwen3 回复结束于 <|im_end|>。
        eval_strategy="epoch",          # 每轮后在 validation 上评估。
        save_strategy="epoch",          # 与评估时间点一致，支持最佳 checkpoint 选择。
        load_best_model_at_end=True,     # 训练结束加载 validation loss 最低的一次保存。
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2,             # 限制 checkpoint 数量，节省磁盘。
        save_safetensors=True,          # 完整模型权重使用 safetensors 格式保存。
        logging_steps=1,                # 小实验逐个更新步记录 loss / 学习率 / 梯度等。
        report_to="none",               # 只用本地日志，不要求登录外部追踪服务。
        push_to_hub=False,              # 本课保存到本地；发布前还须审阅结果和许可。
        dataloader_num_workers=0,       # Windows 单文件教学避免额外 DataLoader 进程。
        seed=SEED, data_seed=SEED,
    )
    # trainer 负责动态 padding、loss、反向传播、优化器更新、评估与 checkpoint。
    # 直接传入完整模型，Trainer 默认收集所有 requires_grad=True 的参数建立优化器。
    trainer = SFTTrainer(
        model=model, args=training_args, processing_class=tokenizer,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
    )
    # total_parameters、trainable_parameters 分别统计总参数量和实际开启梯度的参数量。
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainer.model.parameters()
                               if parameter.requires_grad)
    print(f"总参数：{total_parameters:,}；可训练参数：{trainable_parameters:,}；"
          f"比例：{100 * trainable_parameters / total_parameters:.2f}%")
    if trainable_parameters != total_parameters:
        raise ValueError("发现冻结参数，不符合本课全量微调的目标，请检查模型设置。")
    # batch 调用实际 collator；它把可变长度样本补齐并生成 labels 和 attention_mask。
    batch = trainer.data_collator([tokenized_splits["train"][0]])
    print("真实 batch 形状：", {name: tuple(value.shape) for name, value in batch.items()})
    print("前 30 个 labels（-100 表示不计 loss）：", batch["labels"][0, :30].tolist())
    if not (batch["labels"][0, :prompt_length] == -100).all().item():
        raise ValueError("prompt 没有正确排除在 loss 之外。")
    if (batch["labels"][0, prompt_length:] == -100).any().item():
        raise ValueError("目标答案被错误掩掉，请检查 collator。")
    del batch

    # ======================================================================
    # 第 7 步：建立基线，用相同留出样本测量原模型
    # ======================================================================
    # comparison_examples 从 test 中挑最多 3 条实质回答，供固定提示的生成比较。
    # 参考答案用于人工评分，绝不作为生成输入；这批回答不能用于选择超参数。
    comparison_examples = [example for example in prepared["test"]
                           if len(example["reference"]) >= 40][:3]
    if not comparison_examples:
        comparison_examples = [prepared["test"][0]]
    # comparisons 保存同一提示的参考答案、原模型回答和微调后回答。
    comparisons = []
    # 尚未调用 train()，此时权重就是刚加载的原始模型；evaluate 不会更新它们。
    before_validation = trainer.evaluate(metric_key_prefix="before_validation")
    before_test = trainer.evaluate(eval_dataset=tokenized_splits["test"], metric_key_prefix="before_test")
    for example in comparison_examples:
        comparisons.append({
            "prompt_messages": example["prompt_messages"],
            "reference": example["reference"],
            "before": generate_reply(trainer.model, tokenizer, example["prompt_messages"], use_bf16=use_bf16),
        })
    print("训练前 validation：", before_validation)
    print("训练前 test：", before_test)

    # ======================================================================
    # 第 8 步：正式训练，理解 Trainer 帮我们执行了什么
    # ======================================================================
    # 一次更新的大致过程：取 batch → 前向预测与交叉熵 → backward → 累积梯度 →
    # clip_grad_norm → optimizer.step → scheduler.step → 清空梯度。
    # Trainer 在 train() 中重新切换到训练模式；本课原模型全部参数都参与更新。
    # train_result 保存本轮的耗时、步数、平均训练 loss 等真实结果，不预设效果。
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()  # 保存 trainer_state.json，包含 global_step 与 log_history。

    # 观察训练日志时，将 train loss 与 validation loss 一起看。
    # train loss 降而 validation loss 持续升，可能过拟合，应检查数据/轮数/学习率。
    # 本课只训练一轮，无法仅凭一个 validation 点判断完整趋势；需多轮实验观察。
    # 学习率与 grad_norm 也有助于发现不稳定更新；loss 低不等于回答一定正确。
    # 中断续训可用 trainer.train(resume_from_checkpoint=".../checkpoint-步数")。
    # 必须保留同样的数据与配置；final_model/ 只保存推理所需文件，不含优化器状态。

    # ======================================================================
    # 第 9 步：评估最佳 validation checkpoint，检查能力是否真有变化
    # ======================================================================
    # after_validation、after_test 使用和基线相同的样本、模板、掩码与 batch size。
    # test 只作前后报告；不要根据它反复调参，否则它就变成了 validation。
    after_validation = trainer.evaluate(metric_key_prefix="after_validation")
    after_test = trainer.evaluate(eval_dataset=tokenized_splits["test"], metric_key_prefix="after_test")
    for comparison in comparisons:
        comparison["after"] = generate_reply(
            trainer.model, tokenizer, comparison["prompt_messages"], use_bf16=use_bf16,
        )
        print("\n提示上下文：", comparison["prompt_messages"])
        print("参考答案：", comparison["reference"])
        print("微调前：", comparison["before"])
        print("微调后：", comparison["after"])

    # report 汇总 loss 与训练结果；exp(loss) 是本课评估 loss 的困惑度近似展示。
    # Trainer 的样本/批次聚合不等于严格按全数据有效 token 加权，故不声称精确 PPL。
    # 该指标只覆盖目标答案，且仅在相同 tokenizer / 数据 / 掩码下比较有意义。
    report = {
        "before_validation": before_validation, "before_test": before_test,
        "after_validation": after_validation, "after_test": after_test,
        "train": train_result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    for phase, metrics in (("before_test", before_test), ("after_test", after_test)):
        # loss 是此阶段 test 平均损失；数值很大时避免 exp 溢出。
        loss = metrics[f"{phase}_loss"]
        report[f"{phase}_approx_perplexity"] = math.exp(loss) if math.isfinite(loss) and loss < 700 else None
    print("\n训练后 validation：", after_validation)
    print("训练后 test：", after_test)
    # 人工复查：回答是否相关、事实是否正确、是否遵循要求、是否复读/遗忘通用能力。
    # 对话通常有多种正确答案，不宜简单把与参考文本完全一致当作质量标准。
    # 正式模型还应增加任务专项测试、通用能力回归与更大、独立的数据集。

    # ======================================================================
    # 第 10 步：保存完整模型、分词器、配置和结果，再从磁盘重载验证
    # ======================================================================
    # 恢复推理使用的 KV cache 配置；之后保存完整微调权重与模型 config.json。
    trainer.model.config.use_cache = True
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))  # 保留词表、特殊 token 和对话模板。
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "comparisons.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    # run_config 是本次实际配置记录；训练参数也由 Trainer 保存在产物中。
    run_config = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION, "seed": SEED,
        "arguments": vars(args), "device": device, "weight_dtype": str(model_dtype),
        "fine_tuning_method": "full", "mixed_precision": "bf16" if use_bf16 else "fp32",
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "versions": {name: version(name) for name in ("torch", "datasets", *API_VERSIONS)},
        "conversation_counts": {name: len(split) for name, split in conversation_splits.items()},
        "example_counts": {name: len(split) for name, split in tokenized_splits.items()},
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 先释放旧 Trainer、模型与优化器，再加载新的模型，避免两个模型同时占显存。
    del trainer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()  # 只释放缓存；仍被引用的张量不会由这个调用释放。
    # reloaded_tokenizer、reloaded_model 从本地恢复；FP32 权重和混合精度计算设置保持一致。
    reloaded_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=model_dtype, attn_implementation="sdpa",
    ).to(device)
    # reloaded_reply 是和首条 after 回答相同提示下的重载结果；通常应一致。
    reloaded_reply = generate_reply(
        reloaded_model, reloaded_tokenizer, comparisons[0]["prompt_messages"], use_bf16=use_bf16,
    )
    print("\n重载后的回答：", reloaded_reply)
    print("是否与保存前的贪心回答一致：", reloaded_reply == comparisons[0]["after"])
    if reloaded_reply != comparisons[0]["after"]:
        print("请核对精度、模板和保存的完整权重；也可能存在设备计算差异。")
    print("实验产物目录：", output_dir)

    # final_model/ 包含 config.json、完整模型 safetensors 权重及分词器相关文件。
    # 较大模型可能保存为多个权重分片和索引，from_pretrained 会自动读取这些文件。
    # checkpoint-*/ 通常还包括优化器、scheduler、随机数状态，用于中断续训。
    # metrics.json / comparisons.json 记录前后对比；data_splits.json 保存实际划分。
    # 交付模型时应带上整个 final_model/，并说明任务、数据来源、评估结果与依赖版本。
    # 原始模型 ID/revision 仍保存在 run_config.json 中，用于追溯训练起点。


# Windows 下保留主入口保护；导入本文件不会下载数据或启动训练。
if __name__ == "__main__":
    main()

# 常见问题与后续练习
# ------------------
# 1. CUDA out of memory：先减小 max-length / batch size，保持梯度累积，并考虑服务器。
#    显存还与词表 logits、激活和精度有关，不能仅用“参数量 × 每参数字节数”估算。
#    梯度累积只减少每次前向的 batch，不减少全部参数的梯度和优化器状态。
# 2. 下载失败：核对网络、缓存空间和代理；不要因此更换环境或取消 revision 固定。
# 3. 回复不停：核对 <|im_end|> 和 EOS；max_new_tokens 是兜底长度，不是结束语义。
# 4. 微调后变差：检查目标和数据，比较相同提示，减少轮数/学习率，排查数据泄漏。
# 5. 改成中文领域任务：替换数据读取部分，保留 messages 格式并重新制定清洗/评分规则。
#    准备经过人工复核的示范答案；英文日常对话分数不能说明中文领域效果。
# 6. 对照实验：一次只改变 learning_rate、max-length 或训练轮数；保留独立目录和相同划分。
#    sample-limit 很小时，方差会很大；先扩大有效数据，再解释细小的分数变化。
