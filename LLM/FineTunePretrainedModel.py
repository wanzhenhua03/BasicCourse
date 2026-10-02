"""Hugging Face 第三章：微调预训练模型（按三部分教程顺序整理）。

内容以用户粘贴的官方教程为准，原文链接如下：
1. 处理数据：https://huggingface.co/learn/llm-course/zh-CN/chapter3/2
2. Trainer：https://huggingface.co/learn/llm-course/zh-CN/chapter3/3
3. 完整训练：https://huggingface.co/learn/llm-course/zh-CN/chapter3/4

运行环境与执行方式
------------------
使用项目已有的 Conda 环境 MLLearn，目标硬件为单张 NVIDIA RTX 4090。
Python 3.9、PyTorch 2.7.1+cu118；Hugging Face API 按 Transformers 4.46.3、
Accelerate 0.34.2 编写，另需 datasets、evaluate、numpy、tqdm、scikit-learn。
上述两个版本是接口参考，并非已经在本机安装或验证的依赖组合。
当前笔记本的 MLLearn 尚缺 transformers、evaluate、accelerate，不能直接完成训练。
依赖准备必须遵守 AGENTS.md，先获许可；本文件不会安装或升级任何依赖。

在项目根目录运行（Windows PowerShell 或 Linux shell 均可）：
    conda run --no-capture-output --name MLLearn python -X utf8 LLM/FineTunePretrainedModel.py

-X utf8 让中文输出使用 UTF-8；--no-capture-output 避免 Conda 捕获后重新编码。

首次运行会下载 BERT、MRPC 数据和 Evaluate 指标；需要访问 Hugging Face。
默认缓存遵循 Hugging Face 设置，可在运行前通过 HF_HOME 指定缓存目录。
Trainer 输出位于项目 checkpoints/LLM/MRPC/，不同教学实验使用不同子目录。
全脚本依次执行：两句话的一次更新、无训练中评估的 Trainer（3 epoch）、
带每轮评估的全新 Trainer（3 epoch）、手写循环（3 epoch）、Accelerate（3 epoch）。
完整保留 batch size 8；默认 FP32，RTX 4090 不要求改成混合精度才能运行本课。

独立脚本所需的衔接与原教程的区别
------------------------------
* Notebook 中单独写出的表达式改为 print，让 .py 文件也能看到教学输出。
* 各节重复的预处理摘要复用第一节结果；原教程的两个函数保持展开，不添加训练封装。
* 新版 Trainer 使用 eval_strategy、processing_class，分别对应原文的
  evaluation_strategy、tokenizer。接口依据：
  https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer
* datasets 新版按列读取可能返回 Column；在直接调用 Tokenizer 时转为 list。
* 独立训练实验重新加载预训练模型，释放上一实验的模型、梯度图和优化器。
* 手写训练在构建优化器之前把模型放到 GPU，确保优化器引用目标设备上的参数。
* 原文 Accelerate 节重复的手写对照代码保留为注释；随后执行完整 Accelerate 代码。
* 练习与 Notebook/多卡启动方法保留为说明，不额外启动 SST-2 或分布式实验。

这里微调的是 BERT 的句子对分类模型，并非让生成式 LLM 预测下一个 token。
代码会展示真实运行结果，不把原教程的 loss、accuracy、F1 写成必然达到的数值。
"""

import gc
from importlib.util import find_spec
from pathlib import Path


if __name__ == "__main__":
    # 运行前检查所有教学阶段的依赖，避免训练到中途才发现后续库缺失。
    # required_packages 保存导入名称；sklearn 对应安装包 scikit-learn。
    required_packages = (
        "torch", "transformers", "datasets", "evaluate", "accelerate",
        "numpy", "tqdm", "sklearn",
    )
    # missing_packages 保存当前解释器中无法找到的依赖，检查不会修改环境。
    missing_packages = [name for name in required_packages if find_spec(name) is None]
    if missing_packages:
        raise SystemExit(
            "当前环境缺少依赖：" + ", ".join(missing_packages)
            + "。请先按 AGENTS.md 确认 MLLearn 中的依赖方案；脚本未安装任何依赖。"
        )

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # device 保存训练设备；本文件面向单张 RTX 4090，正式训练使用 CUDA。
    if not torch.cuda.is_available():
        raise SystemExit("本文件按单张 RTX 4090 编写，请在 CUDA 可用的 MLLearn 中运行。")
    device = torch.device("cuda:0")
    print("运行设备：", torch.cuda.get_device_name(device))

    # SCRIPT_DIR 定位本文件，PROJECT_DIR 定位项目；输出不随启动时的工作目录改变。
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    # OUTPUT_DIR 是两轮 Trainer 实验的公共目录，位于项目已忽略的 checkpoints 下。
    OUTPUT_DIR = PROJECT_DIR / "checkpoints" / "LLM" / "MRPC"

    # ========================================================================
    # 第一部分：处理数据
    # ========================================================================
    print("\n第一部分：处理数据")

    # 1.1 用两句话回顾一次完整的参数更新。
    # Tokenizer 将文本转换为张量；传入 labels 后，模型计算分类交叉熵 loss。
    # backward() 计算梯度，optimizer.step() 按梯度更新可训练参数。
    # 这只是流程演示：两句话和人为指定的标签不足以训练出可靠的分类器。
    # checkpoint 保存官方教程使用的预训练模型标识，之后各实验都从它重新开始。
    checkpoint = "bert-base-uncased"
    # tokenizer 是与 checkpoint 配套的分词器；model 是带序列分类头的 BERT。
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    # sequences 保存演示用的两条单句，尚不是后面 MRPC 的句子对。
    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This course is amazing!",
    ]
    # batch 保存模型输入；padding 补齐长度，truncation 截断，pt 返回 PyTorch 张量。
    batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    batch["labels"] = torch.tensor([1, 1])
    batch = {k: v.to(device) for k, v in batch.items()}
    # optimizer 保存 AdamW；此处保留原教程没有显式指定学习率的一步演示。
    optimizer = AdamW(model.parameters())
    # loss 是本批的分类损失，带有用于反向传播的计算图。
    loss = model(**batch).loss
    print("两句话的演示 loss：", loss.item())
    loss.backward()
    optimizer.step()

    # 拼接适配：下面的 MRPC 实验不沿用这一次更新后的模型。
    del model, optimizer, loss, batch
    gc.collect()
    torch.cuda.empty_cache()

    # 1.2 从 Hub 加载 GLUE 中的 MRPC。
    # MRPC 判断两个句子是否表达同一含义，适合演示句子对分类。
    # DatasetDict 含 train、validation、test；分别为 3668、408、1725 条。
    # Dataset 以 Apache Arrow 为基础，可以利用磁盘缓存，而不必复制全部数据到内存。
    from datasets import load_dataset

    # raw_datasets 保存未经 tokenization 的三个数据划分。
    raw_datasets = load_dataset("glue", "mrpc")
    print(raw_datasets)
    # raw_train_dataset 保存训练划分；每条记录含 sentence1、sentence2、label、idx。
    raw_train_dataset = raw_datasets["train"]
    print("训练集第 1 条：", raw_train_dataset[0])
    print("字段及标签类型：", raw_train_dataset.features)
    # label 已为整数：0 = not_equivalent（不同义），1 = equivalent（同义）。
    # 原教程练习：查看训练集第 15 行和验证集第 87 行的标签。
    # 若按第 1 行对应索引 0，可分别查看 raw_train_dataset[14]、
    # raw_datasets["validation"][86]；不要把序号和 Python 的零起始索引混淆。

    # 1.3 先分别编码两列，再说明为什么句子对需要联合编码。
    # 保留原教程重新加载 Tokenizer 的顺序；已经缓存的文件可以复用。
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenized_sentences_1、2 分别保存两列单独编码的结果。
    # list() 是对新版 datasets.Column 的适配，Tokenizer 接收的是字符串列表。
    tokenized_sentences_1 = tokenizer(list(raw_datasets["train"]["sentence1"]))
    tokenized_sentences_2 = tokenizer(list(raw_datasets["train"]["sentence2"]))
    print("分别编码时第一句的 token：",
          tokenizer.convert_ids_to_tokens(tokenized_sentences_1["input_ids"][0]))
    print("分别编码时第二句的 token：",
          tokenizer.convert_ids_to_tokens(tokenized_sentences_2["input_ids"][0]))

    # inputs 保存一对示例句子的联合编码，第二个位置参数是第二个句子。
    inputs = tokenizer("This is the first sentence.", "This is the second one.")
    print("句子对联合编码：", inputs)
    # input_ids 是词表 ID；attention_mask 标记有效输入与 padding。
    # token_type_ids 区分两句话，其形式是：
    # [CLS] sentence1 [SEP] sentence2 [SEP]
    # [CLS] sentence1 [SEP] 的 token_type_ids 为 0，sentence2 [SEP] 为 1。
    # 原教程练习：将训练集第 15 个元素的两句话分别编码，比较联合编码的差异。
    print("联合编码对应的 token：", tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
    print("句子所属片段：", inputs["token_type_ids"])
    # 不同架构不一定使用 token_type_ids，例如 DistilBERT 不返回该字段。
    # BERT 预训练除了掩码语言建模，还使用下一句预测（NSP）：
    # 一部分句子对在原文中相邻，另一部分随机配对，模型学习判断是否为下一句。
    # NSP 与 MRPC 的“语义是否相同”是不同任务，不能直接使用 NSP 头解决 MRPC。
    # 配套的 Tokenizer 会提供该架构实际需要的字段，无需自行补造 token_type_ids。

    # 1.4 演示一次性编码并 padding 整个训练集。
    # tokenized_dataset 是字典，各字段保存整列编码；padding=True 会按本次
    # 调用中的最长样本补齐，导致所有训练样本都补到整个训练集的最大长度。
    # 这是原教程的对照示例，后续正式训练改用 map() 与动态填充。
    tokenized_dataset = tokenizer(
        list(raw_datasets["train"]["sentence1"]),
        list(raw_datasets["train"]["sentence2"]),
        padding=True,
        truncation=True,
    )
    print("一次性填充后的前 8 条长度：",
          [len(x) for x in tokenized_dataset["input_ids"][:8]])

    # 1.5 用 Dataset.map() 进行批量预处理。
    # 保留教程函数：example 可以是一条记录，也可以是 batched=True 给出的批次字典。
    # 此时只截断，不 padding；将 padding 推迟到实际组成训练 batch 的时候。
    def tokenize_function(example):
        """对 MRPC 的 sentence1、sentence2 联合编码，保留可变序列长度。"""
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    # tokenized_datasets 保存 map() 的结果，原字段仍在，另增加模型输入字段。
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    print("map() 处理后的数据集：", tokenized_datasets)
    # batched=True 让 Rust 实现的 Fast Tokenizer 一次处理多个样本。
    # map() 可以用 num_proc 并行，但本例保留原教程设置，不增加额外进程。
    # 若返回字典含已有字段名，map() 会用返回值更新该列。

    # 1.6 动态填充：collate 函数在实际构建 batch 时补齐变长输入。
    # 默认 DataLoader collate 无法直接堆叠长度不同的序列。
    # DataCollatorWithPadding 使用 Tokenizer 的 pad token 与 padding 方向。
    # 动态填充只补到当前 batch 的最长样本，通常比全数据集统一填充更省计算。
    from transformers import DataCollatorWithPadding

    # data_collator 是后面 Trainer 和 DataLoader 共同使用的动态填充器。
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # samples 是前 8 条记录组成的字典；去掉字符串列和不参与预测的 idx。
    samples = tokenized_datasets["train"][:8]
    samples = {k: v for k, v in samples.items()
               if k not in ["idx", "sentence1", "sentence2"]}
    print("动态填充前的长度：", [len(x) for x in samples["input_ids"]])
    # batch 是动态填充后的张量字典；collator 自动将 label 转成 labels。
    batch = data_collator(samples)
    print("动态填充后的形状：", {k: v.shape for k, v in batch.items()})
    # 原教程前 8 条的最大长度为 67，因此输入张量示例形状为 [8, 67]。
    # TPU 倾向固定形状；本文件目标为 GPU，继续使用动态填充。

    # 拼接适配：释放仅用于讲解的整列填充结果，正式训练复用 tokenized_datasets。
    del tokenized_dataset, tokenized_sentences_1, tokenized_sentences_2, samples, batch

    # ========================================================================
    # 第二部分：使用 Trainer API 微调模型
    # ========================================================================
    print("\n第二部分：使用 Trainer API 微调模型")
    # 原教程此处回顾 load_dataset、AutoTokenizer、tokenize_function、map()、collator。
    # 这些对象已在第一部分按顺序构建，继续使用即可，无需再下载或重复定义函数。
    # Trainer 帮助处理设备放置、DataLoader、优化器、调度器及训练/评估循环。

    # 2.1 定义训练参数，再加载带有两个输出类别的新模型。
    from transformers import TrainingArguments

    # training_args 保存训练超参数；显式列出原教程的主要默认值便于学习。
    # 两轮 Trainer 输出分开保存；report_to="none" 避免自动启动额外日志服务。
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "test-trainer-basic"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        logging_steps=500,
        report_to="none",
    )
    # model 重新从 BERT checkpoint 加载；num_labels=2 指 MRPC 的两个类别。
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # 新分类头的权重随机初始化，因此必须微调；加载时的提示是预期行为。
    # 一些原预训练头的权重不会用于新任务，不代表读取 checkpoint 失败。
    # 原教程提到 push_to_hub=True 可以自动上传，本文件保持默认 False。

    from transformers import Trainer

    # trainer 将模型、参数、两个数据划分、动态填充器和 Tokenizer 组织起来。
    # 原文 tokenizer=tokenizer 在参考版本中已由 processing_class=tokenizer 接替。
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    # 传入 Tokenizer 时，Trainer 可默认创建 DataCollatorWithPadding；这里保留
    # 显式 data_collator，使第一部分的数据处理与当前训练代码可以对应起来。
    trainer.train()
    # 默认每 500 步报告 loss，但不自动在训练中评估：
    # 一是 eval_strategy 默认为 "no"，二是尚未传入 compute_metrics。

    # 2.2 用 predict() 获取验证集输出，再用 Evaluate 计算指标。
    # predictions 是 PredictionOutput，含 predictions、label_ids、metrics。
    predictions = trainer.predict(tokenized_datasets["validation"])
    print("验证 logits 与 labels 的形状：",
          predictions.predictions.shape, predictions.label_ids.shape)
    print("尚未传入 compute_metrics 时的指标：", predictions.metrics)
    # MRPC 验证集有 408 条，通常得到 [408, 2] logits 与 [408] labels。
    # 此时 metrics 含 loss、耗时等；预测类别应取每行 logits 最大值的索引。
    import numpy as np

    # preds 保存第一轮 Trainer 在验证集上预测的 0/1 类别。
    preds = np.argmax(predictions.predictions, axis=-1)

    import evaluate

    # metric 是 GLUE/MRPC 指标对象，报告 accuracy 与 F1。
    metric = evaluate.load("glue", "mrpc")
    print("第一轮 Trainer 的验证指标：",
          metric.compute(predictions=preds, references=predictions.label_ids))
    # 教程的 accuracy/F1 是一次实验结果；随机分类头、打乱及环境差异影响结果。
    # 本文件始终使用 bert-base-uncased，不沿用粘贴文本中关于 cased 模型的误述。

    # 保留教程函数：eval_preds 可解包为 logits 与 labels。
    def compute_metrics(eval_preds):
        """将验证 logits 转成类别索引，返回 MRPC 的 accuracy 和 F1。"""
        # metric 保存本次评估的指标对象；logits、labels 分别为预测分数和真实标签。
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        # predictions 保存每条样本 logits 最大值对应的类别索引。
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 拼接适配：结束第一轮独立实验，避免旧 Trainer 继续持有 GPU 模型和优化器。
    del trainer, model, training_args, predictions, preds, metric
    gc.collect()
    torch.cuda.empty_cache()

    # 2.3 使用新模型和新 Trainer，在每个 epoch 结束时评估。
    # 原文 evaluation_strategy="epoch" 在此写作 eval_strategy="epoch"。
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "test-trainer-evaluation"),
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        logging_steps=500,
        report_to="none",
    )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    # 新模型使本实验从预训练权重重新开始；沿用旧模型则是继续训练。
    trainer.train()
    # Trainer 支持多 GPU/TPU 和混合精度，原文以 fp16=True 举例。
    # 本课按单张 RTX 4090 的 FP32 基础流程执行，保持手写循环也容易逐行对照。
    # 原教程练习：改为 GLUE/SST-2；该任务只有 sentence 一列，因此应同时修改
    # 数据集配置和 tokenize_function，不能直接使用 MRPC 的两列处理函数。

    del trainer, model, training_args
    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================================
    # 第三部分：一个完整的训练过程
    # ========================================================================
    print("\n第三部分：不使用 Trainer 的完整训练")
    # 原教程再次回顾第一部分的预处理，本文件继续复用已有的 Dataset 与 collator。

    # 3.1 训练前准备：手动完成 Trainer 原本会帮我们处理的字段整理。
    # 删除模型不需要的字符串列和 idx；把 label 改为模型 forward 接受的 labels；
    # set_format("torch") 使取出的样本字段成为 PyTorch 张量。
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    print("手写循环使用的列：", tokenized_datasets["train"].column_names)

    from torch.utils.data import DataLoader

    # train_dataloader 打乱训练样本；eval_dataloader 顺序读取验证样本。
    # collate_fn=data_collator 在每次取 batch 时执行动态填充。
    # num_workers 沿用默认 0，避免给 Windows 单文件教学增加多进程要求。
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )
    # batch 保存第一个训练 batch；它的长度会随 shuffle=True 的抽样结果改变。
    for batch in train_dataloader:
        break
    print("手写循环首批形状：", {k: v.shape for k, v in batch.items()})

    # model 是手写实验的新 BERT 分类模型，不延续上一轮 Trainer 的训练权重。
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # 此处先做一次前向检查，保留教程从数据到模型的验证顺序，暂时在 CPU 上。
    # outputs 保存模型输出；labels 使输出同时含 loss 和 logits。
    outputs = model(**batch)
    print("前向检查：", outputs.loss, outputs.logits.shape)
    # 8 条记录、2 个类别对应 [8, 2] logits；loss 的具体数值不是固定答案。
    # 释放这次前向图后，提前完成教程稍后介绍的设备移动，再构建优化器。
    del outputs
    model.to(device)

    # optimizer 使用 AdamW，学习率保持原教程的 5e-5。
    # AdamW 将权重衰减与梯度更新解耦；名称相同不代表全部超参数完全相同：
    # 此处 PyTorch AdamW 的默认 weight_decay 与 Trainer 默认值不同，
    # Trainer 还有梯度裁剪等处理，因此本手写示例不是数值逐步等价的 Trainer 复刻。
    optimizer = AdamW(model.parameters(), lr=5e-5)

    from transformers import get_scheduler

    # num_epochs 保存遍历全训练集的次数；num_training_steps 是优化器更新总步数。
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler 是无 warmup 的线性调度器，将学习率从 5e-5 逐步降至 0。
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print("手写训练总步数：", num_training_steps)
    # MRPC 3668 条、batch size 8：每轮 ceil(3668 / 8) = 459 步，3 轮为 1377 步。

    # 3.2 训练循环。
    # 原教程在此定义 device 并调用 model.to(device)；本文件为优化器参数引用
    # 的正确性，已提前移动模型。每个 batch 仍需在循环内移动到同一 GPU。
    # 原文可回退 CPU：torch.device("cuda") if torch.cuda.is_available() else
    # torch.device("cpu")。本文件明确以 RTX 4090 为运行目标，不自动回退慢速训练。
    print("手写训练设备：", device)

    from tqdm.auto import tqdm

    # progress_bar 保存跨所有 epoch 的进度条；epoch 是当前轮次索引。
    progress_bar = tqdm(range(num_training_steps), desc="手写训练")
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch 是当前批次字典，k 为字段名，v 为张量；全部移动到 CUDA。
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    progress_bar.close()
    # 更新顺序：前向 → loss → 反向 → 参数更新 → 调度器更新 → 清空梯度。
    # PyTorch 默认累积梯度，因此每步更新后必须 zero_grad()。
    # 本循环保持原教程设置，不在训练过程中插入验证循环。

    # 3.3 评估循环：切换 eval()，并使用 no_grad() 禁用梯度记录。
    # eval() 控制 dropout 等层的行为，no_grad() 控制自动微分，两者作用不同。
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        # logits 是本批类别分数；predictions 是每条记录的预测类别索引。
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # add_batch() 累积每一批的预测与真实标签，最后统一计算完整验证集指标。
        metric.add_batch(predictions=predictions, references=batch["labels"])
    print("手写循环验证指标：", metric.compute())
    # 原教程练习：将这个完整循环改为 SST-2，同时调整数据集和单句 tokenization。

    # 释放手写训练遗留的 optimizer 状态及最后一个 loss/outputs 对模型计算图的引用。
    del model, optimizer, lr_scheduler, outputs, loss, batch, logits, predictions, metric
    gc.collect()
    torch.cuda.empty_cache()

    # 3.4 使用 Accelerate 改写训练循环。
    # 原文首先重复展示未使用 Accelerate 的完整循环；下面保留该对照代码，
    # 其步骤已在 3.1～3.3 执行，这里不再启动第五轮相同的手写训练。
    # 注意：此对照及随后 Accelerate 示例都使用原文的 lr=3e-5，
    # 而上一段完整手写示例是 lr=5e-5，不能把二者的指标差异仅归因于 Accelerate。
    #
    # model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # optimizer = AdamW(model.parameters(), lr=3e-5)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)
    # num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    # progress_bar = tqdm(range(num_training_steps))
    # model.train()
    # for epoch in range(num_epochs):
    #     for batch in train_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)
    #
    # 原文以 +/- 展示改动，它们是 diff 标记，不是可以直接执行的 Python。
    # 改动依次是：导入 Accelerator；实例化；prepare 数据加载器、模型、优化器；
    # 删除手动 device 移动；用 accelerator.backward(loss) 替代 loss.backward()。
    # Accelerator 会识别设备与分布式状态；单张 4090 也可以使用相同接口。
    from accelerate import Accelerator

    # accelerator 保存当前训练的设备管理器；默认不额外开启混合精度。
    accelerator = Accelerator()
    # model、optimizer 是本节全新的模型与优化器，学习率按原文为 3e-5。
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # 拼接适配：创建本节自己的 DataLoader，避免复用 Trainer 等阶段持有的迭代状态。
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )
    # train_dl、eval_dl 是 prepare() 后的数据加载器，model、optimizer 同时被包装。
    # 训练 batch 会自动放到 accelerator.device，因此循环不再手动调用 to(device)。
    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    # prepare 后再计算 len(train_dl)：多进程时每个进程拿到的数据量可能改变。
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps), desc="Accelerate 训练")

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    progress_bar.close()
    print("Accelerate 训练完成；设备：", accelerator.device)

    # 原文最后的 Accelerate 版本只展示训练循环，没有附加第二次评估。
    # 原教程的启动与平台说明：
    # 1. 多卡训练可通过 accelerate config 配置，再通过 accelerate launch train.py
    #    启动。应把本节及其数据准备单独作为入口，而不是多进程启动前面全部教学实验。
    #    本文件的默认入口明确是单张 4090，不宣称整课已适配或验证多卡执行。
    # 2. TPU 更倾向固定形状，可在 tokenizer 中使用 padding="max_length"
    #    和 max_length；GPU 示例继续保留动态填充。
    # 3. Notebook 可把该节准备与循环放进 training_function，再调用：
    #       from accelerate import notebook_launcher
    #       notebook_launcher(training_function)
    #    这是原教程针对 Notebook 的另一种入口，本 .py 无需增加该封装函数。
    # Accelerate 官方接口说明：
    # https://huggingface.co/docs/accelerate/v0.34.2/en/quicktour

    print("\n三部分教程已按顺序执行。Trainer 输出目录：", OUTPUT_DIR)
