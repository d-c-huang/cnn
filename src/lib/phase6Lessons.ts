import { lazy } from "react";

const Architecture = lazy(() => import("../experiments/language/TransformerExperiment").then(module => ({ default: module.TransformerArchitecture })));
const Training = lazy(() => import("../experiments/language/TransformerExperiment").then(module => ({ default: module.TransformerTraining })));
const Generation = lazy(() => import("../experiments/language/TransformerExperiment").then(module => ({ default: module.TransformerGenerationLesson })));

export const phase6Lessons = [
  {
    id: "transformer-architecture", name: "Transformer 与因果注意力", en: "ATTEND TO THE PAST",
    title: "让每个字符，看见它的上下文。",
    description: "从 Bigram 的单字符查表走向多头自注意力，连起嵌入、位置、因果掩码、残差与前馈网络。",
    task: "点选一个查询位置，核对可见前文与被屏蔽的未来位置。",
    concept: "Token 嵌入 · 位置嵌入 · 多头注意力 · 因果掩码", component: Architecture,
    code: 'import torch\nfrom backend.transformer import RunConfig, build_transformer\n\nconfig = RunConfig().model_dump()\nmodel = build_transformer(vocab_size=31, config=config)\nx = torch.tensor([[12, 18, 9]])  # 示例 token ID\nlogits, attention = model(x, return_attention=True)\nprint(logits.shape)     # [1, 3, 31]\nprint(attention.shape)  # [1, 2, 3, 3]，最后一层各注意力头\n# scores = (Q @ K.transpose(-2, -1)) / sqrt(head_dim)\n# attention = softmax(scores.masked_fill(upper_triangle, -inf))\n# 上三角是未来位置，注意力权重严格为 0',
  },
  {
    id: "transformer-train", name: "真实训练 Transformer", en: "TRAIN A TINY GPT",
    title: "从随机权重，训练你的第一个 Transformer。",
    description: "设置上下文与模型规模，用 PyTorch、AdamW 和右移标签真实训练，保存当前与最佳验证权重。",
    task: "创建模型，实际训练至少 3 步，让训练损失低于初始化值；对照验证曲线与最佳权重步数。",
    concept: "右移标签 · AdamW · 梯度裁剪 · 检查点 · 过拟合", component: Training,
    code: 'from backend.transformer import RunConfig, TrainingRequest, service\n\nrun = service.create(RunConfig().model_dump())\nfor _ in range(4):\n    run = service.train(run["id"], TrainingRequest(\n        expected_step=run["step"], steps=25,\n    ))\nprint(run["history"][-1])\n# logits = model(input_ids)\n# loss = cross_entropy(logits.reshape(-1, V), targets.reshape(-1))\n# zero_grad → backward → clip_grad_norm_(..., 1.0) → AdamW.step\n# 完整检查点保存在 runs/transformer/<id>/checkpoint.pt',
  },
  {
    id: "transformer-generate", name: "Transformer 文本生成", en: "GENERATE WITH CONTEXT",
    title: "用训练过的注意力网络，逐字续写。",
    description: "载入当前或最佳验证权重，调节温度与 Top-k 生成文本，并查看实际注意力头的上下文分配。",
    task: "使用已训练权重和词表内提示词生成文本，查看首个生成位置的候选概率与真实注意力。",
    concept: "自回归 · 上下文滑窗 · Top-k · 实际注意力", component: Generation,
    code: 'from backend.transformer import GenerationRequest, service\n\nrun = service.list()[0]  # 已创建的实验\nresult = service.generate(run["id"], GenerationRequest(\n    prompt="小猫", length=60, temperature=0.8,\n    top_k=5, seed=7, checkpoint="latest",\n))\nprint(result["prompt"] + result["continuation"])\nprint(result["context_tokens"])\n# model.eval() + torch.inference_mode()\n# 仅使用最后 context_length 个 token；每次采样后接回输入\n# result["attention"] 是首个生成位置、最后一层的各头注意力矩阵',
  },
];
