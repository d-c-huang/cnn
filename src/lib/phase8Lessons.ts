import { lazy } from "react";
const Data = lazy(() => import("../experiments/language/SFTExperiment").then(module => ({ default: module.SFTData })));
const Training = lazy(() => import("../experiments/language/SFTExperiment").then(module => ({ default: module.SFTTraining })));
const Chat = lazy(() => import("../experiments/language/SFTExperiment").then(module => ({ default: module.SFTChat })));

export const phase8Lessons = [
  {
    id: "sft-data", name: "问答数据与回答掩码", en: "TEACH WITH QUESTIONS AND ANSWERS",
    title: "问题是上下文，回答才是监督目标。",
    description: "编辑或导入 JSON / JSONL 问答与多轮样本，检查数据划分、对话模板和真实 token 标签。",
    task: "校验自定义或示例数据集，点击一个回答 token，确认其参与损失而提示词被忽略。",
    concept: "问答数据 · 对话模板 · 回答掩码 · −100 · 样本划分", component: Data,
    code: 'from backend.sft import RunConfig, service\n\nconfig = RunConfig().model_dump()\n# 自定义数据：config["dataset"] = [{"question": "...", "answer": "..."}, ...]\npreview = service.preview(config)\nprint(preview["train_indices"], preview["validation_indices"])\nprint(preview["tokens"])\n# 用户/系统/角色前缀标签为 -100；助手回答和 EOS 保留 token ID\n# max_length 超限会明确拒绝，不静默截掉回答',
  },
  {
    id: "sft-train", name: "真实 LoRA 指令微调", en: "OPTIMIZE ONLY ASSISTANT ANSWERS",
    title: "用示范回答，训练模型的应答方式。",
    description: "在固定预训练基座上真实执行回答监督 LoRA SFT，查看有效 token 加权损失、冻结校验和最佳适配器。",
    task: "创建问答微调实验，实际训练至少 2 步，让回答损失下降并通过基座冻结校验。",
    concept: "SFT · 右移标签 · Padding mask · LoRA · 最佳验证适配器", component: Training,
    code: 'from backend.sft import RunConfig, TrainingRequest, service\n\nrun = service.create(RunConfig().model_dump())\nfor _ in range(20):\n    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=5))\nprint(run["history"][-1])\n# logits[:, :-1] 预测 labels[:, 1:]，只右移一次\n# cross_entropy(..., ignore_index=-100)\n# 分母仅包含有效助手回答与 EOS 目标；填充 attention_mask=0',
  },
  {
    id: "sft-chat", name: "SFT 多轮对话生成", en: "CHAT WITH YOUR FINETUNED ADAPTER",
    title: "把微调后的模型，放进真正的对话。",
    description: "沿用训练模板，加载当前或最佳适配器完成多轮对话，查看上下文裁剪、EOS 停止与实际模型步数。",
    task: "使用训练过的 SFT 适配器发送问题，成功获得非空回答，再尝试追问或切换权重。",
    concept: "多轮历史 · 生成前缀 · EOS · 上下文预算 · 模板导出", component: Chat,
    code: 'from backend.sft import ChatRequest, service\n\nrun = service.list()[0]\nmessages = [{"role": "user", "content": "What is Zhiwei?"}]\nresult = service.chat(run["id"], ChatRequest(messages=messages))\nprint(result["reply"], result["finish_reason"])\nmessages += [{"role": "assistant", "content": result["reply"]},\n             {"role": "user", "content": "What is LoRA?"}]\n# 下一次请求沿用同一模板；追加 Assistant:，不提供真实回答\n# 适配器 ZIP 同时导出 chat_template.jinja 与 sft_format.json',
  },
];
