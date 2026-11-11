import { lazy } from "react";
const Data = lazy(() => import("../experiments/language/DPOExperiment").then(m => ({ default: m.DPOData })));
const Train = lazy(() => import("../experiments/language/DPOExperiment").then(m => ({ default: m.DPOTraining })));
const Compare = lazy(() => import("../experiments/language/DPOExperiment").then(m => ({ default: m.DPOCompare })));
export const phase9Lessons = [
  { id: "dpo-data", name: "偏好数据与候选计分", en: "ONE PROMPT, TWO RESPONSES", title: "用两个回答，表达一份偏好。", description: "导入自定义 prompt / chosen / rejected 数据，检查分组、候选长度与真实回答掩码。", task: "校验偏好数据，点击候选回答 token，确认只为最终回答和 EOS 计分。", concept: "偏好对 · Chosen / Rejected · 回答掩码 · 数据划分", component: Data,
    code: 'from backend.dpo import RunConfig, service\n\nconfig = RunConfig().model_dump()\n# config["dataset"] = [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]\npreview = service.preview(config)\nprint(preview["chosen"], preview["rejected"])\n# 共享同一 prompt；仅最终候选与 EOS 计分，历史回答不计分\n# 超长候选拒绝，不静默截断偏好对' },
  { id: "dpo-train", name: "真实 LoRA DPO 优化", en: "LEARN PREFERENCES AGAINST A FIXED REFERENCE", title: "固定参考，让偏好推动策略更新。", description: "从 SFT 或预训练基座初始化策略与参考，真实训练 LoRA，观察 DPO 损失与参考相对间隔。", task: "真实 DPO 至少 2 步，使训练偏好损失下降，核对基座和参考保持固定。", concept: "DPO · 固定参考 · 序列 log 概率 · Beta · LoRA", component: Train,
    code: 'from backend.dpo import RunConfig, TrainingRequest, service\n\nconfig = RunConfig().model_dump()\n# config["source_sft_id"] = 已训练的SFT实验ID\nrun = service.create(config)\nfor _ in range(4):\n    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=5))\nprint(run["history"][-1])\n# loss = -logsigmoid(beta * ((policy_c-policy_r)-(ref_c-ref_r)))\n# logp 对候选 + EOS 求和；参考为创建时的固定快照' },
  { id: "dpo-compare", name: "优化前后对话对比", en: "COMPARE GENERATIONS UNDER IDENTICAL CONDITIONS", title: "同一个问题，比较优化前后的回答。", description: "用相同上下文和采样参数，对照固定参考与 DPO 当前或最佳适配器的真实生成。", task: "使用训练过的 DPO 权重完成生成对比，再试一次追问或未见问题。", concept: "固定基线 · 相同上下文 · 共享采样 · 偏好与生成质量", component: Compare,
    code: 'from backend.dpo import service, ChatRequest\n\nrun = service.list()[0]\nresult = service.compare(run["id"], ChatRequest(\n    messages=[{"role": "user", "content": "What is Zhiwei?"}],\n    temperature=0, checkpoint="latest",\n))\nprint("before:", result["before"]["reply"])\nprint("after:", result["after"]["reply"])\n# 两侧同一模板、历史、问题、采样设置；左侧永远是创建时参考' },
];
