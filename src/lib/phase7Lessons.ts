import { lazy } from "react";
const Basics = lazy(() => import("../experiments/language/LoRAExperiment").then(module => ({ default: module.LoRABasics })));
const Training = lazy(() => import("../experiments/language/LoRAExperiment").then(module => ({ default: module.LoRATraining })));
const Generation = lazy(() => import("../experiments/language/LoRAExperiment").then(module => ({ default: module.LoRAGenerationLesson })));

export const phase7Lessons = [
  {
    id: "lora-basics", name: "预训练模型与 LoRA", en: "FREEZE THE BASE, LEARN AN ADAPTER",
    title: "保留预训练知识，只学习一个小更新。",
    description: "准备公开的 Pythia-14M 预训练模型，理解冻结基座、低秩矩阵与 LoRA 的参数节省。",
    task: "准备固定版本预训练模型，点选一个 rank，核对 A、B 形状和可训练参数量。",
    concept: "预训练基座 · 参数冻结 · 低秩更新 · rank · alpha", component: Basics,
    code: 'from backend.lora_assets import assets\nfrom backend.lora import RunConfig, service\n\nassets.ensure()  # 首次下载固定版本，之后使用本地缓存\nrun = service.create(RunConfig(rank=4, alpha=8).model_dump())\nprint(run["model_id"], run["revision"])\nprint(run["base_parameters"], run["trainable_parameters"])\n# W 保持冻结；只训练 ΔW = (alpha / rank) * B @ A\n# 六层 query_key_value：A [4,128]，B [384,4]',
  },
  {
    id: "lora-train", name: "真实 LoRA 微调", en: "FINETUNE A PRETRAINED LANGUAGE MODEL",
    title: "让预训练模型，学习你的领域文本。",
    description: "用原始子词分词器和 PEFT 适配器真实微调，查看训练/验证曲线，并验证基座参数始终冻结。",
    task: "创建 LoRA 实验，实际微调至少 2 步，使训练损失下降，并确认冻结校验通过。",
    concept: "PEFT · 因果语言建模 · AdamW · 冻结校验 · 续训", component: Training,
    code: 'from backend.lora import RunConfig, TrainingRequest, service\n\nrun = service.create(RunConfig().model_dump())\nfor _ in range(4):\n    run = service.train(run["id"], TrainingRequest(\n        expected_step=run["step"], steps=5,\n    ))\nprint(run["history"][-1])\nprint(run["frozen_verified"])\n# optimizer 只接收 requires_grad=True 的 LoRA 参数\n# 检查点保存适配器、AdamW 和采样状态，不重复保存基座',
  },
  {
    id: "lora-generate", name: "微调前后生成对比", en: "COMPARE BASE AND ADAPTED GENERATION",
    title: "用同一段提示，观察适配器带来的变化。",
    description: "加载当前或最佳验证适配器，以相同采样设置对比基座与微调输出，导出标准 PEFT 适配器。",
    task: "使用已微调的适配器完成一次真实生成，对照基座结果并查看适配器导出入口。",
    concept: "适配器开关 · 温度采样 · 基座版本 · PEFT 导出", component: Generation,
    code: 'from backend.lora import GenerationRequest, service\n\nrun = service.list()[0]\noutput = service.generate(run["id"], GenerationRequest(\n    prompt="Zhiwei is", checkpoint="latest", seed=7,\n))\nprint("BASE:", output["baseline"]["text"])\nprint("LORA:", output["adapted"]["text"])\nwith open("adapter.zip", "wb") as file:\n    file.write(service.export(run["id"], "latest"))\n# 解压后可用 PeftModel.from_pretrained(pinned_base, adapter_dir) 加载',
  },
];
