import { lazy } from "react";

const LanguageData = lazy(() => import("../experiments/language/LanguageExperiment").then(module => ({ default: module.LanguageData })));
const LanguageTraining = lazy(() => import("../experiments/language/LanguageExperiment").then(module => ({ default: module.LanguageTraining })));
const LanguageGeneration = lazy(() => import("../experiments/language/LanguageExperiment").then(module => ({ default: module.LanguageGeneration })));

export const phase5Lessons = [
  {
    id: "language-data", name: "文本与下一字符", en: "TEXT TO TOKENS",
    title: "把一段文字，变成预测任务。",
    description: "从字符词表和 token ID 开始，把相邻字符组成输入与标签，理解语言模型怎样从文本中获得监督信号。",
    task: "准备教学语料，点击一个训练样本，核对输入字符与它的下一个字符。",
    concept: "字符分词 · 词表 · 标签右移 · 验证划分", component: LanguageData,
    code: 'from backend.language import CorpusRequest, prepare\n\ndata = prepare(CorpusRequest())\nprint(data["vocab"])\n# 按字符切分；词表只使用前 80% 的训练文本\n# 验证文本中的新字符映射到 <UNK>（ID 0）\nfor pair in data["pairs"]:\n    print(pair["input"], "→", pair["target"])\n# 每一对都是：当前字符 → 紧接着的下一个字符',
  },
  {
    id: "language-train", name: "训练语言模型", en: "LEARN THE NEXT TOKEN",
    title: "让下一个字符，从猜测变成学习。",
    description: "真实训练一个字符级 Bigram 模型，追踪交叉熵、验证损失与困惑度，把前向、梯度和参数更新连起来。",
    task: "执行至少 3 次真实参数更新，让训练交叉熵低于初始化时的值，并比较验证损失。",
    concept: "Bigram · Softmax · 交叉熵 · SGD · 困惑度", component: LanguageTraining,
    code: 'from backend.language import CorpusRequest, TrainRequest, prepare, train\n\ndata = prepare(CorpusRequest())\nresult = train(TrainRequest(\n    weights=data["weights"], steps=100, learning_rate=5.0\n))\nprint(result["history"][-1])\n# 模型：logits = W[current_token]，W 的形状是 [V, V]\n# p = softmax(logits)；L = mean(-log(p[target]))\n# dW = (每行输入次数 × p - 相邻字符计数) / 样本数\n# W -= learning_rate * dW；验证数据不参与更新',
  },
  {
    id: "language-generate", name: "文本生成与温度", en: "SAMPLE ONE TOKEN AT A TIME",
    title: "从一个字符，续写一段文字。",
    description: "用刚训练的权重逐字符采样，比较温度怎样改变概率分布，并理解单字符上下文模型的能力边界。",
    task: "使用已训练的模型和词表内的提示词生成一次文本，再改变温度比较输出。",
    concept: "自回归 · 温度采样 · 随机种子 · 上下文", component: LanguageGeneration,
    code: 'from backend.language import CorpusRequest, TrainRequest, GenerateRequest, prepare, train, generate\n\ndata = prepare(CorpusRequest())\ntrained = train(TrainRequest(weights=data["weights"], steps=200))\noutput = generate(GenerateRequest(\n    weights=trained["weights"], prompt="小猫",\n    temperature=0.8, length=60, seed=7\n))\nprint("小猫" + output["continuation"])\n# 每次按 softmax(W[last_token] / temperature) 采样\n# 新字符放回输入；本模型仅依赖最后一个 token',
  },
];
