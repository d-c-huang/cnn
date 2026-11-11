const knownIds = ["data", "linear", "neuron", "loss", "gradient", "graph", "backprop", "xor", "mnist-data", "mnist-train", "mnist-evaluation", "handwriting", "deployment", "language-data", "language-train", "language-generate", "transformer-architecture", "transformer-train", "transformer-generate", "lora-basics", "lora-train", "lora-generate", "sft-data", "sft-train", "sft-chat", "dpo-data", "dpo-train", "dpo-compare", "grpo-rewards", "grpo-train", "grpo-compare"];
type Flags = Record<string, boolean>;
export interface LearningRecord {
  version: 1;
  active: string;
  completed: string[];
  evidence: { tasks: Flags; quizzes: Flags };
}
function flags(value: unknown): Flags {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("学习证据格式不正确。");
  if (Object.entries(value).some(([id, passed]) => !knownIds.includes(id) || typeof passed !== "boolean")) throw new Error("学习证据必须使用已知实验 ID 和布尔值。");
  return Object.fromEntries(Object.entries(value).filter(([id, passed]) => knownIds.includes(id) && passed === true));
}
export function parseLearningRecord(raw: string): LearningRecord {
  const value = JSON.parse(raw) as Partial<LearningRecord>;
  if (!value || value.version !== 1 || !Array.isArray(value.completed) || !value.evidence || typeof value.active !== "string") throw new Error("请选择知微导出的 v1 学习记录 JSON。");
  if (!knownIds.includes(value.active) || value.completed.some(id => !knownIds.includes(id))) throw new Error("记录中包含未知实验。");
  return { version: 1, active: value.active, completed: [...new Set(value.completed)], evidence: { tasks: flags(value.evidence.tasks), quizzes: flags(value.evidence.quizzes) } };
}
