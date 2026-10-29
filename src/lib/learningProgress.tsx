import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

export const lessonTasks: Record<string, { title: string; instruction: string }> = {
  data: { title: "观察灰度归一化", instruction: "把选中像素的灰度设为 128，并打开归一化，确认转换后的数值约为 0.502。" },
  linear: { title: "找到准确拟合", instruction: "调整权重与偏置，让五个样本的均方误差 MSE 小于 0.001。" },
  neuron: { title: "观察负权重的作用", instruction: "选择 Sigmoid 激活，将权重 w₁ 调为负值，观察输出随输入 x₁ 增大而下降。" },
  loss: { title: "找到损失最低点", instruction: "让预测值等于真实标签，确认平方误差与梯度同时为 0。" },
  gradient: { title: "验证梯度下降", instruction: "实际执行至少 3 次参数更新，并让损失低于本次训练的初始损失。" },
  graph: { title: "读懂局部梯度", instruction: "切换到反向梯度，查看 Tanh 节点的局部导数与梯度传递。" },
  backprop: { title: "完成一次反向传播更新", instruction: "沿计算图完成反向传播，并实际应用至少一次参数更新。" },
  xor: { title: "学会 XOR 分类", instruction: "实际训练网络，让四个样本全部分类正确，且二元交叉熵 BCE 小于 0.2。" },
  "mnist-data": { title: "验证独立的数据划分", instruction: "载入真实 MNIST 数据，确认训练、验证与独立测试集的划分已就绪。" },
  "mnist-train": { title: "获得训练好的模型", instruction: "查看一次已完成的真实训练，确认训练已保存最佳模型。" },
  "mnist-evaluation": { title: "完成独立测试评估", instruction: "使用训练好的模型完成非空测试集评估，查看真实评估结果。" },
  handwriting: { title: "完成一次真实推理", instruction: "提交图像并成功获得模型预测，核对预处理图像与类别分数。" },
  deployment: { title: "验证服务与模型可用", instruction: "确认推理 API 已连接，且模型列表中至少有一个可用模型。" },
};

export const lessonTaskRequirements: Record<string, string> = Object.fromEntries(
  Object.entries(lessonTasks).map(([id, task]) => [id, task.instruction]),
);

type Flags = Record<string, boolean>;
type Evidence = { tasks: Flags; quizzes: Flags };
type ProgressContextValue = Evidence & {
  verifyTask: (id: string) => void;
  verifyQuiz: (id: string) => void;
  reset: () => void;
};

const storageKey = "zhiwei-learning-evidence-v1";
const knownIds = new Set(Object.keys(lessonTasks));
const emptyEvidence = (): Evidence => ({ tasks: {}, quizzes: {} });

function readFlags(value: unknown): Flags {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const source = value as Record<string, unknown>;
  return Object.fromEntries([...knownIds].filter(id => Object.hasOwn(source, id) && source[id] === true).map(id => [id, true]));
}

function parseEvidence(raw: string | null): Evidence {
  if (!raw) return emptyEvidence();
  try {
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object" || Array.isArray(value)) return emptyEvidence();
    const saved = value as Record<string, unknown>;
    if (saved.version !== 1) return emptyEvidence();
    return { tasks: readFlags(saved.tasks), quizzes: readFlags(saved.quizzes) };
  } catch { return emptyEvidence(); }
}

function loadEvidence(): Evidence {
  try { return parseEvidence(localStorage.getItem(storageKey)); }
  catch { return emptyEvidence(); }
}

const ProgressContext = createContext<ProgressContextValue | null>(null);

export function LearningProgressProvider({ children }: { children: ReactNode }) {
  const [evidence, setEvidence] = useState<Evidence>(loadEvidence);
  // The ref also prevents duplicate setters before React commits a verification.
  const latest = useRef(evidence);

  const verifyTask = useCallback((id: string) => {
    if (!knownIds.has(id) || latest.current.tasks[id]) return;
    const next = { ...latest.current, tasks: { ...latest.current.tasks, [id]: true } };
    latest.current = next;
    setEvidence(next);
  }, []);

  const verifyQuiz = useCallback((id: string) => {
    if (!knownIds.has(id) || latest.current.quizzes[id]) return;
    const next = { ...latest.current, quizzes: { ...latest.current.quizzes, [id]: true } };
    latest.current = next;
    setEvidence(next);
  }, []);

  const reset = useCallback(() => {
    if (!Object.keys(latest.current.tasks).length && !Object.keys(latest.current.quizzes).length) return;
    const next = emptyEvidence();
    latest.current = next;
    setEvidence(next);
  }, []);

  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify({ version: 1, ...evidence })); }
    catch { /* Evidence remains usable in memory when storage is unavailable. */ }
  }, [evidence]);

  const value = useMemo(() => ({ ...evidence, verifyTask, verifyQuiz, reset }), [evidence, verifyTask, verifyQuiz, reset]);
  return <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>;
}

export function useLearningProgress(): ProgressContextValue {
  const value = useContext(ProgressContext);
  if (!value) throw new Error("Learning progress must be used inside LearningProgressProvider.");
  return value;
}

/** Record a verified observation once; later parameter changes retain the evidence. */
export function useLessonTask(lessonId: string, verified: boolean) {
  const { verifyTask } = useLearningProgress();
  useEffect(() => {
    if (verified) verifyTask(lessonId);
  }, [lessonId, verified, verifyTask]);
}
