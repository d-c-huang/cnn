import type { CSSProperties } from "react";
import { useEffect, useState } from "react";
import { useMotionLevel } from "../lib/motion";
const nodes = ["数据", "前向", "损失", "梯度", "更新"];
const positions: Record<string, number> = { data: 0, linear: 1, neuron: 1, loss: 2, gradient: 4, graph: 1, backprop: 3, xor: 4, "mnist-data": 0, "mnist-train": 4, "mnist-evaluation": 2, handwriting: 1, deployment: 1, "language-data": 0, "language-train": 4, "language-generate": 1, "transformer-architecture": 1, "transformer-train": 4, "transformer-generate": 1, "lora-basics": 1, "lora-train": 4, "lora-generate": 1, "sft-data": 0, "sft-train": 4, "sft-chat": 1, "dpo-data": 0, "dpo-train": 4, "dpo-compare": 1, "grpo-rewards": 0, "grpo-train": 4, "grpo-compare": 1 };

export default function TraceRibbon({ lessonId }: { lessonId: string }) {
  const motion = useMotionLevel();
  const [entered, setEntered] = useState(() => motion !== "full");
  useEffect(() => { if (motion !== "full") setEntered(true); }, [motion]);
  useEffect(() => {
    const timer = window.setTimeout(() => setEntered(true), 1100);
    return () => window.clearTimeout(timer);
  }, []);
  const backward = ["backprop", "xor", "mnist-train", "language-train", "transformer-train", "lora-train", "sft-train", "dpo-train", "grpo-train"].includes(lessonId);
  return <nav className={`trace-ribbon ${backward ? "trace-backward" : ""}`} data-entered={entered} aria-label="模型计算通路">
    <ol>{nodes.map((label, index) => <li key={label} style={{ "--trace-index": index } as CSSProperties} aria-current={positions[lessonId] === index ? "step" : undefined}><span className="trace-node">{String(index + 1).padStart(2, "0")}</span><span>{label}</span>{index < 4 && <span className="trace-link" aria-hidden="true" />}</li>)}</ol>
    <span className="trace-direction">{backward ? "前向计算 / 反向回传" : "前向计算"}</span>
  </nav>;
}
