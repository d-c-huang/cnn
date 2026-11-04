import { Grid3X3, Layers3, Trophy } from "lucide-react";

export type ExplorationWorkspace = "convolution" | "network-builder" | "challenges";
export type Workspace = "overview" | "lessons" | ExplorationWorkspace;
export type ReadingSize = "standard" | "comfortable" | "large";

export const explorationTools = [
  { id: "convolution" as const, name: "卷积可视化", en: "CONVOLUTION EXPLORER", description: "逐格扫描图像，让卷积、激活与池化的每一次计算都看得见。", icon: Grid3X3 },
  { id: "network-builder" as const, name: "神经网络搭建器", en: "NEURAL NETWORK BUILDER", description: "探索立体网络结构，亲手搭建、训练、评估，并用自己的模型识别数字。", icon: Layers3 },
  { id: "challenges" as const, name: "实验挑战关卡", en: "HANDS-ON CHALLENGES", description: "用真正的计算结果过关，把理解变成可以验证的能力。", icon: Trophy },
];

export function workspaceFromHash(): Workspace {
  const hash = window.location.hash.slice(1);
  if (!hash || hash === "overview" || hash === "main-content") return "overview";
  return explorationTools.some(tool => tool.id === hash) ? hash as Workspace : "lessons";
}

export function readReadingSize(): ReadingSize {
  try {
    const value = localStorage.getItem("zhiwei-reading-size-v1");
    return value === "standard" || value === "large" ? value : "comfortable";
  } catch { return "comfortable"; }
}