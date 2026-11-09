import type { Activation } from "./neuron";

export const ACTIVATIONS: Activation[] = ["relu", "leaky_relu", "elu", "softplus", "silu", "sigmoid", "tanh", "linear"];

export const activationInfo = {
  relu: {
    label: "ReLU", color: "#8fddbc", range: "[0, +∞)", use: "隐藏层的常见起点",
    formula: String.raw`f(z)=\max(0,z)`, derivative: String.raw`f'(z)=\begin{cases}0&z<0\\1&z>0\end{cases}`,
    description: "正值直接通过，负值归零。负半轴的局部导数为 0；若一个单元长期对所有训练输入都落在负区间，它可能难以从梯度更新中恢复。一次输出为 0 并不等于这个单元已经“死亡”。",
    observation: "点击“负值区”，再切换 Leaky ReLU，比较负输入的输出与梯度。",
  },
  leaky_relu: {
    label: "Leaky ReLU", color: "#e4ba7f", range: "(−∞, +∞)", use: "希望负半轴仍传递梯度的隐藏层",
    formula: String.raw`f(z)=\begin{cases}0.1z&z<0\\z&z\ge0\end{cases}`, derivative: String.raw`f'(z)=\begin{cases}0.1&z<0\\1&z>0\end{cases}`,
    description: "本实验固定负半轴斜率为 0.1。负值不会被完全截断，也能传递较小的梯度。零点左右斜率不同，因此这里仍然不可导，不能把它当成平滑函数。",
    observation: "在 z = −3 时比较 ReLU 的导数 0 和 Leaky ReLU 的导数 0.1。",
  },
  elu: {
    label: "ELU", color: "#b5a2e9", range: "(−1, +∞)", use: "允许负输出的隐藏层",
    formula: String.raw`f(z)=\begin{cases}e^z-1&z<0\\z&z\ge0\end{cases}`, derivative: String.raw`f'(z)=\begin{cases}e^z&z<0\\1&z\ge0\end{cases}`,
    description: "本实验固定 α = 1，正半轴保持线性，负半轴逐渐趋近 −1。此时零点处一阶导数连续；很负的输入仍会进入饱和区，局部导数趋近 0。",
    observation: "比较 z = −3 与 z = 0：负输出可以存在，但负侧越饱和，导数越小。",
  },
  softplus: {
    label: "Softplus", color: "#78cbdc", range: "(0, +∞)", use: "平滑激活；需要正数的输出参数",
    formula: String.raw`f(z)=\ln(1+e^z)`, derivative: String.raw`f'(z)=\sigma(z)`,
    description: "可视为 ReLU 的平滑版本，本实验采用 β = 1。输出始终为正，零点没有尖角，f(0) = ln 2；它的导数正好是 Sigmoid。大负输入处的梯度仍会变小。",
    observation: "点击“零点”，比较 ReLU 的尖角与 Softplus 的导数 0.5。",
  },
  silu: {
    label: "SiLU", color: "#e79cae", range: "约 [−0.2785, +∞)", use: "需要平滑激活的隐藏层",
    formula: String.raw`f(z)=z\sigma(z)`, derivative: String.raw`f'(z)=\sigma(z)+z\sigma(z)(1-\sigma(z))`,
    description: "用输入自身的 Sigmoid 值作为门控，也叫 β = 1 的 Swish。它平滑但不是全程单调：部分负区间的导数小于 0，部分正区间的导数大于 1，不能只凭权重正负判断输出曲线方向。",
    observation: "点击“负值区”观察负导数，再点击“零点”观察导数 0.5。",
  },
  sigmoid: {
    label: "Sigmoid", color: "#d5d785", range: "(0, 1)", use: "二分类或多标签的输出映射",
    formula: String.raw`f(z)=\sigma(z)=\frac1{1+e^{-z}}`, derivative: String.raw`f'(z)=\sigma(z)(1-\sigma(z))`,
    description: "将实数压到 0 与 1 之间，零点处导数最大，为 0.25。两端饱和时梯度变小；多层相乘可能让梯度衰减。本实验只展示激活值，它本身不是经过校准的置信度。",
    observation: "比较“零点”和“大正值”：输出接近 1 时，导数如何变小？",
  },
  tanh: {
    label: "Tanh", color: "#9fbbee", range: "(−1, 1)", use: "需要正负、有界激活的隐藏层",
    formula: String.raw`f(z)=\tanh(z)`, derivative: String.raw`f'(z)=1-\tanh^2(z)`,
    description: "输出以 0 为中心，可保留正负方向，零点处导数为 1。两端接近 −1 或 1 时同样会饱和；输出以零为中心并不代表没有梯度衰减。",
    observation: "在同一个 z 下比较 Tanh 与 Sigmoid 的导数，观察各自的饱和速度。",
  },
  linear: {
    label: "线性", color: "#b9c5c4", range: "(−∞, +∞)", use: "不限制范围的回归输出；原始分类分数",
    formula: String.raw`f(z)=z`, derivative: String.raw`f'(z)=1`,
    description: "直接保留加权和，局部导数始终为 1。多个仅含线性变换与偏置的层，仍可合并成一个仿射变换；要表达复杂的非线性关系，隐藏层通常需要非线性激活。",
    observation: "切换三个示例，确认输出始终等于 z，局部导数始终为 1。",
  },
} satisfies Record<Activation, { label: string; color: string; range: string; use: string; formula: string; derivative: string; description: string; observation: string }>;
