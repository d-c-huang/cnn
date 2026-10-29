import { lazy } from "react";
const GraphExperiment = lazy(() => import("../experiments/GraphExperiment"));
const BackpropExperiment = lazy(() => import("../experiments/BackpropExperiment"));
const XorExperiment = lazy(() => import("../experiments/XorExperiment"));

export const phase2Lessons = [
  {
    id: "graph",
    name: "计算图",
    en: "COMPUTATION GRAPH",
    title: "把一次计算，展开成一张图。",
    description: "从输入到损失，逐个查看运算节点，以及连接它们的局部导数。",
    task: "切换到反向梯度，查看 Tanh 计算图中梯度如何沿节点传回。",
    concept: "有向无环图 · 中间值 · 局部导数",
    component: GraphExperiment,
    code: "import numpy as np\n\nx1, x2 = 0.6, 0.9\nw1, w2, b, y = 0.8, -0.4, 0.2, 0.8\np1, p2 = x1 * w1, x2 * w2\ns = p1 + p2\nz = s + b\na = np.tanh(z)\ne = a - y\nloss = 0.5 * e ** 2\n\n# 每条边保存当前输入处的局部导数\n# ∂a/∂z = 1 - a²；∂L/∂e = e",
  },
  {
    id: "backprop",
    name: "反向传播",
    en: "BACKPROPAGATION",
    title: "沿着计算图，找回每一个梯度。",
    description:
      "先完成前向计算，再用链式法则反向传递梯度，最后应用一次真实的参数更新。",
    task: "逐步走到参数更新阶段，记录 w₁ 的梯度与更新前后值，再应用更新查看新损失。",
    concept: "链式法则 · 反向累积 · SGD 更新",
    component: BackpropExperiment,
    code: "# 接续计算图的前向结果\ndL_de = e\ndL_da = dL_de\ndL_dz = dL_da * (1 - a ** 2)\ndL_dw1 = dL_dz * x1\ndL_dw2 = dL_dz * x2\ndL_db = dL_dz\n\nlearning_rate = 0.1\nw1 -= learning_rate * dL_dw1\nw2 -= learning_rate * dL_dw2\nb -= learning_rate * dL_db\n\n# 反向传播负责算梯度；SGD 负责更新参数。\n# 新参数需要重新前向计算，才能得到新损失。",
  },
  {
    id: "xor",
    name: "XOR 网络训练",
    en: "TRAIN A NEURAL NETWORK",
    title: "从一次更新，到学会非线性。",
    description:
      "训练一个 2→4→1 的小型神经网络，同时观察中间激活、反向梯度和决策边界。",
    task: "实际训练网络，让四个 XOR 样本全部分类正确，并将平均 BCE 降到 0.2 以下。",
    concept: "隐藏层 · 二元交叉熵 · 全量训练",
    component: XorExperiment,
    code: "import numpy as np\n\nX = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])\ny = np.array([0., 1., 1., 0.])\nrng = np.random.default_rng(7)\nW1 = rng.normal(size=(2, 4)); b1 = np.zeros(4)\nW2 = rng.normal(size=4); b2 = 0.\n\nfor step in range(500):\n    a1 = np.tanh(X @ W1 + b1)\n    z2 = a1 @ W2 + b2\n    p = np.exp(-np.logaddexp(0., -z2))\n    loss = np.mean(np.logaddexp(0., z2) - y * z2)\n    d2 = (p - y) / len(X)\n    d1 = d2[:, None] * W2 * (1 - a1 ** 2)\n    # 先算完所有梯度，再一起更新；学习率 = 1\n    gW1, gb1 = X.T @ d1, d1.sum(axis=0)\n    gW2, gb2 = a1.T @ d2, d2.sum()\n    W1 -= gW1; b1 -= gb1; W2 -= gW2; b2 -= gb2",
  },
];
