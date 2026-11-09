import { lazy } from "react";
const DataExperiment = lazy(() => import("../experiments/DataExperiment"));
const LinearExperiment = lazy(() => import("../experiments/LinearExperiment"));
const NeuronExperiment = lazy(() => import("../experiments/NeuronExperiment"));
const LossExperiment = lazy(() => import("../experiments/LossExperiment"));
const GradientExperiment = lazy(() => import("../experiments/GradientExperiment"));

export const phase1Lessons = [
  {
    id: "data",
    name: "数据与张量",
    en: "DATA & TENSORS",
    title: "数据，是模型的起点。",
    description: "从一张小小的像素图开始，看看计算机眼中的图片是什么样子。",
    task: "选中一个像素，把灰度调到 128。打开归一化后，它会变成多少？",
    concept: "样本 · 特征 · 矩阵 · 归一化",
    component: DataExperiment,
    code: '# 灰度图片 → 归一化矩阵 → 输入向量\nimport numpy as np\n\n# pixels 是你在实验中编辑的 8×8 灰度数组\nmatrix = np.asarray(pixels, dtype=np.float64)\nnormalized = matrix / 255.0\nvector = normalized.ravel(order="C")\n\nprint(matrix.shape)      # (8, 8)\nprint(vector.shape)      # (64,)\nprint(normalized.mean()) # 转换后的均值',
  },
  {
    id: "linear",
    name: "线性模型",
    en: "LINEAR MODEL",
    title: "一个权重，就能改变预测。",
    description: "模型从最简单的直线开始。调整两个参数，看它如何接近真实数据。",
    task: "先只改变权重，再只改变偏置。让五个样本的均方误差 MSE 降到 0.001 以下。",
    concept: "权重 · 偏置 · 预测 · 均方误差",
    component: LinearExperiment,
    code: "import numpy as np\n\nx = np.array([-2., -1., 0., 1., 2.])\ny = 2.0 * x + 0.5     # 构造教学样本\n\nweight, bias = 0.5, 0.0\nprediction = weight * x + bias\nerror = prediction - y\nmse = np.mean(error ** 2)\n\nprint(prediction)\nprint(mse)",
  },
  {
    id: "neuron",
    name: "单个神经元",
    en: "A SINGLE NEURON",
    title: "看看一个神经元，如何计算。",
    description: "追踪加权求和，比较八种激活函数的输出、局部导数与饱和现象。",
    task: "选择 Sigmoid，将权重 1 调为负数，观察输出曲线如何随输入 1 增大而下降。",
    concept: "加权求和 · 八种激活 · 局部导数与饱和",
    component: NeuronExperiment,
    code: "import numpy as np\n\ninputs = np.array([0.6, 0.9])\nweights = np.array([0.8, -0.4])\nbias = 0.2\nz = (inputs * weights).sum() + bias\np = np.exp(-np.logaddexp(0.0, -z))\n\noutputs = {\n    'ReLU': np.maximum(0.0, z),\n    'Leaky ReLU': np.where(z >= 0, z, 0.1 * z),\n    'ELU': np.where(z >= 0, z, np.expm1(np.minimum(z, 0))),\n    'Softplus': np.logaddexp(0.0, z),\n    'SiLU': z * p,\n    'Sigmoid': p,\n    'Tanh': np.tanh(z),\n    'Linear': z,\n}\n\n# 以 SiLU 为例：局部导数与输入导数\nda_dz = p + z * p * (1 - p)\nda_dx1 = da_dz * weights[0]\n# 完整损失梯度还需要乘以上游 dL/da\nprint(outputs, da_dz, da_dx1)",
  },
  {
    id: "loss",
    name: "损失函数",
    en: "LOSS FUNCTION",
    title: "模型怎么知道，自己错了？",
    description: "比较 MSE、MAE、Huber 与二分类交叉熵，观察不同任务怎样衡量错误、产生梯度。",
    task: "让预测值与真实标签相等，再分别向两边移动相同距离，比较损失和梯度。",
    concept: "回归与分类 · 损失对比 · 梯度",
    component: LossExperiment,
    code: "import numpy as np\n\nprediction, target = -1.0, 2.0\n\n# 单样本平方误差\nloss = (prediction - target) ** 2\n\n# 损失对预测值的导数\ngradient = 2.0 * (prediction - target)\n\nprint(loss)      # 9.0\nprint(gradient)  # -6.0",
  },
  {
    id: "gradient",
    name: "梯度下降",
    en: "GRADIENT DESCENT",
    title: "学习，就是一次次微小的更新。",
    description: "从一次梯度更新出发，对比五种优化器在平方损失与狭长谷底中的速度、振荡和参数轨迹。",
    task: "先用学习率 0.1 完成至少 3 次真实更新，观察损失下降；再比较 0.5 与 0.55 的变化。",
    concept: "学习率 · 动量 · 自适应步长 · 优化器对比",
    component: GradientExperiment,
    code: "import numpy as np\n\nx = np.array([-2., -1., 0., 1., 2.])\ny = 2.0 * x + 0.5\nweight, bias, learning_rate = -2.0, 0.5, 0.1\n\nfor step in range(20):\n    error = weight * x + bias - y\n    loss = np.mean(error ** 2)\n    gradient = 2.0 * np.mean(error * x)\n    weight -= learning_rate * gradient\n    print(step, weight, loss)\n\n# b 固定；每一步使用全部 5 个教学样本",
  },
];
