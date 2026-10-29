export interface LessonInsight {
  objective: string;
  steps: readonly [
    { title: string; description: string },
    { title: string; description: string },
    { title: string; description: string },
  ];
  formula: string;
  formulaNote: string;
  misconception: string;
  quiz: {
    question: string;
    options: readonly [string, string, string];
    answer: 0 | 1 | 2;
    explanation: string;
  };
}

export const lessonInsights: Record<string, LessonInsight> = {
  data: {
    objective: "把像素、矩阵与向量对应起来，理解归一化改变了什么。",
    steps: [
      { title: "像素就是输入", description: "灰度图的每个位置存一个亮度值：0 是黑色，255 是白色，中间值代表不同灰度。" },
      { title: "缩放数值范围", description: "将所有像素除以 255，把范围变为 0–1。图像的相对明暗关系保持不变。" },
      { title: "按顺序展平", description: "逐行读取 8 × 8 矩阵，得到 64 维向量。值与数量不变，排列形状发生变化。" },
    ],
    formula: String.raw`x' = \frac{x}{255} \qquad [8,8] \longrightarrow [64]`,
    formulaNote: "例如灰度 128 归一化后约为 0.502；它仍然对应原来的那个像素。",
    misconception: "展平没有提取新特征，也没有减少元素。这里的 8 × 8 图是教学样本，后续 MNIST 使用 28 × 28 图像。",
    quiz: {
      question: "把 8 × 8 灰度矩阵展平成向量后，元素数量是多少？",
      options: ["8 个", "16 个", "64 个"],
      answer: 2,
      explanation: "8 × 8 = 64。展平只改变形状；归一化只改变数值范围，两者都不会减少元素数量。",
    },
  },
  linear: {
    objective: "分别控制权重和偏置，用误差判断直线是否接近样本。",
    steps: [
      { title: "输入变成预测", description: "模型把输入 x 乘以权重 w，再加上偏置 b，得到预测值。" },
      { title: "两个参数，两种作用", description: "权重决定直线的斜率，偏置决定整体上下移动。每次只改一个参数更容易观察。" },
      { title: "用全部样本衡量", description: "分别计算五个样本的平方误差，再取平均，得到均方误差 MSE。" },
    ],
    formula: String.raw`\begin{gathered}\hat{y} = wx+b \\ \mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2\end{gathered}`,
    formulaNote: "本实验的目标关系是 y = 2x + 0.5。当 w = 2、b = 0.5 时，五个教学样本的误差为 0。",
    misconception: "一条直线经过某一个点，不代表整体拟合好。需要同时观察所有样本与平均损失。",
    quiz: {
      question: "保持权重不变，只增大偏置 b，会发生什么？",
      options: ["整条预测直线上移", "直线斜率变大", "所有预测值都变成 0"],
      answer: 0,
      explanation: "偏置对每个输入都加上同一个数，所以直线平行上移。斜率仍由权重 w 决定。",
    },
  },
  neuron: {
    objective: "逐项追踪输入贡献，并比较不同激活函数如何改变输出。",
    steps: [
      { title: "先看每条连接", description: "每个输入与自己的权重相乘。正负号决定它对加权和的推动方向。" },
      { title: "汇总成加权和", description: "把各项乘积相加，再加入偏置，得到激活前的值 z。" },
      { title: "经过激活函数", description: "ReLU 截断负值，Sigmoid 压到 0–1，Tanh 压到 −1–1；线性激活直接保留 z。" },
    ],
    formula: String.raw`\begin{gathered}z=w_1x_1+w_2x_2+b \\ a=f(z)\end{gathered}`,
    formulaNote: "先判断 z 如何变化，再根据激活函数判断 a。负权重配合更大的输入，会使这一项乘积更小。",
    misconception: "输入增大不一定让输出增大：权重可能为负，ReLU 还可能把一段负的 z 都映射为 0。",
    quiz: {
      question: "当加权和 z = −0.4 时，ReLU 的输出是多少？",
      options: ["−0.4", "0", "0.4"],
      answer: 1,
      explanation: "ReLU(z) = max(0, z)，因此负的加权和被映射为 0；它并不是取绝对值。",
    },
  },
  loss: {
    objective: "区分损失的大小和梯度的方向，读懂平方误差曲线。",
    steps: [
      { title: "先比较预测与标签", description: "预测减去真实标签得到有正负号的误差，表示预测偏大还是偏小。" },
      { title: "把误差平方", description: "平方误差始终非负。离目标同样远的两个预测，损失相同。" },
      { title: "读曲线的斜率", description: "梯度为正表示向右增大预测会增加损失；梯度为负时，向右移动会降低损失。" },
    ],
    formula: String.raw`\begin{gathered}L=(\hat{y}-y)^2 \\ \frac{\partial L}{\partial\hat{y}}=2(\hat{y}-y)\end{gathered}`,
    formulaNote: "当预测等于标签时，损失与梯度都为 0。本节使用单样本平方误差，不是多个样本的平均值。",
    misconception: "相同的损失不一定意味着相同的更新方向。目标两侧的等距点，损失相等，梯度符号相反。",
    quiz: {
      question: "真实标签是 2，预测分别是 1 和 3，它们的平方误差有什么关系？",
      options: ["预测为 3 的损失更大", "损失相同，梯度方向相反", "损失相同，梯度也相同"],
      answer: 1,
      explanation: "两者损失都是 1；梯度分别为 −2 与 +2，因此降低损失所需的移动方向相反。",
    },
  },
  gradient: {
    objective: "把梯度变成实际更新，比较学习率对收敛和振荡的影响。",
    steps: [
      { title: "计算当前梯度", description: "用当前权重计算所有教学样本的平均损失，再得到损失对权重的导数。" },
      { title: "沿负梯度更新", description: "从当前权重减去学习率乘以梯度。学习率决定这一步走多远。" },
      { title: "重新观察损失", description: "更新后的参数需要重新计算损失。连续单步执行，观察下降、来回振荡或发散。" },
    ],
    formula: String.raw`w_{t+1}=w_t-\eta\,\frac{\partial L}{\partial w}`,
    formulaNote: "本节固定 b = 0.5，只更新 w。对应损失为 L = 2(w − 2)²，最低点在 w = 2。",
    misconception: "学习率越大不代表学习越快。步长过大可能越过最低点，持续振荡甚至让损失增大。",
    quiz: {
      question: "当前梯度为正，学习率也为正，梯度下降会怎样更新权重？",
      options: ["增大权重", "保持不变", "减小权重"],
      answer: 2,
      explanation: "更新是 w − η × 梯度。当 η 与梯度都为正时，减去一个正数，所以权重减小。",
    },
  },
  graph: {
    objective: "把整体计算拆成局部运算，连接前向数值与局部导数。",
    steps: [
      { title: "沿箭头前向计算", description: "输入与参数经过乘法、加法和 Tanh，最终得到预测与损失；每个节点保存中间值。" },
      { title: "检查局部导数", description: "每条边描述一个局部依赖。乘法的导数取决于另一个乘数，Tanh 的导数取决于当前输出。" },
      { title: "连接局部变化", description: "沿路径组合局部导数，就能解释某个输入或参数的变化如何影响最终损失。" },
    ],
    formula: String.raw`\begin{gathered}a=\tanh(z) \\ \frac{\partial a}{\partial z}=1-a^2\end{gathered}`,
    formulaNote: "当 Tanh 输出接近 ±1 时，局部导数接近 0。同样的上游梯度经过这里后会变小。",
    misconception: "局部导数与最终损失对参数的梯度不是同一个量。后者还要结合后续节点传回的梯度。",
    quiz: {
      question: "当 Tanh 输出 a 接近 1 时，局部导数 1 − a² 接近多少？",
      options: ["0", "1", "2"],
      answer: 0,
      explanation: "a² 接近 1，因此 1 − a² 接近 0。这解释了 Tanh 在饱和区域对输入变化不敏感的原因。",
    },
  },
  backprop: {
    objective: "跟随反向梯度完成一次参数更新，并验证新损失。",
    steps: [
      { title: "保存前向结果", description: "先得到所有中间值与损失。反向传播需要使用这一次前向计算的数值。" },
      { title: "按链式法则回传", description: "从损失向输入方向传播，上游梯度乘以局部导数；如果存在多条路径，需要累加贡献。" },
      { title: "再执行参数更新", description: "收集权重和偏置的梯度后，用 SGD 更新，再重新前向计算，比较新旧损失。" },
    ],
    formula: String.raw`\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial a}(1-a^2)x_1`,
    formulaNote: "这里 a = tanh(z)。反向传播计算梯度，SGD 使用梯度更新参数，两者各自完成不同步骤。",
    misconception: "算出梯度并不意味着参数已经改变。只有应用更新，下一次前向计算才会使用新权重。",
    quiz: {
      question: "反向传播刚计算完全部梯度，还没有执行 SGD。此时权重发生变化了吗？",
      options: ["已经自动变小", "已经变为梯度值", "没有，仍需执行参数更新"],
      answer: 2,
      explanation: "梯度描述损失对参数的变化率。优化器执行 w ← w − η∇L 后，权重才真正改变。",
    },
  },
  xor: {
    objective: "理解隐藏层的作用，观察非线性网络如何学会 XOR。",
    steps: [
      { title: "读取四个样本", description: "XOR 在两个输入不同时输出 1，相同时输出 0；单条直线无法把两类完全分开。" },
      { title: "隐藏层改变表示", description: "2→4→1 网络先用四个 Tanh 隐藏单元变换输入，再用输出层得到正类概率。" },
      { title: "聚合样本梯度", description: "用四个样本的平均二元交叉熵训练。观察单样本梯度、平均梯度和决策边界如何对应。" },
    ],
    formula: String.raw`\begin{gathered}h=\tanh(XW_1+b_1) \\ p=\operatorname{sigmoid}(hW_2+b_2)\end{gathered}`,
    formulaNote: "分类使用 0.5 阈值；训练保留连续输出的信息，并从 logits 稳定计算二元交叉熵。",
    misconception: "增加层数但全部使用线性激活，仍等价于一个线性变换。隐藏层的非线性是解决 XOR 的关键。",
    quiz: {
      question: "XOR 的两个输入分别为 1 和 0，目标输出是什么？",
      options: ["0，因为存在一个 0", "1，因为两个输入不同", "无法确定"],
      answer: 1,
      explanation: "XOR 表示异或：输入不同为 1，相同为 0。四个目标按 (0,0)、(0,1)、(1,0)、(1,1) 排列是 0、1、1、0。",
    },
  },
  "mnist-data": {
    objective: "建立独立的数据划分，避免用测试答案影响模型选择。",
    steps: [
      { title: "识别样本与标签", description: "每张 MNIST 图像包含 28 × 28 灰度像素，标签是 0–9 中的一个数字。" },
      { title: "划分训练与验证", description: "从官方训练数据中按类别分层划分验证集，用随机种子使相同配置可复现。" },
      { title: "保留独立测试集", description: "训练集用于更新参数，验证集用于比较配置与选模，官方测试集用于最后检验。" },
    ],
    formula: String.raw`\begin{gathered}\mathcal{D}_{\mathrm{train}}\cap\mathcal{D}_{\mathrm{val}}=\varnothing \\ \text{测试集独立保留}\end{gathered}`,
    formulaNote: "完整官方训练源有 60,000 张；验证比例为 10% 且不限制训练子集时，划分为 54,000 / 6,000 张，另有 10,000 张测试图。",
    misconception: "反复查看测试表现并据此调参，会让测试信息间接进入模型选择，使最终分数失去独立评估的意义。",
    quiz: {
      question: "应该使用哪个集合的指标来挑选最佳训练轮次？",
      options: ["验证集", "测试集", "把三种数据合并后评估"],
      answer: 0,
      explanation: "验证集负责选模。本项目按验证损失保存最佳权重，测试集用于训练与模型选择完成后的独立评估。",
    },
  },
  "mnist-train": {
    objective: "训练 MLP 与 CNN，使用验证曲线判断模型是否真正改善。",
    steps: [
      { title: "保持比较条件一致", description: "比较两种网络前，尽量固定数据划分、训练子集、随机种子与训练设置。" },
      { title: "按批次更新参数", description: "图像先标准化，再前向计算 logits、交叉熵与梯度，由优化器更新模型权重。" },
      { title: "每轮检查验证表现", description: "一轮结束后切换评估模式，计算独立验证指标；保存验证损失最低时的权重。" },
    ],
    formula: String.raw`x=\frac{\mathrm{pixels}/255-0.1307}{0.3081}`,
    formulaNote: "本项目使用相同的 MNIST 标准化供训练、验证和推理使用。一个 Epoch 表示完成一次当前训练集的遍历。",
    misconception: "训练损失持续下降不保证泛化能力更好。如果验证损失上升，需要检查过拟合，而不是只增加训练轮次。",
    quiz: {
      question: "训练损失下降，但验证损失持续上升，最需要警惕什么？",
      options: ["数据一定已损坏", "模型已经完全学会所有数字", "模型可能开始过拟合"],
      answer: 2,
      explanation: "模型可能越来越贴合训练样本，却不能改善未参与参数更新的数据表现。应以验证指标辅助判断，并保留最佳权重。",
    },
  },
  "mnist-evaluation": {
    objective: "用混淆矩阵与真实错例解释准确率背后的模型行为。",
    steps: [
      { title: "加载最佳检查点", description: "使用按验证集选出的最佳权重，并保持与训练一致的图像标准化方式。" },
      { title: "读取混淆矩阵", description: "本项目中行代表真实标签，列代表模型预测。对角线上是正确预测，其余位置都是错例。" },
      { title: "回到具体图像", description: "点击非对角单元格，查看对应数字及各类分数，区分难辨笔迹和系统性混淆。" },
    ],
    formula: String.raw`\mathrm{Accuracy}=\frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}}`,
    formulaNote: "某类 Precision 衡量预测成该类时有多少正确；Recall 衡量真实属于该类的样本有多少被找出。",
    misconception: "总体准确率不能说明每个数字都表现良好。相同准确率可能对应完全不同的类别混淆与错误分布。",
    quiz: {
      question: "混淆矩阵第 3 行、第 8 列（数字标签）有错例，代表什么？",
      options: ["真实是 8，预测为 3", "真实是 3，预测为 8", "模型同时预测了 3 和 8"],
      answer: 1,
      explanation: "当前矩阵使用真实标签作行、预测标签作列。因此该单元格统计真实为 3、被预测为 8 的样本。",
    },
  },
  handwriting: {
    objective: "把自己的笔迹送入真实模型，并正确理解预测分数。",
    steps: [
      { title: "把笔迹变成输入", description: "手绘图先裁剪、缩放和居中，转换成 28 × 28 灰度图；官方 MNIST 样本保留原始像素。" },
      { title: "复用训练标准化", description: "根据检查点保存的均值与标准差处理图像，加载相同网络结构与最佳权重进行推理。" },
      { title: "比较十类分数", description: "Softmax 将输出转换为十类相对分数，最大值对应预测数字；可比较 MLP 和 CNN 的结果。" },
    ],
    formula: String.raw`\begin{gathered}p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)} \\ \hat{y}=\operatorname*{arg\,max}_i p_i\end{gathered}`,
    formulaNote: "先检查预处理预览中的数字是否清楚且居中，再判断模型输出。输入分布与训练样本越不同，预测越可能不稳定。",
    misconception: "Softmax 的最高分不是正确率保证。模型遇到奇怪笔迹或训练分布外的输入，也可能给出很高分；本应用会在推理前拒绝空白输入。",
    quiz: {
      question: "模型给数字 7 的 Softmax 分数为 0.98，最准确的理解是什么？",
      options: ["模型对当前输入更偏向类别 7，仍可能预测错误", "这个数字必然是 7", "模型在所有手写数字上的准确率是 98%"],
      answer: 0,
      explanation: "这是当前输入在十个类别中的相对预测分数，不是总体测试准确率，也不保证这一次预测正确。",
    },
  },
  deployment: {
    objective: "把最佳权重、预处理与推理接口连接成可验证的应用。",
    steps: [
      { title: "导出可加载模型", description: "从已完成的训练中导出最佳检查点，让推理服务能够读取权重和必要的模型配置。" },
      { title: "打包应用依赖", description: "使用项目的 Docker Compose 配置启动前端、API 与模型文件，确保服务能够找到所需资源。" },
      { title: "验证完整请求链路", description: "检查健康接口与模型列表，再提交真实图像请求；运行 smoke 脚本验证两种模型的预测链路。" },
    ],
    formula: String.raw`\begin{gathered}\text{图像请求}\longrightarrow\text{预处理} \\ \longrightarrow\text{模型推理}\longrightarrow\text{类别与分数}\end{gathered}`,
    formulaNote: "GET /api/inference/models 查询模型；POST /api/inference/predict 提交图像并返回预测。",
    misconception: "服务启动成功不等于模型可用。还应确认权重已加载、请求格式正确，并用真实图像验证返回结果。",
    quiz: {
      question: "哪项检查最能证明推理服务的核心链路可用？",
      options: ["容器显示正在运行", "首页能够打开", "提交真实图像并收到有效预测结果"],
      answer: 2,
      explanation: "真实预测请求覆盖输入校验、预处理、模型加载和推理输出。首页或健康检查成功，只验证了这条链路的一部分。",
    },
  },
};
