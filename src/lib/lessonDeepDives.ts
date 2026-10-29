export interface LessonDeepDive {
  example: { title: string; steps: readonly string[]; tex: string; takeaway: string };
  observations: readonly { action: string; expected: string }[];
  terms: readonly { term: string; definition: string }[];
  connection: string;
  next: string;
}

/** Worked examples are teaching calculations, not recorded model outputs. */
export const lessonDeepDives: Record<string, LessonDeepDive> = {
  data: {
    example: { title: "一个像素的两种表示", steps: ["选中第 3 行、第 4 列。采用从 0 开始的索引，它的位置是行 2、列 3。", "将灰度设为 128，除以 255 得到约 0.50196。亮度含义没有变，数值范围变了。", "行优先展平时，索引是 2 × 8 + 3 = 19；矩阵和向量中的高亮值应相同。"], tex: String.raw`x' = \frac{128}{255}\approx0.50196,\qquad i=2\times8+3=19`, takeaway: "先核对位置，再核对数值。形状转换与数值缩放是两件独立的事。" },
    observations: [{ action: "关闭归一化，再重新打开。", expected: "矩阵与向量同时在 128 和约 0.502 之间变化，元素数量始终是 64。" }, { action: "只修改一个像素。", expected: "只有对应位置的输入改变；均值会变化，其他像素不会被自动改写。" }],
    terms: [{ term: "张量", definition: "按多个轴组织的数值数组。向量、矩阵分别是一维、二维张量。" }, { term: "特征", definition: "送入模型的输入数值。在本实验中，每个像素就是一个特征。" }],
    connection: "下一节把一个输入数值送入最简单的模型，观察输入怎样变成预测。", next: "linear",
  },
  linear: {
    example: { title: "一个样本算对，不等于整体拟合好", steps: ["取实验中的样本 x = 2，它的真实标签由 y = 2x + 0.5 得到，是 4.5。", "设置 w = 1、b = 0.5，预测为 2.5，误差是 −2，该样本平方误差为 4。", "全部五个样本的 MSE 是 2；改为 w = 2、b = 0.5，五个样本的误差才都归零。"], tex: String.raw`\hat y=1\times2+0.5=2.5,\qquad (2.5-4.5)^2=4`, takeaway: "单样本误差与平均损失不能混用。比较模型时要使用同一组样本。" },
    observations: [{ action: "保持 b = 0.5，将 w 从 1 调到 2。", expected: "直线斜率接近目标，五条误差线段缩短，MSE 降到 0。" }, { action: "保持 w = 2，仅改变 b。", expected: "所有预测一起上移或下移，残差相同；偏置无法补偿错误的斜率。" }],
    terms: [{ term: "残差", definition: "这里指预测减去真实标签，有正负号。平方后就不再保留方向。" }, { term: "参数", definition: "控制模型行为的可调整数值。此处是权重 w 与偏置 b。" }],
    connection: "给加权和加上非线性激活，就得到下一节的单个神经元。", next: "neuron",
  },
  neuron: {
    example: { title: "沿两条连接追踪一次输出", steps: ["使用默认输入与权重：第一路贡献 0.6 × 0.8 = 0.48，第二路贡献 0.9 × (−0.4) = −0.36。", "加入偏置 0.2，得到加权和 z = 0.32。选择 Sigmoid 后，输出约为 0.5793。", "仅把 w₁ 改为 −0.8，z 变成 −0.64，Sigmoid 输出约为 0.3452。"], tex: String.raw`z=0.48-0.36+0.2=0.32,\qquad \sigma(z)=\frac{1}{1+e^{-z}}`, takeaway: "先看每路贡献与加权和，再看激活输出。输出曲线的横轴是输入，激活函数图的横轴是 z。" },
    observations: [{ action: "选择 Sigmoid，横轴选输入 1，将 w₁ 调为负数。", expected: "曲线随输入增大而下降；两端接近饱和时，输出变化变小。" }, { action: "把当前参数设为参考，再只改变偏置。", expected: "实线与参考虚线的位置分开。固定的另一路输入仍通过其乘积影响曲线。" }],
    terms: [{ term: "饱和", definition: "Sigmoid 或 Tanh 在较大正、负输入处趋于平坦，局部导数接近 0。" }, { term: "激活阈值", definition: "例如 ReLU 在 z = 0 处改变行为；偏置会改变到达这一位置所需的输入。" }],
    connection: "有了预测，还需要一个标量来衡量预测偏离目标的程度。", next: "loss",
  },
  loss: {
    example: { title: "损失相同，梯度为什么相反", steps: ["固定真实标签 y = 2。预测为 1 时，误差 −1，损失 1，梯度 −2。", "预测为 3 时，误差 +1，损失仍为 1，但梯度变为 +2。", "预测为 2 时位于最低点，损失和梯度都为 0。梯度描述的是当前位置的变化率。"], tex: String.raw`L(1)=L(3)=1,\qquad L'(1)=-2,\quad L'(3)=2`, takeaway: "损失回答错了多少，梯度帮助判断往哪个方向改变预测能降低损失。" },
    observations: [{ action: "固定标签，移动预测值到目标两侧相同距离。", expected: "标记点的高度相同，梯度符号相反。" }, { action: "保持预测不变，只移动真实标签。", expected: "抛物线的最低点跟随标签移动。观察横轴刻度，避免把坐标范围变化当成函数形状变化。" }],
    terms: [{ term: "标量损失", definition: "用单个数汇总预测质量，供优化算法使用；不同任务可使用不同损失。" }, { term: "导数", definition: "函数对某个变量的局部变化率，不能直接当作该变量的新数值。" }],
    connection: "下一节将梯度乘以学习率，执行一次实际参数更新。", next: "gradient",
  },
  gradient: {
    example: { title: "手算第一步梯度下降", steps: ["本实验固定偏置，只训练 w，损失是 L(w) = 2(w − 2)²。初始 w = −2 时，损失为 32。", "当前位置的梯度是 4(w − 2) = −16，学习率取 0.1。", "更新后 w = −2 − 0.1 × (−16) = −0.4，重新计算损失得到 11.52。"], tex: String.raw`w_{1}=-0.4,\qquad L(w_1)=2(-0.4-2)^2=11.52`, takeaway: "先用旧参数计算梯度，再更新参数，最后重新计算损失；三个步骤不可混淆。" },
    observations: [{ action: "分别使用 0.1、0.5、0.55 的预设，重置后比较相同更新次数。", expected: "本实验中分别表现为收敛、等幅振荡和逐步发散；这三个数不是其他任务的通用阈值。" }, { action: "查看训练历史，再导出 CSV。", expected: "记录的是每次真实更新后的损失，可以与当前权重对应核对。" }],
    terms: [{ term: "学习率", definition: "缩放梯度更新步长的系数；过大可能越过低损失区域。" }, { term: "收敛", definition: "这里指参数接近最低点、损失趋于稳定，不意味着任何训练都能找到全局最优解。" }],
    connection: "更复杂模型中的梯度，来自计算图上每一个局部运算的组合。", next: "graph",
  },
  graph: {
    example: { title: "激活节点怎样缩放梯度", steps: ["默认参数产生 z = 0.32，Tanh 输出 a 约为 0.3095。", "Tanh 在这里的局部导数 1 − a² 约为 0.9042。", "本节使用带 ½ 的平方损失，目标为 0.8，上游梯度 a − y 约为 −0.4905；经过激活后，z 的梯度约为 −0.4435。"], tex: String.raw`\frac{\partial L}{\partial z}=(a-y)(1-a^2)\approx-0.4435`, takeaway: "边上的局部导数和节点收到的总梯度含义不同；总梯度还包含后续运算的影响。" },
    observations: [{ action: "切换前向数值与反向梯度，点击 a 节点。", expected: "同一个节点在两种模式下显示不同量；详情可帮助区分数值和导数。" }, { action: "增加偏置，使 Tanh 输出接近 1。", expected: "激活的局部导数减小。总梯度还受上游误差影响，要结合详情观察。" }],
    terms: [{ term: "拓扑顺序", definition: "保证一个运算所需的输入先被算出的顺序；反向传播按相反的依赖方向进行。" }, { term: "链式法则", definition: "复合函数的变化率由沿路径的局部变化率相乘得到；多条路径的贡献需要相加。" }],
    connection: "下一节将计算图分步播放，直到用这些梯度更新权重。", next: "backprop",
  },
  backprop: {
    example: { title: "从 z 的梯度回到第一路权重", steps: ["沿用默认参数，z 的梯度约为 −0.4435。", "由于 z 对 w₁ 的局部导数是输入 x₁ = 0.6，w₁ 的梯度约为 −0.2661。", "学习率 0.1 时，w₁ 从 0.8 增加到约 0.8266。偏置与第二路权重也要使用同一次前向结果计算的梯度一起更新。"], tex: String.raw`\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial z}x_1,\qquad w_1'=0.8-0.1(-0.2661)`, takeaway: "反向传播负责计算梯度，优化器负责更新参数。应用后要重新前向计算，不能继续沿用旧梯度。" },
    observations: [{ action: "在更新阶段先核对表格，再点击应用参数更新。", expected: "点击前只是更新预览；点击后参数才改变，并出现已应用次数与新旧损失。" }, { action: "调高学习率后再比较更新前后损失。", expected: "负梯度给出局部下降方向，但有限步长不保证一定降低损失。" }],
    terms: [{ term: "梯度累加", definition: "某个量影响损失的路径不止一条时，需要将所有路径的贡献求和。" }, { term: "SGD", definition: "按梯度执行参数更新的优化方法。本图每次展示一个样本的更新。" }],
    connection: "将相同机制用于多个隐藏单元和四个样本，就能训练 XOR 网络。", next: "xor",
  },
  xor: {
    example: { title: "分类正确之后为什么还要看损失", steps: ["按 (0,0)、(0,1)、(1,0)、(1,1) 排列，目标是 0、1、1、0。", "假设四个输出概率为 0.1、0.9、0.9、0.1，以 0.5 为阈值可全部分类正确。", "每个样本的 BCE 都是 −ln(0.9)，平均约为 0.1054。这是教学概率示例，页面中的实际概率取决于训练。"], tex: String.raw`\mathrm{BCE}=-\frac1N\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)]`, takeaway: "准确率只检查阈值两侧，交叉熵还衡量输出分数；四个预测正确不代表损失已经很低。" },
    observations: [{ action: "训练前后比较决策边界与四个训练样本。", expected: "隐藏层的非线性变换使网络能分开单条直线无法分开的两类。" }, { action: "比较单样本梯度与四样本平均梯度。", expected: "更新基于平均梯度，改善整体目标不保证每个样本都在同一步改善。" }],
    terms: [{ term: "决策边界", definition: "本实验中输出概率等于 0.5 的位置，分隔两种预测类别。" }, { term: "隐藏表示", definition: "隐藏层对原始输入的变换结果，使输出层能够组合更复杂的分类规则。" }],
    connection: "XOR 的样本很少。下一阶段使用真实图像，并先建立独立的数据划分。", next: "mnist-data",
  },
  "mnist-data": {
    example: { title: "三份数据分别用在哪里", steps: ["官方训练源包含 60,000 张图。验证比例取 10%，得到训练 54,000 张、验证 6,000 张。", "另有 10,000 张官方测试图，独立保留，不参与权重更新或最佳轮次选择。", "若限制训练子集，只减少实际用于训练的数量，不把剩余图像自动并入测试集。"], tex: String.raw`60000=54000+6000,\qquad N_{\mathrm{test}}=10000`, takeaway: "先确定划分，再比较模型。测试集不能反复承担调参和选模的工作。" },
    observations: [{ action: "数据就绪后，改变验证比例并应用划分。", expected: "训练与验证数量联动变化，独立测试集数量保持不变。" }, { action: "使用相同种子与配置重新应用。", expected: "划分应可复现；切换集合查看样本时，要核对其训练、验证或测试标签。" }],
    terms: [{ term: "分层抽样", definition: "按类别分别进行划分，尽量让各集合保留原始类别比例。" }, { term: "数据泄漏", definition: "本不应参与训练或模型选择的信息影响了决策，使评估过于乐观。" }],
    connection: "数据角色明确后，才有条件公平比较 MLP 与 CNN 的训练表现。", next: "mnist-train",
  },
  "mnist-train": {
    example: { title: "一轮训练不等于一次更新", steps: ["假设当前实际训练子集为 1,000 张，batch size 为 64，且保留最后一个不满批次。", "前 15 批处理 960 张，最后一批处理 40 张，因此一轮有 16 次参数更新。", "轮末计算验证指标，再按验证损失保存最佳检查点；最后一轮不一定是最佳轮次。"], tex: String.raw`\mathrm{steps\ per\ epoch}=\left\lceil\frac{1000}{64}\right\rceil=16`, takeaway: "比较训练记录时，同时核对样本数、批次大小、轮数和验证条件。这里的 1,000 张是计算示例。" },
    observations: [{ action: "查看训练和验证曲线的走向。", expected: "训练损失下降而验证损失持续升高时，需要警惕过拟合。单轮小幅波动不能单独证明过拟合。" }, { action: "保持划分、子集和种子一致，分别比较 MLP 与 CNN。", expected: "比较验证指标、参数量和时间；模型名称本身不能保证在每次设置下都更好。" }],
    terms: [{ term: "Epoch", definition: "完成一次当前实际训练集的遍历，不是只看一个批次。" }, { term: "检查点", definition: "保存的模型权重及相关配置。本项目依据验证损失挑选最佳权重。" }],
    connection: "模型选择完成后，下一节用独立测试集分析它在哪些样本上出错。", next: "mnist-evaluation",
  },
  "mnist-evaluation": {
    example: { title: "同一张矩阵读出三种指标", steps: ["设一个十类教学矩阵中，仅 C₃₃=45、C₃₈=5、C₈₃=10、C₈₈=40，其余位置为 0，共 100 个样本。", "对角线正确数 45 + 40 = 85，准确率为 85%。", "对数字 8，Precision = 40/(5+40) ≈ 88.9%，Recall = 40/(10+40) = 80%。这不是当前模型的实测分数。"], tex: String.raw`P_8=\frac{40}{45},\qquad R_8=\frac{40}{50},\qquad \mathrm{Accuracy}=\frac{85}{100}`, takeaway: "Precision 看预测成某类的列，Recall 看真实属于某类的行。本项目行是真实标签，列是预测。" },
    observations: [{ action: "点击混淆矩阵的非对角位置。", expected: "错例列表按真实标签和预测标签筛选，便于找到一种具体混淆。" }, { action: "对比类别指标与错例图像。", expected: "少数困难类别可能被总体准确率掩盖；先核对样本数量，再解释百分比。" }],
    terms: [{ term: "支持数", definition: "测试集中真实属于某类别的样本数，解释该类指标时应一并查看。" }, { term: "混淆矩阵", definition: "将真实类别与预测类别交叉统计的计数表，用来定位错误分布。" }],
    connection: "测试集上的成绩还不能保证任意笔迹都可靠。下一节用自己的输入检验模型。", next: "handwriting",
  },
  handwriting: {
    example: { title: "十个分数是怎样归一化的", steps: ["取一个十分类教学例子：数字 7 的 logit 为 4，其余九类的 logit 都为 0。", "Softmax 后，数字 7 的分数是 e⁴/(e⁴+9)，约为 0.8585，十类分数之和为 1。", "这个数仅表示模型对当前输入的相对偏好，不能直接当成总体准确率或正确性的保证。"], tex: String.raw`p_7=\frac{e^4}{e^4+9}\approx0.8585`, takeaway: "先检查预处理图像，再解释输出分数。这个 logit 例子用于理解计算，并非页面识别结果。" },
    observations: [{ action: "先载入官方样本识别，再手写同一个数字。", expected: "比较预处理预览与分数；偏移、笔画粗细和书写风格都会改变模型输入。" }, { action: "更换模型、编辑画布或改变热力图选项。", expected: "旧预测失效，需要重新识别。同一输入重复识别时则可保留上次确认结果等待刷新。" }],
    terms: [{ term: "Logit", definition: "Softmax 之前的模型输出分数，不要求非负，也不要求总和为 1。" }, { term: "输入敏感度", definition: "某个类别输出对输入像素的局部梯度大小，反映局部敏感性，不能单独证明因果关系。" }],
    connection: "模型、预处理与输出协议需要一起交付，才能形成可复现的推理服务。", next: "deployment",
  },
  deployment: {
    example: { title: "逐层核对一次推理请求", steps: ["先读取模型目录，取得真实 model_id，而不是凭模型名称猜测标识符。", "提交原始 MNIST 图时，mode 设为 mnist，pixels 必须是 28 × 28 的灰度矩阵，共 784 个数。", "核对返回的预处理图、模型标识、预测类别和十类分数；健康接口成功只说明服务可连接。"], tex: String.raw`28\times28=784,\qquad \sum_{i=0}^{9}p_i\approx1`, takeaway: "输入格式、预处理、权重和输出协议要一起验证；正常启动不等于完整推理链路已经可用。" },
    observations: [{ action: "比较模型目录中的来源、最佳轮次与权重摘要。", expected: "能够区分发布模型包和训练记录中的模型，避免比较错检查点。" }, { action: "按照部署页给出的命令执行构建和 smoke 检查。", expected: "smoke 检查发送真实样本并校验输出；这是需要在本机实际执行的操作，阅读命令本身不会启动服务。" }],
    terms: [{ term: "推理契约", definition: "调用双方共同遵守的输入、预处理、模型选择与输出格式要求。" }, { term: "可复现部署", definition: "使同一份模型及其依赖和预处理配置在目标环境中被一致地加载与运行。" }],
    connection: "回到学习总览，选择尚未完成的实验，或进入探索工坊组合已经学过的知识。", next: "overview",
  },
};
