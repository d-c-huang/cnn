import { grpoTransfer, grpoRecall, grpoPredictions } from "./grpoContent.ts";
import { dpoTransfer, dpoRecall, dpoPredictions } from "./dpoContent.ts";
import { languageTransfer, languageRecall, languagePredictions } from "./languageContent.ts";
import { transformerTransfer, transformerRecall, transformerPredictions } from "./transformerContent.ts";
import { loraTransfer, loraRecall, loraPredictions } from "./loraContent.ts";
import { sftTransfer, sftRecall, sftPredictions } from "./sftContent.ts";

export interface StudyQuestion {
  question: string;
  options: readonly [string, string, string];
  answer: 0 | 1 | 2;
  explanation: string;
  whyWrong: readonly [string, string, string];
}

export const transferQuestions: Record<string, StudyQuestion> = {
  ...languageTransfer,
  ...transformerTransfer,
  ...loraTransfer,
  ...sftTransfer,
  ...dpoTransfer,
  ...grpoTransfer,
  data: { question: "将 28 × 28 图像展平并除以 255，灰度为 51 的像素变成多少？", options: ["0.2，向量仍有 784 项", "51，向量只有 28 项", "0.2，向量只有 255 项"], answer: 0, explanation: "51/255 = 0.2，28×28 = 784。数值缩放与形状变换互不改变元素数量。", whyWrong: ["", "展平不会减少像素，除以 255 也会改变数值。", "255 是缩放分母，元素数量由图像尺寸决定。"] },
  linear: { question: "x = 3，w = −2，b = 1 时，预测值是多少？", options: ["−5", "7", "−6"], answer: 0, explanation: "预测为 wx+b = −2×3+1 = −5。", whyWrong: ["", "负权重的符号不能丢掉，−2×3 是 −6。", "还需要加上偏置 1。"] },
  neuron: { question: "同一个 z = −2 分别输入 ReLU 和 Sigmoid，哪一项正确？", options: ["两者都等于 0", "ReLU 等于 0，Sigmoid 大于 0", "Sigmoid 等于 −2"], answer: 1, explanation: "ReLU 截断负值；Sigmoid 在有限输入处始终介于 0 与 1。", whyWrong: ["Sigmoid 只是接近 0，不会在有限 z 处严格等于 0。", "", "Sigmoid 是非线性变换，不是恒等函数。"] },
  loss: { question: "MSE 单样本中 y = 2、预测为 4，损失和对预测的梯度分别是多少？", options: ["4 和 4", "4 和 −4", "2 和 4"], answer: 0, explanation: "L = (4−2)² = 4；dL/dŷ = 2(4−2) = 4。此处没有 ½ 系数。", whyWrong: ["", "预测偏大时梯度为正，减去梯度才会向标签靠近。", "有符号误差为 2，平方损失是 4。"] },
  gradient: { question: "L(w)=2(w−2)²，w=−2、η=0.1，更新一次后的 w 是多少？", options: ["−3.6", "−0.4", "1.6"], answer: 1, explanation: "梯度为 4(−2−2)=−16；w′=−2−0.1×(−16)=−0.4。", whyWrong: ["这是把负梯度继续加到负方向；更新应减去梯度。", "", "1.6 是参数变化量，更新后的参数还要加上原值 −2。"] },
  graph: { question: "上游梯度为 −2，局部导数为 0.25，沿唯一这条边回传多少？", options: ["−0.5", "−1.75", "−8"], answer: 0, explanation: "链式法则相乘：−2×0.25=−0.5。", whyWrong: ["", "沿同一路径应相乘，多个路径汇合才相加。", "不能用上游梯度除以局部导数。"] },
  backprop: { question: "权重 0.8、损失梯度 −0.3、学习率 0.1，应用更新后是多少？", options: ["0.77", "0.83", "−0.03"], answer: 1, explanation: "w′=0.8−0.1×(−0.3)=0.83。", whyWrong: ["梯度为负时，减去它会让权重增大。", "", "这只是带符号的步长，不能替代整个参数。"] },
  xor: { question: "四个样本的分类都正确，是否说明 BCE 已经低于 0.2？", options: ["必然低于", "不一定，还要看概率离标签有多近", "BCE 必然为 0"], answer: 1, explanation: "0.49 与 0.51 也可在阈值两侧分类正确，但 BCE 仍可较高。", whyWrong: ["准确率只看阈值两侧，不能约束到 0.2。", "", "分类正确并不意味着目标类别的概率为 1。"] },
  "mnist-data": { question: "从 60,000 张训练源中取 20% 验证，独立测试集仍为 10,000 张，三集合数量是？", options: ["48,000 / 12,000 / 10,000", "48,000 / 10,000 / 12,000", "60,000 / 12,000 / 10,000"], answer: 0, explanation: "验证从训练源划出；测试源独立保留，不能参与调参。", whyWrong: ["", "验证比例不会改变官方独立测试集。", "验证样本必须从训练中移出，否则两个集合重叠。"] },
  "mnist-train": { question: "三轮验证损失为 0.20、0.15、0.18，应保留哪轮最佳权重？", options: ["第 3 轮，因为最新", "第 1 轮，因为最早", "第 2 轮，因为验证损失最低"], answer: 2, explanation: "本项目按验证损失最低保存最佳检查点，不用测试集来选择。", whyWrong: ["后面的轮次不保证泛化更好。", "轮次顺序不是选择检查点的依据。", ""] },
  "mnist-evaluation": { question: "真实数字 5 共 100 张，其中 80 张预测为 5；另有 20 张其他数字被预测为 5。该类 Recall 是多少？", options: ["100%", "80%", "20%"], answer: 1, explanation: "Recall = TP/(TP+FN)=80/100=80%。其他数字的误报影响 Precision 的分母。", whyWrong: ["真实数字 5 中有 20 张被漏掉，召回率不是 100%。", "", "20% 是漏检比例，召回率统计被找出的比例。"] },
  handwriting: { question: "把手写笔迹平移后模型预测改变，先检查哪项？", options: ["预处理后的 28×28 图像是否完整居中", "直接把最高分当成正确率", "使用测试集重新调权重"], answer: 0, explanation: "先核对裁剪、缩放与居中后的实际输入，再讨论模型行为。", whyWrong: ["", "最高分只描述当前输出，不证明预测正确。", "独立测试集不能用于重新训练权重。"] },
  deployment: { question: "首页正常，但预测接口提示模型文件不存在，下一步应检查什么？", options: ["修改首页标题", "只重复健康检查", "模型路径、manifest 与实际权重是否匹配"], answer: 2, explanation: "网页资源正常并不验证模型加载；需沿失败的推理链路核对模型包。", whyWrong: ["标题不影响后端权重加载。", "健康接口无法替代真实的模型加载与推理检查。", ""] },
};

/** Each distractor describes a different misconception in the existing first question. */
export const recallFeedback: Record<string, readonly [string, string, string]> = {
  ...languageRecall,
  ...transformerRecall,
  ...loraRecall,
  ...sftRecall,
  ...dpoRecall,
  ...grpoRecall,
  data: ["8 只是单边长度，展平保留每一行的所有像素。", "展平不是两条边相加，而是行数乘列数。", ""],
  linear: ["", "斜率由权重决定，偏置不改变斜率。", "偏置只是加上常数，不会清空输入贡献。"],
  neuron: ["ReLU 的负区间被截断，不能直接保留 z。", "", "ReLU 不是绝对值函数。"],
  loss: ["两者距标签都为 1，平方损失都是 1。", "", "两者误差分别为 −1 和 +1，梯度符号相反。"],
  gradient: ["正梯度表示局部上坡方向，SGD 应减去它。", "正学习率乘正梯度非零，权重不会保持不变。", ""],
  graph: ["", "a² 接近 1，因此 1−a² 接近 0，而不是 1。", "Tanh 的局部导数 1−a² 最大为 1，不可能接近 2。"],
  backprop: ["反向传播只求梯度，权重尚未更新；更新也不保证总是变小。", "梯度是变化率，不能直接替代权重数值。", ""],
  xor: ["异或不是与运算；只要两个输入不同，目标就是 1。", "", "两个二值输入已知，异或的目标可以唯一确定。"],
  "mnist-data": ["", "测试集必须保留给最终独立检验，不能拿来选择轮次。", "合并三个集合会让训练数据影响评估，失去验证的独立性。"],
  "mnist-train": ["仅凭训练与验证趋势不能断定数据损坏。", "训练集拟合更好不代表所有新样本都正确。", ""],
  "mnist-evaluation": ["当前矩阵约定真实标签作行，预测标签作列。", "", "一个单元格统计一个真实标签与预测标签组合。"],
  handwriting: ["", "高分输出仍可能来自误识别或分布外输入。", "这是一个输入的分数，不是整个测试集的准确率。"],
  deployment: ["容器存活没有验证权重加载和模型输出。", "首页只验证静态资源服务。", ""],
};

export const predictionPrompts: Record<string, { question: string; options: string[]; observation: string }> = {
  ...languagePredictions,
  ...transformerPredictions,
  ...loraPredictions,
  ...sftPredictions,
  ...dpoPredictions,
  ...grpoPredictions,
  data: { question: "打开归一化，图像的相对明暗会改变吗？", options: ["会改变", "保持不变", "不确定"], observation: "把灰度设为 128，比较图像与转换后的约 0.502；记录形状与数值范围分别发生什么变化。" },
  linear: { question: "保持 w 不变、增大 b，整条直线会怎样？", options: ["上移", "变陡", "下移"], observation: "只调偏置，对照整条直线和五个样本残差，不要同时改变权重。" },
  loss: { question: "预测值从标签左侧移到等距右侧，梯度如何变化？", options: ["符号反转", "保持不变", "总为零"], observation: "固定标签，比较两侧等距位置的损失高度和梯度符号。" },
  graph: { question: "Tanh 接近饱和时，局部导数会如何变化？", options: ["接近零", "接近一", "持续增大"], observation: "调大偏置，选择激活节点，比较局部导数与完整梯度。" },
  backprop: { question: "应用一次参数更新后，还能直接用旧梯度继续更新吗？", options: ["可以", "需要重新前向和反向", "只需换学习率"], observation: "执行一次更新，核对新参数、新损失和下一次求导的参数版本。" },
  xor: { question: "分类准确率到 100% 时，损失一定为零吗？", options: ["一定", "不一定", "损失不再有意义"], observation: "训练到四样本分类正确后，继续比较概率与 BCE；保持其他条件不变。" },
  "mnist-data": { question: "增加验证比例，独立测试集数量会改变吗？", options: ["增加", "减少", "保持不变"], observation: "应用新的划分后，对比三集合真实样本数量。" },
  "mnist-train": { question: "增加训练轮次，最佳验证损失一定持续下降吗？", options: ["一定", "不一定", "与训练无关"], observation: "比较每轮训练与验证曲线，以及最佳检查点所在轮次。" },
  "mnist-evaluation": { question: "总体准确率相同的两个模型，各类召回率也相同吗？", options: ["一定相同", "可能不同", "无法计算召回率"], observation: "选择已完成的真实评估，核对混淆矩阵每行与分类别指标。" },
  handwriting: { question: "最高类别分数很高，就能保证这次预测正确吗？", options: ["能", "不能", "只对 CNN 能"], observation: "提交笔迹后核对预处理图像、真实笔迹与十类分数；分数不是正确性证明。" },
  deployment: { question: "健康接口成功，能证明真实图像推理可用吗？", options: ["能", "还需真实预测请求", "只需打开首页"], observation: "查看可用模型，提交图像，逐步核对输入、模型加载与返回结果。" },
};
