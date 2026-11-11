import type { LessonInsight } from "./lessonInsights";
import type { LessonDeepDive } from "./lessonDeepDives";
import type { StudyQuestion } from "./studyQuestions";

export const languageInsights: Record<string, LessonInsight> = {
  "language-data": {
    objective: "把原文转换为字符 ID，说明输入、右移标签与验证集的关系。",
    steps: [
      { title: "先划分文本", description: "按原文顺序保留后 20% 作验证，只用前 80% 建词表。两部分分别构造样本，不跨边界。" },
      { title: "每个字符一个索引", description: "中文、标点、空格和换行都各自成为 token；词表外的验证字符映射到 UNK。真实大模型通常使用子词分词，token 不一定等于字符。" },
      { title: "从文本本身取标签", description: "例如“小猫看花”生成“小→猫、猫→看、看→花”三个样本。模型在看到当前字符时预测下一字符，不能把目标当输入。" },
    ],
    formula: String.raw`x_t=\mathrm{ID}(c_t),\qquad y_t=\mathrm{ID}(c_{t+1})`,
    formulaNote: "一段有 N 个字符的文本产生 N−1 个相邻字符对。ID 的数值大小没有语义含义。",
    misconception: "用全部文本建立词表会提前看到验证字符。本实验先划分再建词表；小语料和字符重叠本身也不能代表广泛泛化。",
    quiz: { question: "“小猫看花”按当前字符预测下一字符，可以组成多少对？", options: ["3 对", "4 对", "1 对"], answer: 0, explanation: "四个字符只有三个相邻位置，最后一个字符没有后继标签。" },
  },
  "language-train": {
    objective: "用真实交叉熵梯度更新 Bigram 参数，区分训练损失、验证损失与困惑度。",
    steps: [
      { title: "查表得到 logits", description: "V×V 参数表中，第 i 行给出当前字符 i 后每个候选字符的原始分数。Softmax 把这行转成概率分布。" },
      { title: "用下一字符监督", description: "真实后继字符的概率越低，负对数损失越高。梯度是预测概率减去目标指示，再按所有样本平均；SGD 更新参数。" },
      { title: "独立计算验证损失", description: "每步更新后，用新权重重新评估两部分文本。验证仅计算损失，不参与梯度。验证困惑度为 exp(验证平均交叉熵)。" },
    ],
    formula: String.raw`p(j\mid i)=\mathrm{softmax}(W_i)_j,\quad L=-\frac1N\sum_t\ln p(y_t\mid x_t)`,
    formulaNote: "本实验用全量 SGD，不含隐藏层或注意力。均匀初始概率为 1/V，初始交叉熵为 ln V。",
    misconception: "训练损失下降不保证验证损失一直下降，更不代表模型已理解语言。验证文本很小，指标只用于本次教学比较。",
    quiz: { question: "验证文本应该在哪一步参与参数更新？", options: ["与训练文本混在一起", "不参与，只计算评估指标", "每隔 100 步参与一次"], answer: 1, explanation: "验证提供未参与更新的观测，若也用于梯度更新，就失去这个用途。" },
  },
  "language-generate": {
    objective: "使用已训练权重自回归采样，解释温度、随机种子与上下文长度的影响。",
    steps: [
      { title: "先算候选分布", description: "取提示词的最后一个 token，查找对应的一行 logits，除以温度再做 Softmax。" },
      { title: "采样并接回输入", description: "按照候选分布抽取一个 token，追加到文本末尾，再用新 token 预测下一位，直到达到指定长度。" },
      { title: "观察模型边界", description: "温度低通常更集中，温度高更分散。Bigram 只看最后一个字符，无法区分相同末字但前文不同的提示词。" },
    ],
    formula: String.raw`p_T(j\mid i)=\frac{\exp(W_{ij}/T)}{\sum_k\exp(W_{ik}/T)},\qquad T>0`,
    formulaNote: "固定权重、提示词、长度、温度与随机种子可复现输出。温度不会改变已训练参数。",
    misconception: "降低温度不会让模型学到新知识；更确定的输出也可能反复循环或不通顺。Bigram 不具备 Transformer 的多 token 上下文。",
    quiz: { question: "保持权重不变，提高采样温度通常会怎样？", options: ["重新训练权重", "增加上下文长度", "使候选概率更分散"], answer: 2, explanation: "温度缩放 logits；较高温度压小相对分差，概率趋于分散，不改变模型结构或权重。" },
  },
};

export const languageTransfer: Record<string, StudyQuestion> = {
  "language-data": { question: "验证出现训练词表中没有的字符，应如何处理？", options: ["映射到 UNK", "用验证集扩充词表并偷偷训练", "删除全部验证集"], answer: 0, explanation: "本实验保持训练词表固定，把未知字符映射到 ID 0，并报告数量。", whyWrong: ["", "扩充会改变模型输出维度，且引入验证信息。", "仍需保留验证观测，未知字符可以显式处理。"] },
  "language-train": { question: "验证交叉熵从 ln(10) 降到 ln(5)，验证困惑度如何变化？", options: ["10 降到 5", "5 升到 10", "始终是 1"], answer: 0, explanation: "PPL = exp(L)，所以 exp(ln 10)=10，exp(ln 5)=5。", whyWrong: ["", "指数函数单调递增，损失下降对应困惑度下降。", "只有交叉熵为 0 时困惑度才为 1。"] },
  "language-generate": { question: "Bigram 中，“小猫”和“看猫”在相同温度下的下一字符分布有何关系？", options: ["前者一定更准确", "相同，因为末字相同", "必须看全部前文才知道"], answer: 1, explanation: "这个模型只查“猫”对应的参数行，因此分布相同；Transformer 可利用更早的上下文。", whyWrong: ["这个模型看不到末字之前的内容。", "", "Bigram 上下文固定只有一个字符。"] },
};
export const languageRecall: Record<string, readonly [string, string, string]> = {
  "language-data": ["", "最后一个字符没有下一个字符，不能产生第四对。", "每两个相邻字符都能形成一个样本，并非只有整句一对。"],
  "language-train": ["混合后验证文本就参与了拟合。", "", "间隔训练仍属于训练，不能再视为未参与更新。"],
  "language-generate": ["采样阶段不计算训练梯度。", "模型上下文由结构决定，不能通过温度改变。", ""],
};
export const languagePredictions = {
  "language-data": { question: "“春天来了”有四个字符，是否也有四个下一字符标签？", options: ["有", "只有三个", "需要人工标注"], observation: "准备语料并逐对点选，检查最后一个字符是否有后继字符。" },
  "language-train": { question: "训练损失下降时，验证损失一定同时下降吗？", options: ["一定", "不一定", "验证不需要计算"], observation: "运行相同步数，对比两条真实损失曲线，不把训练拟合当成泛化结论。" },
  "language-generate": { question: "把温度从 1 改成 0.2，参数会发生变化吗？", options: ["会更新权重", "权重不变", "上下文变长"], observation: "固定随机种子，对照首个位置的候选分布及生成文本。" },
};
export const languageDeepDives: Record<string, LessonDeepDive> = {
  "language-data": {
    example: { title: "把四个字符变成三对样本", steps: ["设教学词表为小=1、猫=2、看=3、花=4；ID 0 留给未知字符。", "输入序列是 [1,2,3]，标签序列是 [2,3,4]，来自原文向右错开一个位置。", "“小”这一样本的目标是“猫”，训练时模型只能根据“小”预测，不能读取目标。"], tex: String.raw`[1,2,3]\longrightarrow[2,3,4],\qquad N_{pairs}=4-1=3`, takeaway: "自监督标签取自文本本身。这里的 ID 是教学举例，页面按真实词表显示。" },
    observations: [{ action: "选中连续两个样本。", expected: "前一对的标签是后一对的输入，二者沿原文移动。" }, { action: "改写语料后重新准备。", expected: "词表、划分与参数表一起重建，已训练的权重归零。" }],
    terms: [{ term: "Token", definition: "模型处理的离散单位，此处恰好是一个 Unicode 字符。" }, { term: "自监督", definition: "从数据本身构造监督信号，例如用下一字符作标签。" }],
    connection: "有了输入与标签，下一节用交叉熵把预测错误变成参数梯度。", next: "language-train",
  },
  "language-train": {
    example: { title: "均匀猜测的损失基线", steps: ["假设词表有 10 个 token，零初始化时每个候选的概率都是 0.1。", "不论真实目标是哪一个，单样本交叉熵都是 −ln(0.1)，约为 2.3026。", "如果平均交叉熵降到 ln(5)，困惑度就是 5；它不是准确率，也不是词表变成了 5 个。"], tex: String.raw`L_0=-\ln(1/10)=\ln10,\qquad \mathrm{PPL}=e^{\ln5}=5`, takeaway: "用相同词表和划分比较损失；不同分词单位的困惑度不能直接横向比较。" },
    observations: [{ action: "先单步，再连续训练 100 步。", expected: "曲线从初始化基线开始，所有点来自实际更新后的重新评估。" }, { action: "观察验证损失和训练损失的差距。", expected: "训练样本转移更常见时拟合更好，验证数据未必以同样速度改善。" }],
    terms: [{ term: "Bigram", definition: "用当前一个 token 估计下一 token 的条件概率模型。" }, { term: "困惑度", definition: "平均负对数似然的指数；此处使用自然对数，越低表示目标 token 概率整体更高。" }],
    connection: "下一节把预测分布变成真实采样，并把生成字符接回模型。", next: "language-generate",
  },
  "language-generate": {
    example: { title: "温度如何改变两个候选", steps: ["设两个候选 logits 分别为 ln(3) 和 0，在 T=1 时概率为 0.75 和 0.25。", "把 T 降为 0.5，相当于把 logits 加倍，概率变为 0.9 和 0.1。", "两个候选仍然都可能被采样；改变温度不添加新参数，也不改变训练数据。"], tex: String.raw`p_{T=1}=\frac3{3+1}=0.75,\qquad p_{T=0.5}=\frac9{9+1}=0.9`, takeaway: "较低温度放大相对分差；只有相同权重和设置下才适合比较输出。" },
    observations: [{ action: "固定种子，使用两个末字相同的提示词。", expected: "生成后缀相同，因为 Bigram 不使用更早的前文。" }, { action: "固定提示词与权重，只调整温度。", expected: "候选分布发生变化，训练步数和权重仍保持原样。" }],
    terms: [{ term: "自回归生成", definition: "将先前得到的 token 作为后续输入，一次生成一个新 token。" }, { term: "因果注意力", definition: "Transformer 中只允许当前位置关注当前与更早位置的机制，避免训练时读取未来答案。" }],
    connection: "下一阶段将从因果注意力开始，真实训练具有多 token 上下文的 Transformer。", next: "transformer-architecture",
  },
};
