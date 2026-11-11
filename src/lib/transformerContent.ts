import type { LessonInsight } from "./lessonInsights";
import type { LessonDeepDive } from "./lessonDeepDives";
import type { StudyQuestion } from "./studyQuestions";

export const transformerInsights: Record<string, LessonInsight> = {
  "transformer-architecture": {
    objective: "追踪 token 与位置嵌入如何进入因果多头注意力，说明为什么必须屏蔽未来。",
    steps: [
      { title: "给字符和位置各一个向量", description: "Token 嵌入表达字符，位置嵌入表达顺序；两者相加，形成 B×T×D 的张量。" },
      { title: "查询、键和值共同工作", description: "每个头根据 Q 与 K 的点积决定如何汇总 V。分数除以头维度的平方根，在 Softmax 前把未来位置屏蔽为负无穷。" },
      { title: "堆叠因果块", description: "本模型使用 Pre-LN：先 LayerNorm 再注意力或前馈子层，每个子层带残差；最后映射为词表 logits。每个位置预测其下一个字符。" },
    ],
    formula: String.raw`\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V`,
    formulaNote: "M 在当前与过去位置为 0，在未来位置为 −∞；Softmax 后未来权重为 0。",
    misconception: "训练虽然一次并行处理多个位置，但每个位置的注意力只能看到自身与过去。去掉掩码会泄漏下一字符答案。",
    quiz: { question: "位置 3 的表示在因果注意力中能读取哪些输入？", options: ["位置 1、2、3", "所有位置，包括未来", "只能位置 3"], answer: 0, explanation: "因果掩码保留当前及过去，屏蔽位置 4 及之后。它比 Bigram 能读取更多前文。" },
  },
  "transformer-train": {
    objective: "运行真实 PyTorch 训练，区分当前与最佳验证权重，理解检查点如何支持继续训练。",
    steps: [
      { title: "采样窗口并构造右移标签", description: "每批仅从训练文本采样连续窗口。每个输入位置的标签是原文的下一个字符，批内所有位置的交叉熵取平均。" },
      { title: "反向传播并应用 AdamW", description: "清空梯度，计算 logits 和交叉熵，反向传播，裁剪梯度范数到 1，再由 AdamW 更新参数。验证文本不参与这些步骤。" },
      { title: "评估后原子保存", description: "每个提交批次完成后重新计算完整训练与验证损失；保留最低验证损失权重，并保存当前权重、优化器状态和采样 RNG，供下次继续。" },
    ],
    formula: String.raw`L=-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\ln p_\theta(x_{b,t+1}\mid x_{b,\le t})`,
    formulaNote: "曲线记录提交批次结束后的全量分块评估；不是每一步小批次损失。最佳权重由验证损失决定，不一定是最新权重。",
    misconception: "仅保存模型参数不能保证原样继续 AdamW 训练；还需要优化器矩状态与随机窗口采样状态。本模型没有 Dropout。",
    quiz: { question: "验证损失先下降再上升，应该如何选择用于对比的最佳权重？", options: ["总是取最后一步", "取验证损失最低的已保存步", "使用验证集更新参数"], answer: 1, explanation: "训练拟合可持续改善而验证变差。最低验证损失权重提供一致的选择标准，验证不用于梯度更新。" },
  },
  "transformer-generate": {
    objective: "用持久化权重自回归采样，解释 Top-k、上下文滑窗与实际注意力的含义。",
    steps: [
      { title: "加载并标明所选权重", description: "可以选择当前权重或最佳验证权重。模型进入 eval 与 inference_mode，输入只取提示词最后 C 个 token，位置从 0 开始。" },
      { title: "温度缩放并筛选候选", description: "logits 除以温度后，Top-k 可只保留分数最高的 k 个 token，其余设为负无穷，重新 Softmax 后按概率采样。" },
      { title: "接回生成 token，再读上下文", description: "每次新增 token 都接回输入，长度超过 C 时丢弃最早部分。首个生成位置返回最后一层各注意力头的真实矩阵。" },
    ],
    formula: String.raw`p(j)=\mathrm{softmax}\left(\mathrm{TopKMask}(z/T)\right)_j,\qquad x_{\mathrm{context}}=x_{-C:}`,
    formulaNote: "Top-k=0 表示保留全部词表项。显示的前 8 项可能不是全部候选；注意力归一化与采样概率是两种不同分布。",
    misconception: "注意力权重是在某一层、某个头、某个查询位置上的信息汇总系数，不是词表概率，也不能当作因果证明。",
    quiz: { question: "Top-k=5 时，下一 token 从哪些候选中采样？", options: ["固定词表前 5 个 ID", "任意 token，概率完全不变", "当前 logits 最高的至多 5 个 token"], answer: 2, explanation: "Top-k 按当前分数筛选，再在保留集合内归一化采样。词表不足 5 项时保留全部。" },
  },
};

export const transformerTransfer: Record<string, StudyQuestion> = {
  "transformer-architecture": { question: "嵌入维度 D=32，注意力头数 H=4，每个头的维度是多少？", options: ["8", "32", "128"], answer: 0, explanation: "本模型平均分头，每头维度 D/H=32/4=8，注意力缩放因子为 √8。", whyWrong: ["", "32 是所有头拼接后的总维度。", "分头是相除，不能乘以头数。"] },
  "transformer-train": { question: "服务重启后要继续同一条 AdamW 训练轨迹，哪份保存更完整？", options: ["仅模型参数", "仅种子", "参数、优化器、步数和随机采样状态"], answer: 2, explanation: "参数、AdamW 的矩状态与采样 RNG 一起恢复，才能延续本实验的更新轨迹。", whyWrong: ["缺少优化器矩状态和后续采样状态，继续训练的轨迹会不同。", "种子只能回到起点，不能恢复已训练的参数与优化器。", ""] },
  "transformer-generate": { question: "上下文窗口为 16，提示词有 30 个 token，首次前向使用哪些输入？", options: ["最后 16 个 token", "最前面 16 个 token", "30 个全部使用"], answer: 0, explanation: "本模型保留最近 16 个 token，位置重新从 0 编号；输出显示仍包含完整提示词。", whyWrong: ["", "本实现滑窗保留最近前文，丢弃过早部分。", "学习式位置嵌入只有 16 个位置，不能超过配置窗口。"] },
};
export const transformerRecall: Record<string, readonly [string, string, string]> = {
  "transformer-architecture": ["", "未来 token 含有待预测答案，必须屏蔽。", "仅看当前位置是 Bigram 式限制，因果注意力也允许过去位置。"],
  "transformer-train": ["最新权重可能已经过拟合，验证损失更高。", "", "验证只评估，不能用来更新参数。"],
  "transformer-generate": ["候选按当前 logits 排序，与 ID 顺序无关。", "Top-k 会把集合外的概率置零并重新归一化。", ""],
};
export const transformerPredictions = {
  "transformer-architecture": { question: "训练并行处理一整段文本，位置 2 能读取位置 3 吗？", options: ["能", "被因果掩码阻止", "只有训练时能"], observation: "点选第 2 个位置，核对掩码矩阵中右侧位置的状态。" },
  "transformer-train": { question: "训练步数增加，最新权重一定是最佳验证权重吗？", options: ["一定", "不一定", "无需验证"], observation: "训练 100 步后比较训练与验证曲线，核对最佳权重步数。" },
  "transformer-generate": { question: "两个提示词最后一个字相同，Transformer 的下一 token 分布必须相同吗？", options: ["必须相同", "前文不同可能改变分布", "只由随机种子决定分布"], observation: "固定权重与温度，比较不同前文的候选概率，区别于 Bigram。" },
};
export const transformerDeepDives: Record<string, LessonDeepDive> = {
  "transformer-architecture": {
    example: { title: "四个位置的因果读取边界", steps: ["对“小猫看花”四个输入，位置 1 只可读取“小”，位置 2 可读取“小猫”。", "位置 3 的查询可汇总“小猫看”，但不能读取“花”，因为“花”是这个位置要预测的答案。", "每头维度为 D/H；D=32、H=4 时，每头 8 维，最后拼接恢复 32 维。"], tex: String.raw`M=\begin{bmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{bmatrix}`, takeaway: "并行计算不意味着可以读取未来。标签右移与因果掩码必须同时成立。" },
    observations: [{ action: "依次点选最前与最后一个输入位置。", expected: "最前位置只看到自身，最后位置可以读取全部已给出的前文。" }, { action: "把当前位置看作预测下一字符的查询。", expected: "可见输入中不包含这个查询对应的下一字符标签。" }],
    terms: [{ term: "多头注意力", definition: "把表示分到多个子空间，分别汇总上下文，再拼接投影。" }, { term: "位置嵌入", definition: "为序列位置提供可学习向量，使模型区分字符的先后顺序。" }],
    connection: "下一节创建小型 Transformer，运行真正的反向传播与 AdamW 更新。", next: "transformer-train",
  },
  "transformer-train": {
    example: { title: "批次、训练步和最佳权重", steps: ["批大小 16、上下文 16 时，每个训练步产生 16×16=256 个下一字符监督位置；窗口可能重叠。", "点击训练 100 步，界面按 4 个 25 步请求提交；每个请求完成后全量评估并保存一个曲线点。", "若已保存步 25、50、75 的验证损失分别为 2.8、3.2、3.7，则最佳权重来自步 25，当前权重仍可继续训练。"], tex: String.raw`N_{\mathrm{targets/step}}=BT=16\times16=256,\qquad s_{best}=\arg\min_s L_{val}(s)`, takeaway: "最佳权重和当前权重用途不同；暂停只阻止后续提交，不撤销已提交批次。这里的损失数值是教学例子。" },
    observations: [{ action: "创建默认实验，连续训练 100 步。", expected: "当前步数与记录持续更新，验证曲线可能先下降再上升，最佳步数随最低损失保存。" }, { action: "刷新页面后选择同一实验，再训练 1 步。", expected: "从服务端保存的当前步数继续，AdamW 和窗口采样状态一并恢复。" }],
    terms: [{ term: "梯度裁剪", definition: "把总梯度范数限制在阈值内，减少过大梯度导致的更新不稳定；本模型阈值为 1。" }, { term: "检查点", definition: "保存模型与训练状态的本机文件，包含当前和最佳权重、优化器及随机采样状态。" }],
    connection: "下一节把检查点中的权重用于自回归采样，观察真实注意力。", next: "transformer-generate",
  },
  "transformer-generate": {
    example: { title: "候选分布与注意力不是一回事", steps: ["对含 8 个上下文 token 的输入，最后一层每个头返回一个 8×8 注意力矩阵。", "选择最后一个查询位置，会得到对 8 个输入位置的权重；输出层则产生对 V 个词表项的预测 logits。", "温度与 Top-k 作用于词表 logits。生成新 token 后接回输入，下一步重新计算上下文和注意力。"], tex: String.raw`A\in\mathbb{R}^{H\times T\times T},\quad z\in\mathbb{R}^{V},\quad \sum_j A_{h,t,j}=1`, takeaway: "输入位置注意力描述怎样汇总前文；词表概率描述接下来输出什么。二者不能互相替代。" },
    observations: [{ action: "生成后切换 Head，并选择靠前的查询位置。", expected: "各头分配可不同，但所有未来位置始终为 0，每行总和约为 1。" }, { action: "固定权重与提示词，只改变温度或 Top-k。", expected: "采样分布变化，首个位置的原始注意力不变，因为模型前向输入和权重未变。" }],
    terms: [{ term: "Top-k 采样", definition: "仅保留当前分数最高的至多 k 个 token，归一化后采样。" }, { term: "上下文滑窗", definition: "长度超过模型窗口时仅保留最近的 token；本实现将窗口中的位置从 0 重新编号。" }],
    connection: "下一阶段载入公开预训练基座，使用 LoRA 学习少量低秩参数并对比生成。", next: "lora-basics",
  },
};
