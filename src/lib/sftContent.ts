import type { LessonInsight } from "./lessonInsights";
import type { LessonDeepDive } from "./lessonDeepDives";
import type { StudyQuestion } from "./studyQuestions";

export const sftInsights: Record<string, LessonInsight> = {
  "sft-data": {
    objective: "把自定义问答组织成一致的对话模板，区分输入上下文和真正参与损失的助手目标。",
    steps: [
      { title: "先校验结构与划分", description: "支持 question/answer 或 user、assistant 交替的 messages。按首问分组，80% 组训练、20% 组验证，避免同一首问跨集合。" },
      { title: "使用统一的教学模板", description: "Pythia 没有原生聊天模板，本课程使用版本化的 System / User / Assistant 文本格式。训练和推理始终共用该模板。" },
      { title: "只保留回答标签", description: "系统、用户、角色前缀的 label 为 −100；助手回答与 EOS 保留 token ID。忽略损失不代表从输入删除：问题仍是生成回答需要的上下文。" },
    ],
    formula: String.raw`y_t=\begin{cases}x_t,&t\in\mathrm{assistant\ answer\ or\ EOS}\\-100,&\mathrm{otherwise}\end{cases}`,
    formulaNote: "标签与输入先对齐保存，损失再用前一位置 logits 预测下一位置 label。回答掩码由实际模板分词得到。",
    misconception: "不能先把问题 token 删除再训练回答，否则模型看不到问题。−100 只表示此目标不计入交叉熵。",
    quiz: { question: "用户问题的 label=−100，意味着什么？", options: ["问题仍在输入中，只忽略对应损失", "把问题从输入彻底删除", "把问题 token ID 改成 0"], answer: 0, explanation: "问题提供上下文；掩码只控制哪些目标参与优化，并不改变原输入 ID。" },
  },
  "sft-train": {
    objective: "执行回答监督的真实 LoRA 微调，核对一次右移、填充掩码、有效目标分母与验证隔离。",
    steps: [
      { title: "组成带掩码的批次", description: "每步只抽取训练样本，右侧补齐到当前批次最长长度；填充位置 attention_mask=0 且 label=−100。不会把不同样本拼接成同一对话。" },
      { title: "只右移一次计算损失", description: "模型位置 t 的 logits 预测位置 t+1 的 label。交叉熵忽略 −100，按有效回答和 EOS 目标数平均；优化器只更新 LoRA A、B。" },
      { title: "完整评估并保存状态", description: "每批结束评估完整训练与验证样本，按有效目标数加权。验证不反传；保存当前/最佳适配器、优化器、采样 RNG 与数据和模板快照。" },
    ],
    formula: String.raw`L=-\frac1{\sum_{b,t}m_{b,t+1}}\sum_{b,t}m_{b,t+1}\ln p_\theta(x_{b,t+1}\mid x_{b,\le t})`,
    formulaNote: "m=1 表示有效助手目标，m=0 表示忽略。SFT 是监督方式，LoRA 是参数更新方式，两者可以组合。",
    misconception: "若再手工右移 labels 后又让损失右移一次，会错开两个位置。把填充 EOS 也计入损失，同样会污染目标与指标。",
    quiz: { question: "本实验训练回答交叉熵的平均分母是什么？", options: ["全部输入和填充 token 数", "有效回答与 EOS 目标数", "数据集字符总数"], answer: 1, explanation: "只有 label 不为 −100 的下一 token 目标进入损失及其平均分母。" },
  },
  "sft-chat": {
    objective: "使用同一训练模板进行多轮推理，识别生成前缀、历史裁剪、结束条件和真实适配器来源。",
    steps: [
      { title: "沿用完整模板与权重", description: "载入所选 SFT 适配器，系统提示固定为实验快照；用户、助手历史按原模板排列，最后补 Assistant:，不附带真实回答。" },
      { title: "按完整轮次管理预算", description: "请求最多携带最近四轮。服务进一步丢弃最早的完整轮次，保留系统提示与最新问题，并为新回答预留上下文。无法容纳最新问题时明确报错。" },
      { title: "真实逐 token 解码", description: "温度为 0 时贪心，否则按温度与 Top-k 采样。遇 EOS、角色边界或长度预算停止；界面显示所用步数、上下文长度与结束原因。" },
    ],
    formula: String.raw`\mathrm{prompt}=\mathrm{Template}(\mathrm{system},\mathrm{history},\mathrm{user})+\text{Assistant:}`,
    formulaNote: "导出的适配器同时携带 chat_template.jinja 和系统提示/长度配置。正确模板是推理契约的一部分。",
    misconception: "界面上出现聊天气泡，不代表基座已经变成可靠助手；小模型可能复述、答非所问或提前结束。判断时应区分训练问题和未见过的验证问题。",
    quiz: { question: "对话推理在最新用户问题之后应添加什么？", options: ["数据集中的真实答案", "另一个随机用户问题", "与训练一致的 Assistant: 生成前缀"], answer: 2, explanation: "只提供助手开始的格式，让模型自回归生成回答；提前放入答案会失去推理检验意义。" },
  },
};
export const sftTransfer: Record<string, StudyQuestion> = {
  "sft-data": { question: "两条对话首问相同但后续轮次不同，应如何划分？", options: ["放在同一组，再划分训练/验证", "强制一条训练、一条验证", "只比较文件行号"], answer: 0, explanation: "本项目按标准化首问分组，避免重复首问跨集合；它不等同于语义去重。", whyWrong: ["", "这样会让同一首问在两个集合中重复出现。", "行号不同不代表问题不同，需按实际首问分组。"] },
  "sft-train": { question: "一个样本有 3 个回答 token、1 个真实 EOS、5 个填充 EOS，应计多少个监督目标？", options: ["9 个", "4 个", "只有 3 个"], answer: 1, explanation: "回答的 3 个 token 加真实 EOS 共 4 个；填充虽然 ID 相同，但 label=−100，被忽略。", whyWrong: ["填充不是回答目标，不能纳入分母。", "", "真实 EOS 也是本课程监督的结束目标。"] },
  "sft-chat": { question: "多轮历史超出预算时，本实现怎样处理？", options: ["优先丢弃最早完整轮次，保留最新问题", "把最新问题删除", "无提示地切掉问题一半"], answer: 0, explanation: "先裁剪完整旧轮次；如果系统提示与最新问题仍放不下，返回明确错误。", whyWrong: ["", "最新用户问题是当前回答的必要条件。", "不应静默破坏最新问题或角色边界。"] },
};
export const sftRecall: Record<string, readonly [string, string, string]> = {
  "sft-data": ["", "输入仍需要问题上下文；掩码只影响目标损失。", "输入 ID 与 label 是两份数据，不能把 −100 解释为输入 ID 0。"],
  "sft-train": ["输入、角色前缀和填充中有大量被忽略位置，不能放进平均分母。", "", "模型训练使用 token 标签，且本实验只统计助手目标。"],
  "sft-chat": ["附带真实答案会泄漏要生成的目标。", "另加用户问题会改变当前任务。", ""],
};
export const sftPredictions = {
  "sft-data": { question: "不计算问题的损失，还需要把问题送进模型吗？", options: ["需要", "不需要", "只送 token 数"], observation: "预览真实 token ID 与标签，比较灰色输入和亮色回答目标。" },
  "sft-train": { question: "填充使用 EOS token ID，它是否也都应该被监督？", options: ["全部监督", "只有真实回答 EOS，填充忽略", "所有 EOS 都忽略"], observation: "观察有效回答目标数与损失定义，确认填充标签为 −100。" },
  "sft-chat": { question: "换一套角色前缀，模型行为会保持完全相同吗？", options: ["一定相同", "可能改变", "与训练格式无关"], observation: "检查实际模板版本和适配器来源，再进行多轮生成。" },
};
export const sftDeepDives: Record<string, LessonDeepDive> = {
  "sft-data": {
    example: { title: "输入和标签是两张不同的表", steps: ["假设已分词输入依次包含角色前缀、问题、助手前缀、回答和 EOS，所有 token 都保留在 input_ids。", "问题和角色前缀的 labels 改成 −100；回答与 EOS 的 labels 保留真实 token ID。", "训练时对齐 logits 的前 N−1 个位置与 labels 的后 N−1 个位置，首个回答由它前一个位置预测。"], tex: String.raw`\mathrm{logits}_{0:N-1}\longleftrightarrow\mathrm{labels}_{1:N},\qquad \mathrm{ignore\_index}=-100`, takeaway: "输入掩码与目标掩码作用不同；本课的亮色表示参与目标损失，不表示其他输入被删除。" },
    observations: [{ action: "分别点击一个问题 token 和一个回答 token。", expected: "问题显示 label=−100，回答显示真实 ID；两者都在输入序列中。" }, { action: "导入 JSONL 后运行校验。", expected: "真实分词、首问分组、样本长度与回答目标数更新；不合规数据不会进入训练。" }],
    terms: [{ term: "回答掩码", definition: "指示哪些 token 标签参与 SFT 损失，系统与用户文本通常作为条件而非目标。" }, { term: "对话模板", definition: "将角色消息序列化为模型输入的规则。本课使用明确版本化的教学模板。" }],
    connection: "下一节将这些标签用于真正的 LoRA 反向传播。", next: "sft-train",
  },
  "sft-train": {
    example: { title: "为什么不能按全部输入长度平均", steps: ["假设一个批次有 12 个输入位置，其中只有 4 个下一 token 目标是助手回答或真实 EOS。", "这四个目标的负对数概率若分别为 1、2、1、2，总和为 6。", "回答损失为 6/4=1.5；用全部 12 个输入位置作分母会错误得到 0.5。"], tex: String.raw`L_{answer}=\frac{1+2+1+2}{4}=1.5\ne\frac6{12}`, takeaway: "这个数字例子只用于说明有效目标加权，不是预录的训练结果。" },
    observations: [{ action: "用默认数据创建实验，再训练 20 或 100 步。", expected: "训练与验证回答损失来自实际计算；二者可能分离，不保证验证持续下降。" }, { action: "暂停后刷新，再继续相同实验。", expected: "数据指纹、模板与 LoRA 状态恢复，继续使用原训练集合及优化器。" }],
    terms: [{ term: "SFT", definition: "通过示范输入与期望输出监督模型；本课仅监督助手回答和 EOS。" }, { term: "有效目标数", definition: "右移后 label 不为 −100 的位置数量，是本课回答交叉熵的平均分母。" }],
    connection: "下一节沿用同一模板，让所训练适配器生成真实对话回答。", next: "sft-chat",
  },
  "sft-chat": {
    example: { title: "保留角色边界再管理上下文", steps: ["将系统提示、已完成的用户/助手轮次与最新用户问题按训练模板拼接，最后添加 Assistant:。", "如果历史过长，先丢弃最早的一整轮 user+assistant，不从最新问题中间截断。", "在剩余长度中生成回答，遇 EOS 或下一角色边界停止；超过预算时不会把不可见答案补进输入。"], tex: String.raw`N_{context}+N_{new}\le C,\qquad \mathrm{stop}\in\{EOS,\mathrm{role},\mathrm{length}\}`, takeaway: "对话内容与模型实际使用上下文并不总相同，界面会报告忽略轮次与实际 token 数。" },
    observations: [{ action: "发送一个训练问题，再继续追问。", expected: "请求包含真实历史与新问题，回复来自所选适配器，而非数据集答案直接返回。" }, { action: "切换当前/最佳适配器。", expected: "旧对话清空，实际步数更新，避免把不同模型输出混作同一会话来源。" }],
    terms: [{ term: "生成前缀", definition: "提示模型现在应开始助手回答的格式片段，例如本课的 Assistant:。" }, { term: "EOS", definition: "结束序列 token。本课把真实回答 EOS 纳入监督，生成时用它停止。" }],
    connection: "下一阶段用偏好对继续训练，比较固定参考与 DPO 策略。", next: "dpo-data",
  },
};
