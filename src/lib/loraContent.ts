import type { LessonInsight } from "./lessonInsights";
import type { LessonDeepDive } from "./lessonDeepDives";
import type { StudyQuestion } from "./studyQuestions";

export const loraInsights: Record<string, LessonInsight> = {
  "lora-basics": {
    objective: "区分预训练与微调，用 A、B 的形状和 rank 解释 LoRA 为什么只需更新少量参数。",
    steps: [
      { title: "加载真正的预训练权重", description: "Pythia-14M 已经学习过通用英文文本。本课程下载固定版本 Safetensors，并继续使用原始子词分词器，不从随机基座开始。" },
      { title: "冻结原映射并增加旁路", description: "线性层的 W 不接收梯度。LoRA 新增 A 与 B，先压缩到 r 维，再还原到输出维度，与原映射输出相加。" },
      { title: "控制秩和缩放", description: "r 控制适配器参数规模，α/r 缩放旁路输出。A 随机初始化、B 为零，因此初始增量为零；随着训练才改变预测。" },
    ],
    formula: String.raw`y=Wx+\frac{\alpha}{r}BAx,\quad A\in\mathbb{R}^{r\times d_{in}},\quad B\in\mathbb{R}^{d_{out}\times r}`,
    formulaNote: "单层可训练参数从全量更新的 d_out×d_in 变为 r(d_in+d_out)，但仍需加载冻结的基座权重。",
    misconception: "LoRA 减少可训练参数，不会把整个模型缩小成只有 A、B。使用适配器进行推理仍需匹配的预训练基座。",
    quiz: { question: "标准 LoRA 微调时，本课程的哪些参数会被优化器更新？", options: ["只有新增 A、B", "所有基座参数", "只有词嵌入"], answer: 0, explanation: "本课程冻结全部基座，只让目标 QKV 层的 LoRA A、B 参与优化。" },
  },
  "lora-train": {
    objective: "运行真正的预训练模型 LoRA 微调，检查训练损失、验证结果与基座冻结证据。",
    steps: [
      { title: "先分文本，再使用原分词器", description: "原文前 80% 用于训练、后 20% 验证，分别编码为子词 token。词表继承预训练模型，不用微调语料重建。" },
      { title: "只对 LoRA 参数求更新", description: "模型仍执行完整因果前向，使用下一 token 交叉熵；优化器只接收 requires_grad=True 的 LoRA 参数，梯度裁剪后执行 AdamW。" },
      { title: "校验冻结并保存", description: "每批比较基座参数 SHA-256，只有保持一致才保存。检查点包含当前和最佳适配器、优化器与随机采样状态，并记录固定基座版本。" },
    ],
    formula: String.raw`L=-\frac1{BT}\sum_{b,t}\ln p_{W, A, B}(x_{b,t+1}\mid x_{b,\le t}),\qquad W\ \mathrm{frozen}`,
    formulaNote: "本实验做英文领域文本的因果语言建模，不使用聊天模板或只对回答计算损失的 SFT 掩码。",
    misconception: "只训练少量参数也可能过拟合。训练损失降低与输出更像语料，不证明回答更准确；需要对照未参与梯度的验证文本。",
    quiz: { question: "验证曲线来自哪部分数据？", options: ["每次混入训练的验证文本", "未参与参数更新的后 20% 文本", "生成出来的随机文本"], answer: 1, explanation: "划分后独立编码和评估，验证文本不进入训练窗口采样。" },
  },
  "lora-generate": {
    objective: "在同一固定基座上开关适配器进行生成对比，并说明可移植适配器包包含什么。",
    steps: [
      { title: "对齐比较条件", description: "两列共用基座、提示词、温度、Top-k、长度和同种子采样器。左侧关闭 LoRA，右侧加载当前或最佳验证适配器。" },
      { title: "确认使用的适配器步数", description: "最佳验证适配器不一定是最新一步；步 0 的 B=0，仍是原始预测。生成不会修改微调检查点。" },
      { title: "保存适配器和基座身份", description: "ZIP 包提供 adapter_model.safetensors、adapter_config.json 及版本清单；重新推理时先载入同一 revision 的基座，再通过 PEFT 挂载适配器。" },
    ],
    formula: String.raw`f_{base}(x)=f_W(x),\qquad f_{adapted}(x)=f_{W+\Delta W}(x),\quad \Delta W=\frac\alpha r BA`,
    formulaNote: "适配器不含完整基座，也不含继续训练所需的 AdamW 状态；继续训练应使用另一份续训检查点。",
    misconception: "不同输出只能证明生成路径受到适配器影响，不能单凭一段文本认定微调模型更好。固定种子也不能保证两个不同分布采样出相同 token。",
    quiz: { question: "只拿到 PEFT 适配器 ZIP，能直接独立运行完整模型吗？", options: ["能，里面有全部基座权重", "只需要随便一个同大小模型", "还需匹配的预训练基座与版本"], answer: 2, explanation: "A、B 是针对固定基座学习的低秩更新，必须与对应结构和权重一起使用。" },
  },
};
export const loraTransfer: Record<string, StudyQuestion> = {
  "lora-basics": { question: "W 是 384×128，rank=4 时，A、B 一共多少个可训练参数？", options: ["2,048", "49,152", "512"], answer: 0, explanation: "A 有 4×128 个，B 有 384×4 个，共 4×512=2,048。六个目标层共 12,288。", whyWrong: ["", "49,152 是原始 W 的参数量，本课程将其冻结。", "512 是输入和输出维度之和，还需要乘 rank 4。"] },
  "lora-train": { question: "刷新服务后精确继续同一实验，需要恢复什么？", options: ["只恢复 A、B 即可", "固定基座、A/B、优化器和采样状态", "随机重新初始化同尺寸基座"], answer: 1, explanation: "基座版本提供同一函数起点，适配器和 AdamW、采样 RNG 状态一起延续更新轨迹。", whyWrong: ["缺少 AdamW 矩状态和采样状态，后续轨迹会改变。", "", "随机基座已经不是学习该适配器时使用的预训练模型。"] },
  "lora-generate": { question: "刚创建但未训练的适配器 B=0，固定输入和采样设置时应该看到什么？", options: ["两列输出一致", "必然右侧更好", "基座权重已经被替换"], answer: 0, explanation: "BA=0，启用适配器不会改变 logits；同种子与设置产生一致输出。", whyWrong: ["", "未训练的零增量不能带来能力提升。", "LoRA 添加旁路，没有替换原始基座参数。"] },
};
export const loraRecall: Record<string, readonly [string, string, string]> = {
  "lora-basics": ["", "全部基座参数在本实现中 requires_grad=False。", "词嵌入保持冻结，实际目标是 QKV 层的低秩旁路。"],
  "lora-train": ["进入训练窗口就不再是未参与更新的验证集。", "", "验证使用原始保留文本，不使用采样生成文本。"],
  "lora-generate": ["ZIP 只包含适配器和配置，不重复打包基座。", "同样参数规模不代表结构、分词器与权重完全相同。", ""],
};
export const loraPredictions = {
  "lora-basics": { question: "只训练 LoRA，还需要把预训练基座载入内存吗？", options: ["需要", "只需 A、B", "与基座无关"], observation: "准备公开权重，再比较冻结基座参数与少量可训练参数。" },
  "lora-train": { question: "微调 20 步后，冻结基座的参数校验值应该改变吗？", options: ["应该改变", "应保持相同", "无需检查"], observation: "运行真实微调，检查训练损失和冻结参数 SHA-256 校验状态。" },
  "lora-generate": { question: "初始 B=0 时，开启 LoRA 会改变模型输出吗？", options: ["会", "不会", "一定更通顺"], observation: "先对比步 0，再用微调后的适配器，在相同条件下生成。" },
};
export const loraDeepDives: Record<string, LessonDeepDive> = {
  "lora-basics": {
    example: { title: "为一层 QKV 计算 LoRA 参数量", steps: ["原始 W 有 384×128=49,152 个参数，这些参数全部冻结。", "rank=4 时，A 是 4×128，B 是 384×4，共 2,048 个可训练参数。", "六层都添加同样的适配器，共 12,288 个；若 α=8，低秩更新缩放为 8/4=2。"], tex: String.raw`N_{LoRA}=6\times4(128+384)=12288,\qquad \alpha/r=2`, takeaway: "这里只在融合 QKV 线性层添加 LoRA；整个基座的词嵌入和其他层仍需保留。" },
    observations: [{ action: "点选 rank 2、4、8、16。", expected: "A、B 形状和可训练参数随 rank 线性增加，W 的参数量保持不变。" }, { action: "查看固定 α=8 的缩放。", expected: "rank 越高时 α/r 反而降低，所以比较秩时也要核对缩放。" }],
    terms: [{ term: "预训练基座", definition: "已经在通用文本上学习得到参数的模型，作为后续适配的起点。" }, { term: "LoRA rank", definition: "低秩旁路的中间维度，控制 A、B 的形状与容量。" }],
    connection: "下一节使用固定版本预训练模型，真实更新这些低秩矩阵。", next: "lora-train",
  },
  "lora-train": {
    example: { title: "完整模型与少量可训练参数", steps: ["Pythia-14M 的实际基座参数数为 14,067,712，默认适配器新增 12,288 个。", "可训练占比采用 LoRA 参数除以含适配器的总参数，约为 0.0873%。", "每批保存前对所有非 LoRA 参数逐字节计算 SHA-256，确认训练没有改变基座。"], tex: String.raw`\mathrm{ratio}=\frac{12288}{14067712+12288}\approx0.0873\%`, takeaway: "参数效率不等于全部内存都按该比例减少；前向计算与冻结权重仍然存在。" },
    observations: [{ action: "创建默认实验，再微调 20 步。", expected: "第 0 步是原预训练基座损失，后续曲线来自实际 LoRA 更新，冻结校验保持通过。" }, { action: "暂停后刷新页面，选择同一实验继续 1 步。", expected: "从已保存步数继续，微调配置与语料不因编辑新实验配置而改变。" }],
    terms: [{ term: "参数冻结", definition: "从优化器中排除原始参数，并关闭其梯度更新；模型仍使用这些权重参与前向。" }, { term: "领域微调", definition: "在已有模型上，用领域文本继续优化目标任务。本例使用下一 token 语言建模。" }],
    connection: "下一节在同一基座上开关适配器，比较实际文本生成。", next: "lora-generate",
  },
  "lora-generate": {
    example: { title: "适配器包与续训检查点的区别", steps: ["adapter_model.safetensors 保存 A、B，adapter_config.json 保存 rank、alpha、目标模块和基座引用。", "版本清单记录基座 revision、原始权重 SHA-256、实验 ID 和实际导出步数。", "如果需要继续微调，使用包含 AdamW 和随机采样状态的续训检查点；适配器 ZIP 主要用于挂载推理。"], tex: String.raw`\mathrm{Inference}=\mathrm{Pinned\ Base}+\mathrm{LoRA\ Adapter}`, takeaway: "必须保留基座身份，不能把任意同规模模型与适配器混用。" },
    observations: [{ action: "先选当前适配器，再选最佳验证适配器，固定提示词和种子。", expected: "页面显示实际加载步数，基座输出保持一致，适配器输出可能不同。" }, { action: "下载适配器 ZIP，查看文件清单。", expected: "标准 PEFT 权重、配置和基座版本清单存在，完整基座权重不在包内。" }],
    terms: [{ term: "PEFT 适配器", definition: "按 PEFT 格式保存的参数高效微调模块，可挂载到匹配基座上使用。" }, { term: "基座版本", definition: "预训练仓库的固定提交及权重身份，用于确保适配器加载到相同起点。" }],
    connection: "下一阶段把自定义问答变成回答监督 SFT，再用相同模板进行多轮对话。", next: "sft-data",
  },
};
