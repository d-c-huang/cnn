import type { LessonInsight } from "./lessonInsights";
import type { LessonDeepDive } from "./lessonDeepDives";
import type { StudyQuestion } from "./studyQuestions";
export const dpoInsights: Record<string, LessonInsight> = {
  "dpo-data": {
    objective: "把自定义偏好标注变为同一问题的两个候选，并检查真正计分的 token。",
    steps: [{title:"一份上下文，两份候选",description:"每条包含 prompt、chosen 和 rejected。prompt 可以包含历史，但两候选必须回应相同的最新用户问题；chosen 表示数据标注者更偏好，不等于客观正确。"},{title:"整对分组与校验",description:"按首问分组划分训练和验证，整对进入同一集合。重复 prompt、相同候选、空白文本拒绝；超长候选不静默截断。"},{title:"只为最终回答计分",description:"系统、用户、历史助手、角色前缀和填充 label 为 −100。最终候选正文与真实 EOS 保留 ID，用于下一节序列 log 概率之和。"}],
    formula:String.raw`(x,y_w,y_l),\quad\log\pi(y\mid x)=\sum_{t\in\mathrm{completion+EOS}}\log\pi(y_t\mid x,y_{<t})`,
    formulaNote:"w 是偏好候选，l 是非偏好候选。本课求和而不按候选长度平均，长短回答可能有不同的序列概率。",
    misconception:"不要把 chosen 和 rejected 当成两个独立 SFT 样本，也不能把历史助手回答一起计入最终候选分数。",
    quiz:{question:"DPO 的一条偏好记录要求两个候选共享什么？",options:["相同回答长度","相同 prompt 上下文","相同 token ID"],answer:1,explanation:"偏好比较针对同一条件下的两个回答，不要求长度相同。"},
  },
  "dpo-train": {
    objective:"理解参考相对概率差，执行真实 LoRA DPO，并区分偏好指标与自由生成质量。",
    steps:[{title:"保存不可变参考",description:"策略与参考从相同的 SFT 或预训练权重开始。参考适配器与所有偏好对的参考分数保存为快照，优化器不包含参考参数。"},{title:"优化参考相对间隔",description:"求策略 chosen、rejected 的回答 log 概率之和，减去固定参考对应差，再乘 beta。最小化 −log sigmoid，让偏好间隔增加。"},{title:"只用训练对更新",description:"验证只用于评估和选择最佳权重。保存策略、参考、AdamW 和采样 RNG，支持精确续训；基座与参考分别校验。"}],
    formula:String.raw`L_{\mathrm{DPO}}=-\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}\right]\right)`,
    formulaNote:"本课使用 sigmoid DPO，无额外 SFT 项和长度归一化。初始策略=参考时，间隔为 0、损失为 ln 2≈0.6931。",
    misconception:"不能每步把参考替换成当前策略；也不能把 beta 当成学习率。间隔增大不要求 chosen 的绝对概率每次都增加。",
    quiz:{question:"策略与固定参考完全相同时，单对 DPO 损失是多少？",options:["0","1","ln 2，约 0.6931"],answer:2,explanation:"参考相对差为 0，−log sigmoid(0)=−log(1/2)=ln 2。"},
  },
  "dpo-compare": {
    objective:"用受控生成比较权重变化，理解为什么偏好损失下降不等于对话一定改善。",
    steps:[{title:"固定左侧基线",description:"左侧为创建时参考，右侧为所选当前或最佳 DPO 适配器。SFT 来源后续训练不改变本实验基线，结果注明实际模型步数。"},{title:"共享输入与采样",description:"两侧系统、历史、用户问题、温度、Top-k、长度和种子相同。追问共用右侧历史，按完整轮次裁剪，避免两侧历史不同混淆比较。"},{title:"检查未见问题与副作用",description:"对照训练问题和未见问题，观察是否更符合标注，也留意重复、回避、提前结束。系统不自动判赢家，偏好准确率不是人类胜率。"}],
    formula:String.raw`\mathrm{before}=\mathrm{Decode}(\pi_{\rm ref},x,s),\qquad\mathrm{after}=\mathrm{Decode}(\pi_\theta,x,s)`,
    formulaNote:"两侧使用同一输入 x 和种子 s；温度 0 为贪心，此时种子与 Top-k 不影响选择。不同权重下相同种子不保证输出相同。",
    misconception:"训练间隔为正，只说明相对参考更偏向 chosen；它不等于 chosen 的原始概率一定超过 rejected，更不保证自由生成正确。",
    quiz:{question:"为了隔离权重变化，两侧生成应保持哪项一致？",options:["输入历史和采样参数","最后生成的文本","训练后的模型参数"],answer:0,explanation:"保持输入和采样设置相同，才能把观察到的差异主要归因于权重变化。"},
  },
};
export const dpoTransfer: Record<string, StudyQuestion> = {
  "dpo-data":{question:"prompt 中已有一轮助手回答，它是否进入最终候选 log 概率？",options:["全部计分","只计最终候选及 EOS，历史作条件","删除所有历史"],answer:1,explanation:"本课只优化最后一条 completion，历史仍保留在模型输入。",whyWrong:["历史助手是两个候选共享的条件，不是最终比较目标。","","删除历史会改变候选的条件分布。"]},
  "dpo-train":{question:"策略 chosen−rejected=-2，参考差=-4，beta=0.1，偏好间隔是多少？",options:["−0.2","−0.6","0.2"],answer:2,explanation:"间隔=0.1×[−2−(−4)]=0.2。虽然原始差为负，相对参考仍更偏向 chosen。",whyWrong:["还需要减去参考差，不能只缩放策略差。","减去负数等于相加，不是再减 4。",""]},
  "dpo-compare":{question:"训练偏好准确率达到 100%，能推出什么？",options:["训练对的参考相对间隔均大于 0","所有新问题回答正确","人类胜率是 100%"],answer:0,explanation:"该指标只度量已标注候选的参考相对间隔，不是独立生成评估。",whyWrong:["","新问题的泛化不能从训练指标直接推出。","没有人类盲测，不能称作人类胜率。"]},
};
export const dpoRecall: Record<string, readonly [string,string,string]> = {
  "dpo-data":["序列长度可以不同，比较的前提是相同上下文。","","候选文本不同通常意味着 token 序列不同。"],
  "dpo-train":["初始间隔为 0，但 sigmoid(0)=1/2，损失不为 0。","损失是负对数，不是固定 1。",""],
  "dpo-compare":["","输出是被观察的结果，不应预先要求相同。","两侧正是要比较优化前后不同的权重。"],
};
export const dpoPredictions = {
  "dpo-data":{question:"一个候选更长，就应该把序列 log 概率除以长度吗？",options:["本课不除，使用标准求和","总是除","只对 chosen 除"],observation:"检查两候选的有效 token 数，确认本课使用序列求和，保留长度影响。"},
  "dpo-train":{question:"如果每步把参考更新成策略，仍是本课的固定参考 DPO 吗？",options:["是","不是","只与步数有关"],observation:"比较初始损失、训练间隔与不可变参考指纹。"},
  "dpo-compare":{question:"偏好损失下降后，自由生成一定更好吗？",options:["一定","需要实际比较","只看训练准确率"],observation:"用训练问题和未见问题对比两侧输出，记录重复与提前结束。"},
};
export const dpoDeepDives: Record<string, LessonDeepDive> = {
  "dpo-data":{example:{title:"从条件到最终候选",steps:["prompt 可以包含 user、assistant、user 三条消息，两候选在最后用户问题之后接入。","分别序列化 prompt+chosen 和 prompt+rejected，模型看到相同的对话上下文。","只保留最后候选正文与 EOS 的标签；历史助手目标同样设为 −100。"],tex:String.raw`\log\pi(y\mid x)=\sum_{t=1}^{|y|+1}\log\pi(y_t\mid x,y_{<t})`,takeaway:"计分目标与上下文不同；EOS 也是候选序列的结束目标。"},observations:[{action:"导入含多轮 prompt 的偏好对并校验。",expected:"历史助手 token 为灰色，最终候选与 EOS 为亮色。"},{action:"把 rejected 改成与 chosen 相同，或缩短最大长度。",expected:"相同回答报错，超长样本明确拒绝，不自动截断。"}],terms:[{term:"偏好对",definition:"在相同条件下，标注更偏好与较不偏好的两个回答。"},{term:"序列 log 概率",definition:"回答各 token 条件 log 概率之和，通常是负数，不是 token 平均概率。"}],connection:"下一节将这两个序列分数与固定参考结合，执行真实 DPO。",next:"dpo-train"},
  "dpo-train":{example:{title:"减去参考，得到偏好间隔",steps:["假设策略的 chosen−rejected log 概率差为 −2，固定参考对应差为 −4。","相对改善为 −2−(−4)=2；beta=0.1 时，间隔 m=0.2。","sigmoid(0.2)≈0.5498，DPO 损失=−ln(0.5498)≈0.5981，小于初始 ln 2。"],tex:String.raw`m=0.1[-2-(-4)]=0.2,\quad L=-\log\sigma(0.2)\approx0.5981`,takeaway:"原始 chosen 概率仍可低于 rejected，但相对于参考的偏好间隔已经为正。"},observations:[{action:"创建实验，不训练，查看初始损失与准确率。",expected:"损失约 0.6931，间隔为 0；严格大于 0 的准确率将平局记为 0%。"},{action:"训练 20 步，暂停刷新后继续。",expected:"策略和指标变化，参考指纹与基座保持不变；验证损失可能上升。"}],terms:[{term:"固定参考",definition:"创建时模型的不可变快照，用于计算 DPO 的 log 概率比。"},{term:"Beta",definition:"缩放策略与参考相对偏好差的系数；不是 AdamW 学习率。"}],connection:"下一节使用参考与策略的真实生成，检查偏好指标之外的行为。",next:"dpo-compare"},
  "dpo-compare":{example:{title:"同样的追问，需要同样的历史",steps:["第一轮给两侧相同用户问题和采样设置，分别生成回答。","第二轮若分别携带各自回答，输入条件已经不同，权重与历史两种变化会混在一起。","本课两侧共用优化后第一轮回答作为历史，再给同一追问；超过长度预算时两侧按相同规则裁剪。"],tex:String.raw`x_{\rm before}=x_{\rm after},\quad\theta_{\rm before}=\theta_{\rm ref}`,takeaway:"受控对比会共用历史；独立分叉聊天是另一种评估问题。"},observations:[{action:"在第 0 步先生成一次对比，再训练后重试。",expected:"初始两侧相同；训练后可能产生差异，也可能保持相同。"},{action:"使用未见过的问题，切换当前与最佳验证权重。",expected:"显示实际权重步数；更低偏好损失不保证每条回答都改善。"}],terms:[{term:"参考相对准确率",definition:"候选偏好间隔大于 0 的比例，平局记为未胜。"},{term:"生成评估",definition:"直接观察模型解码得到的文本，关注偏好、准确性与退化，不能用训练指标代替。"}],connection:"下一阶段通过在线采样与自定义奖励进行 GRPO 强化学习微调。",next:"grpo-rewards"},
};
