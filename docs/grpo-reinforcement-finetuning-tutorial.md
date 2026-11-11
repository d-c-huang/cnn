# 第十阶段：自定义奖励的真实 LoRA GRPO

本阶段使用已缓存 Pythia-14M 实现可在 CPU 运行的在线强化学习微调。每轮为同一训练问题真实采样 G 个回答，用自定义规则评分，计算组相对优势，再通过裁剪策略梯度与固定参考 KL 更新 LoRA。没有 critic、独立奖励模型，也没有把示例答案当作 SFT 标签或 DPO 偏好对。

推荐先完成 [SFT](sft-instruction-tuning-tutorial.md) 或 [DPO](dpo-preference-optimization-tutorial.md)，选择其适配器作为初始策略和固定参考。也可从原预训练基座开始观察算法，但小型英文模型不保证中文或通用对话能力。

## 开始学习

安装现有 `requirements.txt`，运行 `npm run dev`，选择「10 GRPO」：

1. [奖励规则与在线采样](http://127.0.0.1:5173/#grpo-rewards)：导入问题集和奖励规则，校验真实 token 预算，试算逐项奖励。
2. [真实 LoRA GRPO 更新](http://127.0.0.1:5173/#grpo-train)：选择 SFT / DPO 来源或预训练基座，在线采样并更新策略，查看奖励、优势、KL 和裁剪比例。
3. [强化学习前后对话对比](http://127.0.0.1:5173/#grpo-compare)：相同输入和采样设置下比较初始参考与 GRPO 策略，生成后再按规则评分。

三节均有操作任务、双题自测、手算、解析和持久化进度；全课程共十阶段、31 个实验。复用现有预训练缓存和 Transformers / PEFT / PyTorch 环境，不下载新模型，也不依赖 TRL Trainer。

## 自定义问题集

支持 JSON 数组或每行一条记录的 JSONL，文件最多 100 KB。训练要求 5–32 条记录、至少五个不同首问，总问题 / target 文本不超过 16,000 字符；每条消息和 target 最多 1,000 字符。

```json
[
  {"prompt":"What is Zhiwei?","target":"lab"},
  {"prompt":"What do we learn?","target":"models"},
  {"prompt":"How should I start?","target":"small"},
  {"prompt":"What is LoRA?","target":"adapter"},
  {"prompt":"Are base weights updated?","target":"frozen"}
]
```

`target` 是可选的评分元数据，不是模型输入，也不是必须生成的 token 标签。只有 contains_target / exact_target 规则要求每条记录填写非空 target。

多轮 prompt 支持 user / assistant 交替、从 user 开始且以 user 结束的消息列表，最多七条消息：

```json
{"prompt":[
  {"role":"user","content":"What is Zhiwei?"},
  {"role":"assistant","content":"Zhiwei is a learning lab."},
  {"role":"user","content":"What do we learn?"}
],"target":"models"}
```

完整重复 prompt 会拒绝；首问按大小写和空白标准化分组，同首问的历史变体放在同一集合。用固定种子划分 80% 组训练、20% 组验证，训练只抽训练问题。此分组不等于语义去重。

共享系统提示在页面单独设置。上下文总长度可选 64 / 128 / 256 tokens，训练每条回答预算可选 8 / 16 / 32；校验要求 `prompt tokens + 回答预算 ≤ 总长度`，不截断问题，也不会把 target 补进 prompt。

## 自定义奖励规则

编辑或导入 1–8 条 JSON 规则，文件最多 20 KB。每条产生 0 / 1 命中分，乘以 `weight` 后求和。权重默认 1，可为 −5 到 5 的有限非零数；负权重可以表达惩罚。

```json
[
  {"type":"contains_target","weight":1},
  {"type":"length","min_chars":8,"max_chars":100,"weight":0.25},
  {"type":"nonempty","weight":0.1}
]
```

| type | 命中条件 | 可用参数 |
| --- | --- | --- |
| nonempty | 去除首尾空白后非空 | weight |
| contains | 包含字面子串 | text、case_sensitive、weight |
| avoids | 不包含字面子串 | text、case_sensitive、weight |
| contains_target | 包含记录的 target | case_sensitive、weight |
| exact_target | 与 target 相同，压缩连续空白后比较 | case_sensitive、weight |
| length | 去除首尾空白后的字符数位于闭区间 | min_chars、max_chars、weight |
| ends_with | 以字面文本结尾 | text、case_sensitive、weight |
| json_object | 整段文本能解析为 JSON 对象，并具有指定顶层键 | required_keys、weight |

文本匹配默认不区分大小写。contains 是子串检查，不是语义匹配、正则或单词边界检查。JSON 不接受 NaN / Infinity，required_keys 只检查顶层键是否存在，不检查值的正确性。length 默认区间为 1–100，最长可设 4,000 字符。

例如回答包含关键词、长度不合规，关键词权重 2、长度权重 0.5，总分就是 `2×1 + 0.5×0 = 2`。页面「试算自定义奖励」直接显示每个组件、权重和总分，不调用模型训练。可以分别试算自然回答、空白回答、堆关键词、错误 JSON，以发现规则缺口。

avoids 可能给空回答加分；长度和关键词也可能鼓励模型输出无意义内容。建议与 nonempty 等条件组合，但组合规则仍不能自动验证事实、推理质量或人类满意度。配置采用固定规则类型，不执行 Python / JavaScript 代码。

## 在线采样与实际 token

创建实验后，问题、规则、对话模板、系统和配置全部成为服务端快照；修改浏览器草稿不会改变已有实验。

一轮训练执行以下操作：

1. 用持久化的局部 RNG 从训练集合有放回抽一个 prompt。
2. 使用当前策略、温度 **1**、完整 softmax（不做 Top-k）采样 G=2 / 4 条回答。这个策略是本轮 `old policy`。
3. 对采样文本计算规则奖励、组优势；保留原始采样 token IDs。
4. 在同一组上进行 μ=1 / 2 次真实梯度更新，旧策略的 token log 概率在组内固定。
5. 对完整验证集合逐问题做贪心解码、评分；保存状态，再开始下一轮的新采样。

固定温度与全词表采样确保行为分布与计算 ratio 时使用的原始旧策略概率一致。训练采样使用 KV cache；反向传播用完整原始序列重新评分，逐 token 梯度覆盖实际生成位置。

遇 EOS 停止并把该真实 EOS 纳入优化；达到回答预算则标为 `length`，保留已生成 token 参与训练，**不额外补 EOS**。本教学实现不屏蔽长度截断样本。训练中不按解码出的角色词提前停止；这类文本仍受奖励规则评分。前后聊天为了呈现单条助手回复，会在新角色边界停止。

系统、历史、最新问题和 Assistant: 前缀的 labels 为 −100；右侧填充同时设置 attention_mask=0、label=−100。采样 token 不经过解码再分词，避免改变实际采样动作。只右移一次，用位置 t 的 logits 给位置 t+1 的采样 token 求 log 概率。

## 组相对优势与裁剪目标

本课采用总体标准差（correction=0）：

```python
advantages = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-4)
```

奖励为 `[0, 0, 1, 1]` 时，均值与总体标准差均为 0.5，优势约为 `[-0.9998, -0.9998, 0.9998, 0.9998]`。优势在组内固定、停止梯度，不训练价值网络。

如果同组全部同分，优势全为零，奖励项没有区分信号。本课仍计算 KL 并执行 AdamW，已有动量也可能使参数变化。因此界面分别统计「策略实际改变」和「出现非同分组」，不把所有变化都当作奖励学习；操作任务需至少两轮、非同分组数大于零及真实参数变化。

每个实际生成 token 上：

```python
ratio = (logp_policy - logp_old.detach()).exp()
surrogate = minimum(ratio * advantage, clamp(ratio, 1-eps, 1+eps) * advantage)
delta = logp_reference.detach() - logp_policy
kl_k3 = delta.exp() - delta - 1
loss_per_token = -surrogate + beta * kl_k3
```

本课使用原始 GRPO 的归一化：**先对每条回答的有效生成 token 求均值，再对 G 条回答求均值**。这与按所有 token 总数归一化的 DAPO 等变体不同；存在长度与组内标准差相关偏差，不能声称实现了所有 GRPO 变体或当前 TRL 默认目标。

ratio 分母是生成当前组的旧策略；KL 参考则是创建时冻结的初始策略。二者从第二轮起通常不同。裁剪 ε 默认 0.2，KL β 默认 0.02；本课的 clip_fraction 统计实际因优势符号受到裁剪的有效 token 比例。KL 是采样 token 上的 k3 估计，不是全词表精确 KL。

μ=1 时第一次 ratio 数值等于 1，但对分子的梯度仍存在；μ=2 时同一组会复用一次，裁剪更容易发挥作用。代理损失可能为零或负数，不适合以「持续下降」作为训练成功标准。

默认 G=4、μ=2、学习率 0.00003、rank=8、alpha=16、LoRA dropout=0、AdamW 权重衰减=0、梯度范数裁剪=1。固定基座有 14,067,712 个参数，默认 24,576 个 LoRA 参数可训练。非有限损失或冻结校验失败时拒绝保存当前请求。

## 参考、评估与续训

初始策略可选预训练基座、已训练 SFT 或已训练 DPO 的当前 / 最佳适配器。从来源复制完整 LoRA 状态并继承 rank / alpha；系统提示、规则、预算使用 GRPO 草稿。固定参考独立保存，来源后续训练不会改变它。

曲线中的实线是**更新前最近采样组**的平均规则分，虚线是**更新后完整验证集贪心回答**的平均规则分。它们的采样方式和问题集合不同，不能当作两个同分布总体均值直接比较。验证只评分，不反传，也不消耗训练 RNG。

按验证贪心规则分最高保存最佳适配器；平分保留较早权重，所以最佳可能仍是第 0 轮。来源 SFT / DPO 可能已见过验证问题，没有额外独立测试集。奖励与验证分波动、生成不变或退化都可能发生。

界面显示采样组均值 / 标准差、KL、裁剪比例、逐轮代理损失、优化器更新次数、非同分组数、完整验证文本以及最近真实采样组。每条采样展示正文、原始 token、EOS / 长度结束、奖励组件和优势。

单实验最多 100 轮。页面支持 1 / 5 / 20 轮，逐轮提交与保存；API 单请求最多两轮。暂停或离开只阻止后续提交，当前请求仍可能保存，网络中断后先刷新。旧 expected_step 返回 409，避免重复更新。使用默认单进程 API。

`runs/grpo/<id>/checkpoint.pt` 保存策略当前 / 最佳适配器、固定参考、规则版本与数据指纹、来源、优化器、采样 RNG、累计计数、指标和最近采样组。每批核对基座 SHA 与参考摘要，原子替换文件；重启后可精确续训。

## 本机在线训练观察

浏览器验证使用 16 条自定义问题、五条规则（target 关键词、8–70 字符、非空、句号结尾、不含指定回避词），从 SFT 第 106 步开始。实际完成 22 轮、44 次优化器更新，其中 11 个采样组非同分，基座与参考冻结校验通过。

第 22 轮四条回答奖励为 1.6、1.6、0.6、1.6，均值为 1.35、总体标准差约 0.433；优势约为 0.577、0.577、−1.732、0.577。第一轮所有回答都得到 1.6，则组优势全为 0，不能把满分当作存在奖励学习信号。

初始与第 22 轮的验证贪心规则分均为 0.6；吉祥物问题的两侧文本均为 The lab mascot is a curious cat.，各得 1.6。这验证了真实采样、规则打分和参数更新，但并不意味着生成质量提升或贪心文本必然改变。具体数值依赖问题、规则、初始权重及采样种子。

## 优化前后的对话对比

两侧共享系统、模板、历史、最新用户问题、温度、Top-k、长度和种子；左侧为创建时参考，右侧为当前 / 最佳 GRPO 策略。追问共用右侧回答作为历史，以固定输入条件，不是两条独立分叉对话。最多最近四轮，再按整轮裁剪；系统与最新问题无法容纳时明确报错。

「评分 target」只在生成完成后用于奖励，不进入提示。未填写时省略 contains_target / exact_target 规则，界面明确提示，此时不能把剩余规则分当成完整训练奖励。改变评分 target，在相同采样设置下不应改变生成 token，只能改变分数。

对比温度 0 为贪心；其他温度可配 Top-k 与局部随机种子，设置同时用于两侧。结果显示实际 GRPO 轮次、上下文长度、结束原因和规则分；切换模型或权重会清空历史。优化后无可见回答时需开始新对比。可导出带真实上下文和采样参数的 JSON。

不要自动把规则分更高的一侧判作更好。应同时检查事实、相关性、可读性、重复、回避、EOS 提前结束，并用未用于更新的问题观察泛化。规则优化不等于获得数学推理能力。

## 导出与 Python / HTTP 接口

适配器 ZIP 包含标准 PEFT `adapter_model.safetensors`、`adapter_config.json`、基座 manifest、README、`chat_template.jinja` 和 `grpo_format.json`。格式文件带规则与模板版本、系统、长度、来源、参考摘要和数据指纹。从 SFT / DPO 开始时，该策略适配器已包含所有初始 LoRA 状态与本轮更新，只需原固定基座加此适配器，不再叠加另一份来源适配器。

ZIP 用于策略推理，不带参考、AdamW 或采样 RNG。精确续训和恢复前后对照需完整检查点。原基座离线加载与标准 PEFT 方式见 [LoRA 教程](lora-finetuning-tutorial.md)，使用导出的对话模板见 [SFT 教程](sft-instruction-tuning-tutorial.md)。

```python
from backend.grpo import RunConfig, TrainingRequest, ComparisonRequest, service

config = RunConfig().model_dump()
# config.update(source_type="sft", source_run_id=已训练SFT实验ID)
# config["rules"] = [{"type":"contains", "text":"lab", "weight":1}]
run = service.create(config)
for _ in range(5):
    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=1))
print(run["last_rollout"])
result = service.compare(run["id"], ComparisonRequest(
    messages=[{"role":"user", "content":"What is Zhiwei?"}],
    reward_target="lab", temperature=0,
))
print(result["before"]["reply"], result["after"]["reply"])
```

`/api/grpo` 提供 `GET /status`、`POST /prepare`、`POST /reward`、`POST /preview`、`GET/POST /runs`、`GET /runs/{id}`、`POST /runs/{id}/train`、`POST /runs/{id}/compare`、`GET /runs/{id}/adapter?checkpoint=latest|best`、`GET /runs/{id}/checkpoint`。详细参数见 `/api/docs`。

```powershell
python -m unittest backend.tests.test_grpo -v
python -m unittest discover -s backend/tests -v
npm run test:grpo
npm run check
```

验证包含奖励边界和权重默认值、组优势与零方差、裁剪梯度方向、KL / 旧概率停止梯度、EOS / 长度与填充掩码、在线真实 token 采样、参数与参考冻结、SFT / DPO 快照、精确续训、验证不消耗训练 RNG、评分 target 不泄漏、标准 PEFT 重载和 HTTP 参数。缓存完整时执行真实 Pythia 在线 GRPO 测试，不为测试自动下载权重。

参考：[DeepSeekMath / GRPO 原论文](https://arxiv.org/abs/2402.03300)、[Hugging Face GRPO 说明与不同目标变体](https://huggingface.co/docs/trl/grpo_trainer)。
