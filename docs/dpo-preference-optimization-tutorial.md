# 第九阶段：自定义偏好数据的真实 LoRA DPO

本阶段在已有 Pythia-14M 缓存上，用 PyTorch 显式实现标准 sigmoid Direct Preference Optimization（DPO）。只训练策略的 LoRA，不另外训练奖励模型，不附加 SFT 损失。推荐先完成 [SFT 教程](sft-instruction-tuning-tutorial.md)，再把其适配器作为初始策略和固定参考；也支持直接从预训练基座开始观察 DPO 公式。

## 开始学习

安装现有 `requirements.txt` 后运行 `npm run dev`，选择「09 DPO」：

1. [偏好数据与候选计分](http://127.0.0.1:5173/#dpo-data)：编辑或导入 JSON / JSONL，校验两候选及真实 token 掩码。
2. [真实 LoRA DPO 优化](http://127.0.0.1:5173/#dpo-train)：选择 SFT 来源或预训练基座，创建固定参考，真实训练、暂停与续训。
3. [优化前后对话对比](http://127.0.0.1:5173/#dpo-compare)：相同输入和采样条件下，对照创建时参考与 DPO 当前 / 最佳适配器。

三节均包含操作任务、两道自测、解析、手算和进度记录。全课程现为九阶段、28 个实验。默认使用英文示例；14M 小型英文基座不保证中文或通用助手能力。

## 自定义偏好对

JSON 数组或每行一个对象的 JSONL，文件最多 200 KB。需要 5–64 条记录、至少 5 个不同首问，总消息文本不超过 20,000 字符；每条消息和候选最长 1,000 字符。每条的 `chosen`、`rejected` 都是非空字符串且内容不同：

```json
[
  {"prompt":"What is Zhiwei?","chosen":"Zhiwei is a learning lab.","rejected":"I do not know. Try something else."},
  {"prompt":"What is LoRA?","chosen":"LoRA learns small adapter matrices.","rejected":"LoRA updates every base weight."},
  {"prompt":"Are base weights updated?","chosen":"No. The base weights stay frozen.","rejected":"Yes. All weights are updated."},
  {"prompt":"Why use validation data?","chosen":"Validation checks unseen examples.","rejected":"Validation is used for every gradient update."},
  {"prompt":"What is the next step?","chosen":"Try one change and inspect the result.","rejected":"Change every setting at once."}
]
```

支持多轮 prompt，但两个候选仍是对最后用户问题的单条回答。列表必须 user / assistant 交替、以 user 开始及结束，最多七条消息：

```json
{"prompt":[
  {"role":"user","content":"What is Zhiwei?"},
  {"role":"assistant","content":"Zhiwei is a learning lab."},
  {"role":"user","content":"How should I start?"}
],"chosen":"Start with a small experiment.","rejected":"Do not try any experiments."}
```

共享系统提示在页面独立设置，不写进 prompt 列表。`chosen` 是标注者偏好的候选，不是客观正确性保证。默认负例仅供教学；实际偏好应根据一致标准人工检查，避免两候选无关、矛盾标注、模板特征泄漏和长度偏差。

按标准化首问分组，用局部随机种子把前 80% 组用于训练、其余组验证；同一首问的后续对话不会跨集合，整对始终在同一集合。完整重复 prompt 拒绝，不做语义近似去重。每候选最大完整输入为 64 / 128 / 256 tokens，包含系统、历史、角色和 EOS；超长明确报错，创建实验时拒绝，不静默截断。

草稿保存在当前浏览器。创建实验后数据与配置成为服务端快照，后续编辑不会改变原实验。

## 真实序列计分与 DPO

沿用 SFT 的 `zhiwei-sft-v1` 教学对话模板。分别分词 `prompt+chosen`、`prompt+rejected`，只为最终候选正文与真实 EOS 保留标签；系统、用户、历史助手回答、角色前缀、分隔换行及右侧填充标签为 `−100`。历史依然是输入上下文。

```python
targets = labels[:, 1:]
mask = targets != -100
logps = logits[:, :-1].log_softmax(-1)
chosen_token_logps = logps.gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
sequence_logp = chosen_token_logps.masked_fill(~mask, 0).sum(-1)
```

仅右移一次，按序列求和，不按 token 数平均。填充使用 EOS ID，但 `attention_mask=0`、`label=−100`；真实 EOS 则参与计分。

令 `pc`、`pr` 为策略的 chosen / rejected 序列 log 概率，`rc`、`rr` 为固定参考对应分数：

```python
margin = beta * ((pc - pr) - (rc - rr))
loss = -torch.nn.functional.logsigmoid(margin).mean()
```

每个偏好对权重相同。策略与参考初始相同，所以间隔为 0、损失约 `ln 2 = 0.6931`。例如策略差为 −2、参考差为 −4、beta=0.1，则间隔为 0.2，损失约 0.5981。原始 chosen 概率仍可能低于 rejected；DPO 优化的是相对参考的偏好变化。它也不要求 chosen 的绝对概率每一步都增加。

`beta` 是参考相对 log 概率差的缩放系数，同时具有 DPO 参考约束解释；它与 AdamW 学习率不同。本课默认 beta=0.1、学习率 0.0001、batch=2 对、rank=8、alpha=16、seed=7、LoRA dropout=0、权重衰减=0、梯度范数裁剪=1。基座 14,067,712 个参数冻结，默认 24,576 个 LoRA 参数可训练。

## 初始策略、固定参考与保存

选择已训练 SFT 的当前或最佳适配器时，从其快照复制相同初始策略与参考，并继承 rank / alpha。系统提示和最大长度使用偏好草稿，应主动保持与所需行为一致。来源 SFT 的后续训练不会改变 DPO 参考。直接选预训练基座时，初始 LoRA B=0，策略和参考都等价于该基座。

创建时为所有偏好对预计算参考分数，此后训练只读取固定分数，不更新参考参数。每步有放回抽取训练对，对 chosen / rejected 一起右侧填充并进行真实反向传播。验证对从不参与 DPO 更新；每批评估完整训练与验证数据，没有额外独立测试集。来源 SFT 可能已见过对应问题，因此 DPO 验证集不能自动称为对整个训练历史完全未见的数据。

界面显示 DPO 损失、参考相对平均间隔及偏好准确率（间隔严格大于 0 的比例，初始平局记为 0%）。这是标注候选指标，不是人类盲测胜率，也不是对话正确率。按验证 DPO 损失选择最佳适配器。

每次请求最多 5 步，界面提供 1 / 5 / 20 / 100 步，单实验上限 500 步。暂停或离开停止后续提交，已提交批仍可能保存；中断后先刷新。`expected_step` 防止重复更新。默认单进程 API。

`runs/dpo/<id>/checkpoint.pt` 保存数据、划分、模板版本、来源、固定参考适配器与分数、当前 / 最佳策略适配器、AdamW、采样 RNG 及指标。基座 SHA-256 与参考适配器摘要经过核对；文件原子替换。重启可精确续训，完整检查点也可下载备份。

## 优化前后对话

左侧始终是创建时参考（可为 SFT 快照），右侧是所选 DPO 策略。两侧使用相同系统、历史、问题、模板、温度、Top-k、长度与种子；模型实际推理，不直接返回数据集答案。第 0 步两侧在同设置下相同，训练后也可能保持相同。

追问共用右侧上轮回答作为历史，以固定输入条件；不是各自携带不同回答的分叉聊天。浏览器最多提交最近四轮，两侧服务再按相同规则丢弃最早完整轮次，保留系统与最新问题，为回答预留预算。若系统与最新问题无法容纳则明确报错。新回答加上下文不超过实验最大长度，生成过程中不滚动丢弃最新问题。

温度 0 为贪心；此时 Top-k 和种子不影响选择。正温度用相同局部随机种子分别采样。遇 EOS、角色边界、回答长度或上下文预算停止。结果显示实际 DPO 步数、上下文长度、生成数与停止原因，导出的对比 JSON 包含真实输入与采样参数。切换实验或优化后权重清空历史；没有可见优化后回答时开始新对比。

比较训练问题、验证问题和新的未见问题，留意重复、回避及语言退化。偏好损失下降不保证每条生成更好；小数据和相似负例容易过拟合，不自动判定赢家。

## 本机真实对照观察

调试时使用较高学习率 0.0003、同一句通用 rejected 的 16 条偏好数据，从现有 SFT 第 106 步开始训练 DPO。DPO 第 106 步训练损失约 0.0004、验证损失约 0.092，但实际生成出现重复词句；最佳验证第 41 步在吉祥物问题上立即 EOS。指标改善没有转化为可用回答。

因此课程默认采用每个问题对应的不同负例，并将学习率调低为 0.0001。另一次相同 SFT 来源的 16 对实验中，第 20 / 60 / 100 步训练损失约 0.4291 / 0.1924 / 0.0909；吉祥物问题仍生成原来的 cat，没有按偏好改成 owl。这同样是有效观察：训练优化真实发生，贪心生成却可能保持不变。数值依赖具体数据与来源，不应承诺复现任意一次文本变化。

建议先训练 5–20 步，查看输出再继续。若出现退化，尝试减少步数和学习率、改进候选对，并保留创建时参考进行比较。

## 适配器导出与接口

ZIP 包含标准 PEFT `adapter_model.safetensors`、`adapter_config.json`、基座 manifest、README、`chat_template.jinja` 和 `dpo_format.json`。DPO 格式文件包含系统、长度、beta、目标定义、SFT 来源、参考摘要与数据指纹。若从 SFT 开始，这份策略适配器已包含其初始 LoRA 状态上的全部更新，加载时只需固定版本原基座加此适配器，不再叠加一份 SFT。

ZIP 用于策略推理，不包含参考适配器、参考分数或 AdamW；精确续训、恢复前后对比需完整检查点。固定基座版本与离线加载方式见 [LoRA 教程](lora-finetuning-tutorial.md)，使用导出的模板构造对话见 [SFT 教程](sft-instruction-tuning-tutorial.md)。

```python
from backend.dpo import RunConfig, TrainingRequest, ChatRequest, service

config = RunConfig().model_dump()
# config["dataset"] = 自定义偏好记录列表
# config["source_sft_id"] = 已训练SFT实验ID
run = service.create(config)
for _ in range(4):
    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=5))
result = service.compare(run["id"], ChatRequest(
    messages=[{"role":"user", "content":"What is Zhiwei?"}], temperature=0,
))
print(result["before"]["reply"], result["after"]["reply"])
```

接口前缀 `/api/dpo`：`GET /status`、`POST /prepare`、`POST /preview`、`GET/POST /runs`、`GET /runs/{id}`、`POST /runs/{id}/train`、`POST /runs/{id}/compare`、`GET /runs/{id}/adapter?checkpoint=latest|best`、`GET /runs/{id}/checkpoint`。详细参数见 `/api/docs`。

```powershell
python -m unittest backend.tests.test_dpo -v
python -m unittest discover -s backend/tests -v
npm run test:dpo
npm run check
```

验证覆盖公式与梯度方向、序列求和、单次右移、填充和历史掩码、分组与长度、真实 LoRA 更新、固定参考、验证隔离、精确续训、初始前后生成一致、采样 RNG 隔离、标准 PEFT 导出重载和 HTTP 参数。缓存完整时执行真实 Pythia DPO 集成测试，不为测试自动下载模型。

参考：[DPO 原论文](https://arxiv.org/abs/2305.18290)、[Hugging Face DPO 数据与训练说明](https://huggingface.co/docs/trl/dpo_trainer)。课程采用上述标准 sigmoid 目标，用显式 PyTorch 实现，不依赖 TRL Trainer。
