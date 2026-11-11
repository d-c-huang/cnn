# 第八阶段：自定义问答的真实 LoRA SFT 与对话

本阶段在已缓存的固定版本 Pythia-14M 预训练基座上，使用自定义问答进行 **回答监督 SFT**：系统、用户和角色前缀只作上下文，只有助手回答与 EOS 参与交叉熵。参数更新仍使用 PEFT LoRA；原基座保持冻结。

## 开始学习

安装现有 `requirements.txt` 后运行 `npm run dev`，选择「08 SFT」：

1. [问答数据与回答掩码](http://127.0.0.1:5173/#sft-data)：编辑/导入数据，检查真实分词、回答掩码与训练/验证划分。
2. [真实 LoRA 指令微调](http://127.0.0.1:5173/#sft-train)：从草稿建立快照，真实训练、暂停续训，比较回答损失。
3. [SFT 多轮对话生成](http://127.0.0.1:5173/#sft-chat)：加载当前/最佳适配器，用相同模板进行真实多轮生成。

三节都有操作任务、双题自测、解析、手算和学习进度。第七阶段的纯领域文本 LoRA 实验保持独立。

复用 [Pythia-14M 固定预训练基座](https://huggingface.co/EleutherAI/pythia-14m)，revision 为 `cf967c0a9a04383db6f7b1108d86b2962634b4ac`，缓存和下载规则见 [LoRA 教程](lora-finetuning-tutorial.md)。没有新增大模型下载，也不依赖 TRL Trainer。默认示例使用英文；中文文本可导入，但该小型英文基座不保证中文或通用助手能力。

## 自定义问答数据

支持 JSON 数组或一行一条的 JSONL，文件最多 200 KB。训练集要求 5–64 条样本、至少 5 个不同首问，消息总字符数不超过 20,000。每个问题、回答或消息最长 1,000 字符，空白文本会拒绝。

单轮格式：

```json
[
  {"question": "What is Zhiwei?", "answer": "Zhiwei is a learning lab."},
  {"question": "What is LoRA?", "answer": "LoRA learns small adapter matrices."},
  {"question": "What is SFT?", "answer": "SFT learns from example questions and answers."},
  {"question": "Are base weights updated?", "answer": "No. The base weights stay frozen."},
  {"question": "How should I start?", "answer": "Start with a small experiment."}
]
```

也可混入多轮记录，每条最多四轮：

```json
{"messages":[
  {"role":"user","content":"What is Zhiwei?"},
  {"role":"assistant","content":"Zhiwei is a learning lab."},
  {"role":"user","content":"What do we learn?"},
  {"role":"assistant","content":"We learn how models train."}
]}
```

`messages` 必须从 user 开始，user/assistant 交替，以 assistant 结束。共享系统提示在页面单独设置，不放进每条数据的 messages。重复用户消息序列会拒绝；首问仅大小写或空白不同视为相同组。先按首问分组，再用固定种子抽取前 80% 组训练、其余组验证；同一首问的多种后续对话保持在同一集合。分组不是语义去重，近义改写仍应人工检查。

点击「预览与校验数据集」后会显示真实 token、标签、有效回答目标数、样本长度和划分。单样本总长度可设 64 / 128 / 256，包含系统、用户、角色前缀、回答和 EOS。超长样本明确报错，创建实验时拒绝使用，不静默截断回答。

数据草稿、系统提示与新实验配置保存在当前浏览器，可跨课程与刷新保留；训练记录另在服务端保存快照。编辑草稿不会改变已创建实验。

## 统一对话模板

Pythia 原基座没有原生聊天模板，本课程使用明确命名的教学模板 `zhiwei-sft-v1`，而非把它称为模型官方模板：

```text
System: You are Zhiwei. Answer briefly and clearly.
User: What is Zhiwei?
Assistant: Zhiwei is a learning lab.<EOS>
```

模板为每个助手回答正文及 EOS 设置 Jinja `generation` 区间，调用 Transformers 的真实分词与 assistant mask。用户文本、系统提示、`Assistant:` 前缀和回答后的分隔换行不参与目标损失。多轮样本中所有助手回答都受监督。

```text
input_ids: 原始完整 token 序列
labels:    非回答位置为 -100，回答及真实 EOS 保留 token ID
```

问题仍然作为模型输入；−100 不表示删除问题或修改输入 ID。右侧填充使用 EOS ID，但同时设置 `attention_mask=0`、`label=-100`，因此填充 EOS 与真实回答 EOS 的监督含义不同。

损失显式只右移一次：

```python
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

位置 t 的 logits 预测位置 t+1 的标签。首个回答 token 由它前一位置预测。平均分母只包含右移后 label 不为 −100 的目标数；评估按有效回答目标数加权，既不按全部输入长度平均，也不按每条样本简单平均。

## 真实 SFT 与保存

默认 rank=8、alpha=16，共 24,576 个可训练 LoRA 参数；原始 14,067,712 个基座参数冻结。默认 batch size 2、学习率 0.003、seed 7，AdamW 权重衰减 0，梯度范数裁剪 1，LoRA dropout=0。

每步从训练样本中有放回抽样，按该批最长样本右侧补齐。不同样本不拼接成同一个对话。每次请求最多 5 步，单实验最多 500 步；界面支持 1 / 5 / 20 / 100 步。每批结束对两份完整数据计算回答交叉熵，验证不参与梯度更新，没有额外独立测试集。

默认 14 条问答分为 11 条训练、3 条验证。以下是本机真实默认记录：

| 保存步数 | 训练回答损失 | 验证回答损失 |
| ---: | ---: | ---: |
| 0 | 5.6933 | 5.9344 |
| 20 | 2.2706 | 5.3450 |
| 40 | 1.3127 | 5.1473 |
| 60 | 0.7762 | 4.9421 |
| 80 | 0.6515 | 5.4143 |
| 100 | 0.4833 | 5.3369 |

默认当前步 100、贪心解码时，对训练问题“What is Zhiwei?”实际生成“Zhiwei is a learning lab.”并以 EOS 结束。这说明模型学会了部分示范；验证损失仍较高，不能把训练问题上的表现当成广泛泛化或回答可靠性证明。其他版本、种子与数据可能得到不同结果。

`runs/sft/<id>/checkpoint.pt` 保存固定基座身份、模板版本、数据/划分快照与指纹、当前/最佳适配器、AdamW、步数、采样 RNG 和评估记录。每批比较基座参数 SHA-256 后才允许保存。临时文件原子替换，保存失败保留旧记录。

暂停或离开只停止后续请求，已提交的一批仍可能保存；网络中断后先刷新记录。服务重启后可恢复适配器、优化器和采样状态继续训练。陈旧的 `expected_step` 返回 409，避免重复更新。使用默认单进程 API。

## 多轮对话与上下文

对话使用实验快照中的系统提示和同一模板，最后只追加 `Assistant:` 生成前缀，不添加真实回答。当前/最佳验证适配器的实际步数在结果中显示，步 0 不算完成已训练模型的对话任务。

浏览器每次携带最近四轮与最新问题；服务根据上下文预算继续删除最早的完整 user+assistant 轮次。系统提示与最新问题保留，无法容纳时明确报错，不从最新问题中间静默截断。

服务为新回答预留空间，实际生成满足 `上下文 token 数 + 新 token 数 ≤ max_length`。不采用生成中的 token 滑窗，所以系统和最新问题不会在回答中途被无提示切掉。

温度为 0 时贪心解码，Top-k 和种子不影响选词；正温度时可使用 Top-k=0 / 10 / 50 与局部随机种子采样。最多生成 64 个 token，停止原因包括 EOS、角色边界、回答长度上限、上下文上限。遇到新一轮 `User:` / `Assistant:` / `System:` 边界时不把伪造轮次追加为用户消息。

当前对话仅保存在该页面内存，可导出 JSON。更换实验或适配器权重会清空历史。界面同时报告真实模型步数、上下文长度、被忽略的旧轮数与结束原因。小型基座可能重复、答非所问或立即结束；它不是通用可靠聊天助手。

## 适配器与模板一起导出

SFT ZIP 兼容标准 PEFT 适配器，并新增：

```text
adapter_model.safetensors
adapter_config.json
zhiwei_manifest.json
README.txt
chat_template.jinja
sft_format.json
```

`sft_format.json` 记录系统提示、最大长度、回答监督方式、模板版本与数据指纹。单独适配器不包含基座或 AdamW；精确续训使用另一份完整续训检查点。

加载固定版本基座和适配器后，可按导出的模板构造输入：

```python
import json
from pathlib import Path

adapter_dir = Path("解压后的 SFT 适配器目录")
format_info = json.loads((adapter_dir / "sft_format.json").read_text())
tokenizer.chat_template = (adapter_dir / "chat_template.jinja").read_text()
messages = [{"role": "system", "content": format_info["system_prompt"]},
            {"role": "user", "content": "What is Zhiwei?"}]
ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
# model 是已挂载该适配器的固定版本 Pythia-14M；按上述预算生成并在 EOS 停止。
```

## 用项目代码复现

```python
from backend.sft import RunConfig, TrainingRequest, ChatRequest, service

config = RunConfig().model_dump()
# config["dataset"] = 自定义 question/answer 或 messages 列表
print(service.preview(config)["tokens"])
run = service.create(config)
for _ in range(20):
    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=5))
result = service.chat(run["id"], ChatRequest(
    messages=[{"role": "user", "content": "What is Zhiwei?"}], checkpoint="latest",
))
print(result["reply"], result["finish_reason"])
```

接口前缀 `/api/sft`：`GET /status`、`POST /prepare`、`POST /preview`、`GET/POST /runs`、`GET /runs/{id}`、`POST /runs/{id}/train`、`POST /runs/{id}/chat`、`GET /runs/{id}/adapter?checkpoint=latest|best`、`GET /runs/{id}/checkpoint`。详细参数见 `/api/docs`。

```powershell
python -m unittest backend.tests.test_sft -v
python -m unittest discover -s backend/tests -v
npm run test:sft
npm run check
```

验证涵盖回答/EOS 掩码、真实目标右移、填充零损失、首问分组、过长样本拒绝、冻结和精确续训、验证答案不影响更新、历史裁剪、对话复现、模板导出以及 HTTP 参数。缓存完整时另执行真实 Pythia SFT 集成测试；不靠测试自动下载权重。

官方延伸阅读：[SFT 数据格式与回答监督](https://huggingface.co/docs/trl/sft_trainer)、[Transformers 对话模板](https://huggingface.co/docs/transformers/chat_templating)。本项目用显式 PyTorch 损失实现课程，不依赖 TRL Trainer。
