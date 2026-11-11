# 第六阶段：真实训练 Transformer 语言模型

在第五阶段的 Bigram 基础上，本阶段实现一个小型 GPT 风格、decoder-only 字符 Transformer：从随机参数开始，用 PyTorch 在 CPU 上真实训练，保存检查点并自回归生成文本。无需预训练权重、GPU 或外部语料下载。

启动 `npm run dev`，在侧栏选择「06 Transformer」：

1. [Transformer 与因果注意力](http://127.0.0.1:5173/#transformer-architecture)：追踪嵌入、位置、因果多头注意力与前馈网络，点选查询位置检查可见范围。
2. [真实训练 Transformer](http://127.0.0.1:5173/#transformer-train)：创建语料与模型配置的快照，真实训练、暂停、查看训练/验证曲线，并保存当前及最佳验证权重。
3. [Transformer 文本生成](http://127.0.0.1:5173/#transformer-generate)：选择实验与权重，调节温度、Top-k 和种子生成文本，查看实际注意力头。

三节都接入任务判定、双题自测、手算示例、术语与学习进度。原有 Bigram 课程保留独立进度。

## 模型计算

```text
token IDs [B,T]
  → Token embedding + learned position embedding [B,T,D]
  → 1–2 × Pre-LN Transformer block
      x = x + MultiHeadCausalAttention(LayerNorm(x))
      x = x + Linear(GELU(Linear(LayerNorm(x))))
  → LayerNorm → Linear [B,T,V] logits
```

默认上下文长度 16、嵌入维度 32、2 个注意力头、2 层，前馈隐藏维度为 `4D`。可改为上下文 8 / 16 / 32、维度 32 / 64、头数 2 / 4、层数 1 / 2。所有组合均满足 `D % H == 0`。模型不使用 Dropout，不共享输入嵌入与输出投影权重。

每个注意力头计算：

```text
scores = Q Kᵀ / sqrt(D / H)
scores[未来位置] = -inf
attention = softmax(scores)
output = attention V
```

虽然训练会并行处理多个位置，每个位置只能读取当前与过去。输入当前位置对应的目标是原文下一字符，因此未来位置若不屏蔽，就会泄漏标签。

初始化在 meta 设备构造结构，转到 CPU 后用本地 `torch.Generator` 初始化参数。训练窗口采样和生成采样也使用各自的 Generator，不修改进程全局随机状态。

## 数据与真实训练

延用项目自编的小猫与四季语料，也可编辑 40–2,000 个 Unicode 字符的文本。按顺序先取前 80% 训练、后 20% 验证；词表仅使用训练文本，并保留 ID 0 给 `<UNK>`。训练部分含 2–95 种字符，标点、空格和换行均是 token。验证新字符映射到 UNK，并显示数量。

每个训练步从训练部分随机采样连续窗口，批大小可选 8 / 16 / 32：

```text
input  = ids[start : start+C]
target = ids[start+1 : start+C+1]
loss = cross_entropy(logits.reshape(-1,V), target.reshape(-1))
zero_grad → forward → backward → clip_grad_norm_(1.0) → AdamW.step
```

训练文本过短时，实际窗口取 `min(配置窗口, 训练字符数−1)`，界面显示实际长度。默认 AdamW 学习率 0.003、权重衰减 0.01、batch size 16、seed 7；学习率范围 0.0001–0.01。

一次 HTTP 请求最多训练 25 步，单实验累计最多 2,000 步。「训练 100 步」会连续提交 4 批。每批完成后评估并保存，因此曲线只显示初始化与已保存步数的指标，通常为 0、25、50、75、100。

评估时两份文本分别按上下文长度分块，每块从空上下文开始，因果前向后统计下一字符交叉熵。每份文本的 N−1 个目标各计一次，以 token 数加权平均。验证不参与梯度更新。验证困惑度为 `exp(验证平均交叉熵)`，采用自然对数。没有额外独立测试集，不能据此宣称通用语言能力。

默认语料词表大小 31，默认模型为 27,999 个参数。本机验证的默认配置训练指标如下，其他版本或硬件可能出现数值差异：

| 已保存步数 | 训练交叉熵 | 验证交叉熵 |
| ---: | ---: | ---: |
| 0 | 3.4357 | 3.4456 |
| 25 | 1.6833 | 2.8102 |
| 50 | 0.8129 | 3.2110 |
| 75 | 0.4659 | 3.7719 |
| 100 | 0.3513 | 4.1353 |

训练拟合变好而验证变差，是小语料过拟合的直观例子。上述已保存点中的最佳验证权重来自步 25。最佳权重仅在每批评估时比较，未保存的中间步没有单独评估。

## 暂停、保存与继续训练

实验保存到 `runs/transformer/<run_id>/checkpoint.pt`，包括：

- 模型类型、版本、配置、语料、词表、创建时间与参数量。
- 当前 `state_dict` 与最低验证损失对应的 `best_state_dict`。
- AdamW 的参数组、动量矩、步数及窗口采样 Generator 状态。
- 当前步数、最佳步数和全部已保存评估记录。

每批训练先在独立模型上完成，再写入临时文件并原子替换检查点；写入失败时旧检查点保持有效。训练请求携带 `expected_step`，步数过时返回 409，避免同一旧步数重复提交。

「暂停后续训练」停止后续请求，不撤销已提交批次。离开页面、网络中断或客户端超时，也不保证中止服务端已经接收的当前批次。再次进入后先「刷新记录」，读取实际保存步数，再继续。服务正常重启后会恢复权重、优化器与随机采样状态；进程若在保存前被终止，本批未保存工作不计入进度。

页面列出最近 30 个实验；浏览器仅保存当前实验 ID，权重保存在服务端，清除浏览器学习记录不会删除训练检查点。配置编辑只影响新建实验；继续训练始终使用该实验的语料和配置快照。使用项目默认的单进程 API 服务，避免多个服务进程同时写入同一目录。

## 文本生成与真实注意力

选择「当前权重」或「最低验证损失权重」，页面与响应明确显示实际加载步数。若最佳权重仍来自步 0，它尚未训练；使用步 0 生成不能完成训练后生成的课程任务。

模型进入 `eval()`，并在 `torch.inference_mode()` 中前向。每次只保留最近 C 个 token，位置从 0 重新编号；超长提示词只截断模型上下文，显示结果仍保留完整提示词。

采样先把最后一个位置的 logits 除以温度，再按可选 Top-k 保留分数最高的候选，Softmax 后用局部 Generator 采样。Top-k=0 表示全部候选。每次生成的新 token 接回输入，重新计算上下文。固定实验、权重、提示词、温度、Top-k、长度与种子，可以复现输出，采样不会修改检查点。

首个生成位置还返回最后一层的真实注意力矩阵，形状为 `[H,T,T]`。可以切换头和查询位置：每行总和约为 1，未来位置严格为 0。该矩阵描述对输入位置的信息汇总；页面候选概率则是对词表项的采样分布，二者不同。注意力不构成因果解释。

普通生成 token 对应一个字符，UNK 显示为 `<UNK>`。小型字符模型可能重复、不通顺或拼接训练片段；更长上下文与更低训练损失不保证生成更好。本教程不包含大规模预训练、指令微调或聊天模板。

## 在 Python 中复现

在项目根目录启动 Python，使用同一份后端核心计算：

```python
from backend.transformer import RunConfig, TrainingRequest, GenerationRequest, service

run = service.create(RunConfig().model_dump())
for _ in range(4):
    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=25))
print(run["history"][-1])

result = service.generate(run["id"], GenerationRequest(
    prompt="小猫", checkpoint="best", temperature=0.8, top_k=5, seed=7,
))
print(result["prompt"] + result["continuation"])
print(result["attention_layer"], result["context_tokens"])
```

下载的完整检查点可以离线重建模型：

```python
import torch
from backend.transformer import build_transformer

saved = torch.load("transformer-检查点.pt", map_location="cpu", weights_only=True)
model = build_transformer(len(saved["vocab"]), saved["config"])
model.load_state_dict(saved["best_state_dict"])  # 或 state_dict：当前权重
model.eval()
```

## HTTP 接口与验证

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/transformer/status` | 运行依赖状态、默认语料和步数上限 |
| POST | `/api/transformer/runs` | 校验配置并创建初始检查点 |
| GET | `/api/transformer/runs` | 最近 30 个实验与评估记录 |
| GET | `/api/transformer/runs/{id}` | 读取持久化实验 |
| POST | `/api/transformer/runs/{id}/train` | 真实训练，需 `expected_step`、`steps`（1–25） |
| POST | `/api/transformer/runs/{id}/generate` | 使用指定权重生成文本与注意力 |
| GET | `/api/transformer/runs/{id}/checkpoint` | 下载完整检查点 |

API 文档位于 `/api/docs`。没有上传代码或检查点导入接口。

```powershell
python -m unittest backend.tests.test_transformer -v
python -m unittest discover -s backend/tests -v
npm run check
```

测试覆盖因果前缀不变性、未来注意力为零、真实训练损失下降、重启后 AdamW / RNG 精确续训、验证不影响更新、最佳权重真实加载、上下文截断、Top-k 与种子复现、全局 RNG 隔离、原子保存失败恢复及 HTTP 参数边界。

官方延伸阅读：[PyTorch Transformer 组成模块](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)、[因果多头注意力示例](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial)、[交叉熵定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
