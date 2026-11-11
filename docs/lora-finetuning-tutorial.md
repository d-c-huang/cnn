# 第七阶段：预训练语言模型的真实 LoRA 微调

在公开的 **EleutherAI/pythia-14m** 预训练权重上，用 Transformers 与 PEFT 真实微调 LoRA。冻结基座，只更新低秩矩阵；支持 CPU、编辑语料、训练/验证曲线、暂停续训、生成对比和标准适配器导出。

## 开始学习

安装 `requirements.txt` 后运行 `npm run dev`，侧栏选择「07 LoRA」：

1. [预训练模型与 LoRA](http://127.0.0.1:5173/#lora-basics)：准备基座，点选 rank，核对 A、B 形状与参数量。
2. [真实 LoRA 微调](http://127.0.0.1:5173/#lora-train)：创建实验，真实训练并检查冻结证据、训练与验证指标。
3. [微调前后生成对比](http://127.0.0.1:5173/#lora-generate)：对齐生成设置，比较原模型与适配器输出，下载适配器及续训检查点。

三节都接入操作任务、双题自测、错误解析、手算示例、术语与学习进度。

## 准备公开预训练模型

使用 [EleutherAI 官方 Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m)，是英文基础因果语言模型，模型页标注 Apache-2.0。默认微调文本和提示词使用英文，讲解使用中文；它不是聊天指令模型。

```text
model_id: EleutherAI/pythia-14m
revision: cf967c0a9a04383db6f7b1108d86b2962634b4ac
model.safetensors SHA-256:
116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1
```

点击「下载并准备预训练模型」。后台从 Hugging Face 下载固定版本，页面显示真实状态；离开页面不会取消下载。原始权重约 28.1 MB，分词器约 2.1 MB。CPU FP32 的基座参数本身约占 56.3 MB，训练还需额外内存。

缓存保存在 `data/pretrained/pythia-14m/<revision>/`。下载结束校验固定 SHA-256，首次载入再次校验文件字节。仅使用 Safetensors、`trust_remote_code=False` 与 `local_files_only=True` 加载；缓存完整后微调和生成无需联网。缺少权重时明确返回错误，不使用随机模型替代。

新增依赖范围见 `requirements.txt`。本机验证组合为 Transformers 4.57.1、PEFT 0.17.1、Safetensors 0.6.2、Hugging Face Hub 0.35.3。现有 Docker 的 `/app/data` 卷可以持久化预训练缓存，模型不会写入只读的发布权重目录。

## 看懂低秩更新

基座有 6 层、128 维隐藏表示、每层 4 个注意力头，实际参数数为 **14,067,712**。本教程只在六层融合的 `query_key_value` 线性层增加 LoRA：

```text
y = W x + (alpha / rank) B A x
W: [384,128]   冻结
A: [rank,128]  可训练
B: [384,rank]  可训练
```

默认 rank=4、alpha=8，共 `6 × 4 × (128+384) = 12,288` 个可训练参数，占含适配器总参数约 0.0873%。rank 可选 2 / 4 / 8 / 16，alpha 可选 4 / 8 / 16 / 32。

全部原始参数 `requires_grad=False`，优化器仅接收 LoRA A、B。词嵌入、输出层与偏置保持冻结；LoRA dropout=0，不使用量化或 QLoRA。A 随机初始化、B 为零，所以初始更新为零，步 0 的 logits 与原始预训练模型一致。初始化与采样使用局部 Generator，不重设进程全局随机状态。

减少可训练参数不代表省略整个基座：完整权重与前向计算仍然需要保留。

## 数据与真实微调

默认语料是项目编写的 Zhiwei 实验室英文介绍。可改为 100–6,000 个字符。先按原文顺序取前 80% 训练、后 20% 验证，再分别用原始子词 tokenizer 编码，不添加特殊 token、不重建词表。分词后训练至少 4 个 token、验证至少 2 个 token。

这是领域文本的因果语言建模，不使用聊天模板或只对助手回答计损失的 SFT 掩码。

默认上下文 32、batch size 2、AdamW 学习率 0.003、权重衰减 0、seed 7。上下文可选 16 / 32 / 64，批大小 1 / 2 / 4，学习率为 0.0001–0.01。短训练文本实际窗口取 `min(配置窗口, 训练 token 数−1)`。

```text
inputs  = ids[start : start+C]
targets = ids[start+1 : start+C+1]
logits  = model(inputs, use_cache=False).logits
loss    = cross_entropy(logits, targets)
zero_grad → backward → clip_grad_norm_(LoRA参数, 1.0) → AdamW.step
```

每次请求最多 5 步，每个实验最多 500 步。「微调 20 步」连续提交 4 批。每批完成后分别对完整训练/验证文本分块评估，下一 token 目标各计一次，块间重置上下文；记录 token 加权交叉熵与 `exp(验证损失)` 困惑度。验证不参与更新，没有额外独立测试集。

默认配置的本机真实记录如下，数值与耗时受版本和硬件影响：

| 步数 | 训练交叉熵 | 验证交叉熵 |
| ---: | ---: | ---: |
| 0 | 5.6699 | 5.3643 |
| 5 | 4.9180 | 5.1372 |
| 10 | 4.6612 | 4.8492 |
| 15 | 4.2616 | 4.9536 |
| 20 | 3.8376 | 4.9353 |

这些保存点中最佳验证适配器来自步 10。小语料可能过拟合，不能用训练损失或单段输出证明通用能力提升。

## 冻结、保存与继续训练

每批训练前后逐字节计算所有非 LoRA 参数的 SHA-256，与实验创建时比较。只有基座保持一致，才允许保存。界面「冻结校验通过」来自这项实际检查。

`runs/lora/<id>/checkpoint.pt` 包含基座 ID / revision / 权重 SHA、语料和配置、当前/最佳适配器、AdamW 状态、窗口采样 RNG 与评估历史，不重复保存基座。每批先在独立模型副本上更新，再用临时文件原子替换检查点，失败时旧记录保持有效。

训练请求携带 `expected_step`，陈旧步数返回 409。暂停或离开页面只阻止后续提交，已接收的一批仍可能完成并保存；网络中断后先刷新记录再继续。服务重启后恢复基座、适配器、优化器与 RNG；未保存的一批不计入进度。页面列出最近 30 个实验，编辑配置只影响新建实验。使用默认单进程 API，避免多个进程同时写同一目录。

## 生成对比与导出

左侧关闭 LoRA，右侧启用当前/最佳验证适配器。两列使用相同提示词、温度、Top-k、长度与独立的同种子 RNG。页面显示实际适配器步数；若仍为步 0，则增量为零，输出应一致。

每次前向读取最近 C 个 token，位置重新从 0 开始，不使用 KV cache。这里的 C 是实验的 16 / 32 / 64，而非基座原始最大窗口。可选 Top-k=10 / 50 或 0（全部候选），最多生成 64 个新 token，遇 EOS 提前结束。显示时保留提示词，新增文本跳过特殊 token。

生成在 eval 与 inference_mode 中执行，不修改检查点。同一模型及设置可复现输出，不同分布在相同种子下也可能采样不同文本。模型很小，输出可能重复、不通顺或偏离语料。

PEFT 适配器 ZIP 含：

```text
adapter_model.safetensors  仅 A、B
adapter_config.json        标准 PEFT 配置与基座引用
zhiwei_manifest.json       revision、权重 SHA、实验与导出步数
README.txt                 加载说明
```

ZIP 不含完整基座或 AdamW。解压后可按标准 PEFT 方式加载，已用实际导出文件验证：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "EleutherAI/pythia-14m"
revision = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
base = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, use_safetensors=True,
    trust_remote_code=False, dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
model = PeftModel.from_pretrained(base, "解压后的适配器目录")
model.eval()
```

离线使用时把模型地址改为项目缓存目录，并添加 `local_files_only=True`。精确继续本实验需要「续训检查点」，仅适配器 ZIP 不足以恢复优化器轨迹。

## 用项目代码复现

```python
from backend.lora_assets import assets
from backend.lora import RunConfig, TrainingRequest, GenerationRequest, service

assets.ensure()
run = service.create(RunConfig().model_dump())
for _ in range(4):
    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=5))
output = service.generate(run["id"], GenerationRequest(prompt="Zhiwei is", checkpoint="best"))
print("BASE:", output["baseline"]["text"])
print("LORA:", output["adapted"]["text"])
```

接口前缀为 `/api/lora`：`GET /status`、`POST /prepare`、`GET/POST /runs`、`GET /runs/{id}`、`POST /runs/{id}/train`、`POST /runs/{id}/generate`、`GET /runs/{id}/adapter?checkpoint=latest|best`、`GET /runs/{id}/checkpoint`。完整参数见 `/api/docs`；不接受任意模型 ID、上传代码或检查点导入。

```powershell
python -m unittest backend.tests.test_lora -v
python -m unittest discover -s backend/tests -v
npm run check
```

单元测试使用本地小结构验证初始等价、真实梯度、冻结、精确续训、验证隔离、RNG、原子保存和标准 PEFT 重载。缓存完整时额外运行真实固定版本 Pythia-14M 集成测试；测试本身不下载模型，缺缓存时明确跳过。

官方资料：[Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m)、[LoRA 论文](https://arxiv.org/abs/2106.09685)、[PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora)、[PEFT 检查点格式](https://huggingface.co/docs/peft/developer_guides/checkpoint)。
