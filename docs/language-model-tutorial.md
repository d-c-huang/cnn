# 第五阶段：从零训练字符语言模型

这是面向初学者的真实 Bigram 训练教程，使用现有 Python / NumPy 环境即可运行，不需要 GPU、预训练权重或语料下载。课程位于侧栏「05 语言」，也可在启动 `npm run dev` 后打开：

1. [文本与下一字符](http://127.0.0.1:5173/#language-data)：编辑语料，查看词表、token ID、相邻字符标签和实际数据划分。
2. [训练语言模型](http://127.0.0.1:5173/#language-train)：设置学习率，单步或连续训练，比较训练与验证交叉熵、验证困惑度。
3. [文本生成与温度](http://127.0.0.1:5173/#language-generate)：使用当前权重生成文本，固定种子比较不同温度的候选分布。

每节都有操作任务、两道自测、错误选项解析、手算例子和术语。完成任务并答对两题后，可以记录学习进度。原有课程记录继续有效。

## 1. 准备数据

默认语料是项目内编写的四季与小猫短文。可替换为 40–2,000 个 Unicode 字符的文本，训练部分需有 2–95 种不同字符。空格、标点和换行也算字符。

先按原文顺序取前 `floor(0.8 × 字符数)` 个字符作为训练文本，剩余部分作为验证文本。词表只在训练文本上建立，索引 0 是 `<UNK>`，验证中的新字符映射到它。再分别把两部分变成相邻字符对，不生成跨越边界的样本。

例如「小猫看花」产生「小→猫、猫→看、看→花」三对样本。输入和目标相差一个位置，标签直接来自文本本身。这里采用字符分词；常见子词分词器中的一个 token 可能是部分单词或多个字符。

修改文本后点击「准备语料并重置模型」才会生效，并同时重建词表和清空权重。验证集不参与训练；本教学没有第三份独立测试集，验证结果不作为通用语言能力结论。

## 2. 模型与真实训练

词表大小为 V，参数只有一个 `V × V` 矩阵 W。给定当前字符的 ID i，取出第 i 行作为下一字符的 logits：

```text
p(j | i) = softmax(W[i])[j]
L = mean(-ln p(真实下一字符 | 当前字符))
W = W - 学习率 × dL/dW
```

所有权重从零开始，所以初始分布均匀，损失为 `ln(V)`。每步使用全部训练样本，后端以相邻字符计数合并重复样本，计算精确的全量 SGD 梯度：

```text
C[i,j] = 训练中 i→j 的出现次数
n[i] = sum_j C[i,j]
N = sum_i n[i]
dL/dW[i,j] = (n[i] × p(j|i) - C[i,j]) / N
```

一次请求最多更新 200 步，界面累计最多 5,000 步。曲线每一点都是更新之后重新计算的真实损失；验证数据只用于前向计算。验证困惑度为 `exp(验证平均交叉熵)`，使用自然对数。

默认语料有 31 个词表项（含 UNK）、122 个训练字符、31 个验证字符，共 961 个参数。默认学习率 5，执行 200 步时，本次 NumPy 验证的训练损失从约 3.4340 降至 0.8655，验证损失约 3.3525。训练损失与验证损失的差距正好说明：拟合训练文本，不等于能处理各种新文本。

可在项目根目录用同一份核心计算代码复现：

```python
from backend.language import CorpusRequest, TrainRequest, GenerateRequest, prepare, train, generate

data = prepare(CorpusRequest())
result = train(TrainRequest(weights=data["weights"], steps=200, learning_rate=5))
print(result["history"][-1])
output = generate(GenerateRequest(
    weights=result["weights"], prompt="小猫", temperature=0.8, seed=7,
))
print("小猫" + output["continuation"])
```

## 3. 自回归生成

提示词只能包含当前词表中的字符。每次读取最后一个 token，按 `softmax(W[i] / temperature)` 采样一个 token，再把它接回输入。默认生成 60 个 token；普通 token 是一个字符，未知 token 显示为 `<UNK>`。

- 较低温度放大分差，输出更集中，也可能重复循环。
- 较高温度使分布更分散，随机性更强，也可能更不通顺。
- 温度不更新权重。固定权重、提示词、温度、长度与种子可复现输出。
- Bigram 只依赖最后一个字符，因此「小猫」和「看猫」在相同设置下生成相同后缀。

页面显示第一个生成位置的前 8 个候选概率，其他词表项未列出，因此这 8 项不一定相加为 1。

三个实验共享当前标签页内存中的语料、权重和历史；刷新页面会重置实验。下载 JSON 包含模型类型、词表、语料、权重、步数和指标，当前不支持导入。学习进度独立保存在浏览器本地存储。

## 4. 从 Bigram 到 Transformer

Bigram 是最小的一阶语言模型，没有嵌入层、注意力或长上下文。Transformer 通过 token 嵌入、位置信息、因果注意力与前馈网络利用多个历史 token。训练时须屏蔽未来位置，避免直接读取下一 token 答案。

下一步可把单行查表替换为嵌入与多层因果网络，保留「准备语料 → 构造输入与右移标签 → logits → 交叉熵 → 反向传播 → 更新 → 验证 → 采样」的流程。第五阶段保留 Bigram 的最小计算流程；[第六阶段 Transformer 教程](transformer-language-model-tutorial.md) 已实现真实 Transformer 训练、检查点和文本生成。大模型预训练、指令微调或聊天助手不在这两阶段的范围内。

官方延伸阅读：[PyTorch 的词嵌入与 N-gram 语言模型教程](https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)、[CrossEntropyLoss 的 logits 与标签约定](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

## 接口与验证

三个无状态接口为 `POST /api/language/prepare`、`POST /api/language/train`、`POST /api/language/generate`。后两个请求需携带与语料词表尺寸一致的 `weights`；自定义语料时三个请求须使用相同 `corpus`。完整参数可查看 `/api/docs`。端点不会执行上传代码，也不会在服务端保存本实验权重。

```powershell
python -m unittest backend.tests.test_language -v
npm run test:frontend
npm run build
npm run test:bundle
npm run test:design
```

后端测试将实际 SGD 更新与 PyTorch 双精度交叉熵逐步对比，并验证数据划分、未知字符、验证数据不改变梯度、连续训练一致性、采样复现性和接口参数边界。
