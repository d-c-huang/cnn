# Docker 部署与第四期模型包

仓库中的 `models/` 包含从第三期真实训练记录导出的 MLP、CNN 验证集最佳权重，以及每个数字一个真实 MNIST 官方测试示例。镜像会复制这些文件；推理无需重新训练或下载 MNIST。

## 一条命令启动

在项目根目录、Docker Desktop Linux 引擎或 Linux Docker Engine 已运行时执行：

```powershell
docker compose up --build -d
```

打开 <http://127.0.0.1:8080>。默认只绑定本机回环地址。前端由 Node 构建为静态文件，再由同一个 FastAPI 服务提供网页和 `/api`；容器不运行 Vite 开发服务器。

首次构建需要从 npm、Python 包索引与 PyTorch CPU 索引下载依赖。构建完成后的手写识别与已打包示例可离线使用。第三期从头训练仍需准备完整 MNIST。

可复制 `.env.example` 为 `.env`，把 `PORT=8080` 改为其他未占用端口，再运行同一命令。

## 状态、检查与停止

```powershell
docker compose ps
docker compose logs -f app
python scripts/smoke_test.py --base-url http://127.0.0.1:8080
docker compose down
```

HTTP smoke test 检查生产首页、构建资源、未知 API 的 404、两个模型以及真实测试示例的推理结果和概率向量。它不是容器运行结果的替代；只有在目标服务实际启动后通过该命令，才验证了该入口。

Compose 的健康检查使用轻量 `/api/health`。第一次推理才加载 CPU 权重，页面会显示等待状态。服务使用一个 Uvicorn worker，避免第三期单后台训练任务在多个 worker 间产生不一致。

## 持久化与运行身份

容器以非 root 的 UID/GID `10001:10001` 运行。`mnist-data`、`mnist-runs` 两个命名卷分别挂载 `/app/data` 和 `/app/runs`，保留后续下载的数据、训练记录与最佳模型。普通 `docker compose down` 不删除这些卷。

`models/` 中的权重是镜像内的只读发布文件。容器里新训练的模型保存在命名卷中；已有完成记录可被推理模型目录发现。第三期的宿主机 `runs/` 不会自动复制到镜像或覆盖容器数据卷，当前两个模型通过导出包随镜像发布。

## 更新发布的最佳模型

完成第三期的新训练后，在宿主项目中执行：

```powershell
python scripts/export_models.py
docker compose up --build -d
```

默认针对每种架构选择创建时间最新、已完成且具有 `best.pt` 的记录；该权重必须与该次训练的最低验证损失轮次一致。脚本不会按测试准确率挑选模型，也不会修改原始训练记录。可以明确指定来源：

```powershell
python scripts/export_models.py --mlp-run 9fbe732245894df3be3a45ee980c63ff --cnn-run 550f1b22c64045f89696db036db3a2dc
```

脚本使用 `weights_only=True` 加载，严格核对模型结构、验证轮次与标准化参数，然后生成 `models/manifest.json`、两个 `.pt` 文件和 `models/examples.json`。manifest 记录 SHA-256、原始实验 ID、验证轮次与实际测试结果。示例来自缓存中的官方 `t10k` IDX 文件；没有缓存时导出空示例列表，并保持手写识别可用。

若没有完成的 MLP/CNN 权重，导出脚本会失败并要求先训练。推理服务同样会明确报告缺少模型，不会用随机权重替代。

## 本地生产入口验证

不使用 Docker 时，也可以验证与镜像相同的 FastAPI 静态资源入口：

```powershell
npm run build
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8080 --workers 1
```

另一个终端运行 `python scripts/smoke_test.py --base-url http://127.0.0.1:8080`。开发模式仍使用 `npm run dev` 和 <http://127.0.0.1:5173>。

当前配置提供本机 HTTP 入口。对外域名、TLS、账号和公网访问控制应由部署环境的反向代理配置承担；本任务没有自动发布云服务。

## LoRA 预训练基座缓存

第七阶段首次准备会从 Hugging Face 下载固定版本 Pythia-14M。缓存写入 `/app/data/pretrained/`，微调适配器与续训状态写入 `/app/runs/lora/`，分别复用已有 data/runs 命名卷。发布目录 `/app/models` 仍保持只读；首次下载需要网络，缓存完整后 LoRA 可离线运行。完整说明见 [LoRA 微调教程](lora-finetuning-tutorial.md)。

## SFT 指令微调与对话

第八阶段复用上述预训练缓存，SFT 数据快照、适配器、回答指标与续训状态写入 `/app/runs/sft/`，使用已有 runs 卷。自定义 JSON/JSONL 数据经本机 API 校验和分词，问答草稿保存在当前浏览器。训练与对话使用同一版本模板，导出时附带模板与配置。参见 [SFT 教程](sft-instruction-tuning-tutorial.md)。

## DPO 偏好优化与生成对照

第九阶段沿用相同预训练缓存，偏好数据、固定参考适配器与分数、策略适配器和续训状态保存在 `/app/runs/dpo/`，复用 runs 卷。可从现有 SFT 快照初始化；创建完成后参考独立持久化，来源 SFT 的后续更新不会改变该参考。生成对照在本机 API 内运行，无需外部推理服务。仍使用默认单进程 API。参见 [DPO 教程](dpo-preference-optimization-tutorial.md)。

## GRPO 在线强化学习微调

第十阶段复用现有模型缓存和 runs 卷，问题、奖励规则、固定参考、策略适配器、AdamW、采样 RNG 与在线采样组保存到 `/app/runs/grpo/`。可从已训练 SFT / DPO 快照开始，创建后参考独立持久化。GRPO 在线采样与训练的 CPU 开销高于离线 DPO，页面逐轮提交；中断后先刷新再继续。保持默认单进程 API，无需外部奖励或推理服务。参见 [GRPO 教程](grpo-reinforcement-finetuning-tutorial.md)。
