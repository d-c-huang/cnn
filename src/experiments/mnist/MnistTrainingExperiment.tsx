import { useEffect, useRef, useState } from "react";
import { ArrowRight, Download, Play, Square } from "lucide-react";
import { ErrorMessage, ExperimentPanel, Result } from "../../components/ui";
import { format } from "../../lib/api";
import {
  activeStatuses,
  loadMnistConfig,
  mnistRequest,
  selectRunId,
  selectedRunId,
  statusLabels,
  storeMnistConfig,
} from "../../lib/mnist";
import type { MnistStatus, Run, RunConfig } from "../../lib/mnist";
import { downloadText, useMnistResource } from "../../lib/useMnist";
import TrainingCurves from "./TrainingCurves";
import "./mnist-training.css";

export default function MnistTrainingExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [config, setConfig] = useState<RunConfig>(loadMnistConfig);
  const [chosenId, setChosenId] = useState<string | null>(selectedRunId);
  const [freshRun, setFreshRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState(
    "配置新实验后，启动真实训练。后台任务会在离开页面后继续运行。",
  );
  const status = useMnistResource<MnistStatus>("/status");
  const records = useMnistResource<{ runs: Run[] }>("/runs");
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const runs = records.data?.runs || [];
  const run =
    runs.find((item) => item.id === chosenId) ||
    (freshRun?.id === chosenId ? freshRun : null) ||
    runs.find((item) => activeStatuses.has(item.status)) ||
    runs.find((item) => item.status === "completed") ||
    runs[0] ||
    null;
  const active = runs.find((item) => activeStatuses.has(item.status));
  const currentActive = !!(
    active ||
    status.data?.active_run_id ||
    (freshRun &&
      activeStatuses.has(freshRun.status) &&
      !runs.some((item) => item.id === freshRun.id))
  );
  const latest = run?.history.at(-1);
  const progress = run
    ? Math.min(
        1,
        Math.max(
          0,
          (Math.max(run.epoch, 1) -
            1 +
            (run.train_samples
              ? Math.min(1, run.processed_samples / run.train_samples)
              : 0)) /
            run.config.epochs,
        ),
      )
    : 0;

  useEffect(() => {
    storeMnistConfig(config);
  }, [config]);
  useEffect(
    () => () => {
      version.current++;
      request.current?.abort();
    },
    [],
  );

  function change(next: Partial<RunConfig>) {
    setConfig((previous) => ({ ...previous, ...next }));
    onInteract();
  }
  function choose(item: Run) {
    setChosenId(item.id);
    selectRunId(item.id);
    setFreshRun(null);
    onInteract();
  }
  async function operation(action: "start" | "cancel" | "prepare") {
    if (request.current) return;
    const token = ++version.current;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    try {
      if (action === "start") {
        const result = await mnistRequest<Run>("/runs", {
          body: config,
          signal: controller.signal,
        });
        if (token !== version.current) return;
        setFreshRun(result);
        setChosenId(result.id);
        selectRunId(result.id);
        setNotice(
          "训练任务已提交。下方展示所选记录的真实指标，曲线会在每个完整 epoch 后更新。",
        );
      } else if (action === "cancel" && run) {
        const result = await mnistRequest<Run>(`/runs/${run.id}/cancel`, {
          body: {},
          signal: controller.signal,
        });
        if (token !== version.current) return;
        setFreshRun(result);
        setNotice("已发送取消请求，将在批次边界安全停止并保留已完成的记录。");
      } else if (action === "prepare") {
        await mnistRequest<MnistStatus>("/prepare", {
          body: {},
          signal: controller.signal,
        });
        if (token !== version.current) return;
        setNotice(
          "正在后台准备 MNIST。首次使用需要下载官方数据，缓存完成后可重复使用。",
        );
      }
      if (token === version.current) {
        onInteract();
        void records.refresh();
        void status.refresh();
      }
    } catch (cause) {
      if (token === version.current && !controller.signal.aborted)
        setError(cause instanceof Error ? cause.message : "请求失败，请重试。");
    } finally {
      if (token === version.current) {
        request.current = null;
        setBusy(false);
      }
    }
  }

  return (
    <div className="mt-experiment">
      <div className="mt-configuration-grid">
        <ExperimentPanel
          title="选择模型，开始一次完整训练"
          eyebrow="NEW EXPERIMENT · PYTORCH"
        >
          <div
            className="mt-model-options"
            role="group"
            aria-label="选择 MNIST 模型"
          >
            {(["mlp", "cnn"] as const).map((model) => (
              <button
                key={model}
                aria-pressed={config.model === model}
                onClick={() => change({ model })}
              >
                <strong>{model.toUpperCase()}</strong>
                <span>{model === "mlp" ? "全连接网络" : "卷积神经网络"}</span>
                <small>
                  {model === "mlp"
                    ? "784 → 128 → 64 → 10"
                    : "Conv 8 → Pool → Conv 16 → Pool → 64 → 10"}
                </small>
              </button>
            ))}
          </div>
          <div className="mt-fields">
            <label>
              训练轮数 Epochs
              <select
                value={config.epochs}
                onChange={(e) => change({ epochs: Number(e.target.value) })}
              >
                {[1, 2, 3, 5, 10, 20].map((n) => (
                  <option key={n} value={n}>
                    {n} 轮
                  </option>
                ))}
              </select>
            </label>
            <label>
              批次大小 Batch size
              <select
                value={config.batch_size}
                onChange={(e) => change({ batch_size: Number(e.target.value) })}
              >
                {[32, 64, 128, 256].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label>
              学习率
              <select
                value={config.learning_rate}
                onChange={(e) =>
                  change({ learning_rate: Number(e.target.value) })
                }
              >
                {[0.0001, 0.0003, 0.001, 0.003, 0.01].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label>
              训练使用的数据
              <select
                value={config.train_limit ?? "full"}
                onChange={(e) =>
                  change({
                    train_limit:
                      e.target.value === "full" ? null : Number(e.target.value),
                  })
                }
              >
                <option value="full">全部训练样本</option>
                <option value={6000}>快速实验 · 6,000 张</option>
                <option value={12000}>快速实验 · 12,000 张</option>
              </select>
            </label>
            <label>
              验证集比例
              <select
                value={config.validation_ratio}
                onChange={(e) =>
                  change({ validation_ratio: Number(e.target.value) })
                }
              >
                <option value={0.1}>官方训练集的 10%</option>
                <option value={0.2}>官方训练集的 20%</option>
              </select>
            </label>
            <label>
              随机种子
              <input
                type="number"
                min={0}
                max={9999}
                step={1}
                value={config.seed}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (Number.isInteger(value) && value >= 0 && value <= 9999)
                    change({ seed: value });
                }}
              />
            </label>
          </div>
          <div className="mt-start-row">
            <button
              className="primary-button"
              disabled={
                busy ||
                currentActive ||
                status.data?.dataset.status !== "ready" ||
                !status.data?.runtime.available
              }
              onClick={() => void operation("start")}
            >
              <Play size={15} />
              {busy ? "提交中…" : "开始真实训练"}
            </button>
            {status.data && status.data.dataset.status !== "ready" && (
              <button
                className="secondary-button"
                disabled={
                  busy ||
                  status.data.dataset.status === "preparing" ||
                  !status.data.runtime.available
                }
                onClick={() => void operation("prepare")}
              >
                {status.data.dataset.status === "preparing"
                  ? "正在准备数据…"
                  : "准备 MNIST 数据"}
              </button>
            )}
            {active && (
              <button className="text-button" onClick={() => choose(active)}>
                查看正在运行的任务 <ArrowRight size={14} />
              </button>
            )}
          </div>
          <p className="panel-note">
            优化器 Adam · 多分类交叉熵 · CPU。相同种子使用相同的分层划分；官方
            10,000 张测试图像始终独立。
          </p>
          <p className="mt-operation-notice" role="status">
            {notice}
          </p>
          <ErrorMessage
            error={
              error ||
              status.error ||
              records.error ||
              status.data?.dataset.error ||
              null
            }
          />
        </ExperimentPanel>
        <ExperimentPanel
          title="数据怎样参与训练"
          eyebrow="TRAIN / VALIDATION / TEST"
        >
          <div className="mt-split-numbers">
            <Result
              label="实际用于梯度更新"
              value={(
                config.train_limit ||
                Math.round(60000 * (1 - config.validation_ratio))
              ).toLocaleString()}
              hint="训练集"
            />
            <Result
              label="选择最佳 checkpoint"
              value={Math.round(
                60000 * config.validation_ratio,
              ).toLocaleString()}
              hint="验证集"
            />
            <Result label="训练结束后评估" value="10,000" hint="官方测试集" />
          </div>
          <div className="mt-process-note">
            <p>1. 训练集更新模型参数。</p>
            <p>2. 每轮在验证集上计算指标。</p>
            <p>3. 保存验证损失最低的模型。</p>
            <p>4. 用该模型评估独立测试集。</p>
          </div>
          <a className="text-button" href="#mnist-data">
            检查数据划分与样本 <ArrowRight size={14} />
          </a>
          <p className="panel-note">
            这里是新实验配置；修改配置不会改变下方已保存记录的模型与曲线。
          </p>
        </ExperimentPanel>
      </div>

      <ExperimentPanel title="实验记录" eyebrow="PERSISTED RUNS">
        {records.pending ? (
          <p className="panel-note">正在读取记录…</p>
        ) : runs.length ? (
          <div className="mt-run-list" role="group" aria-label="选择训练记录">
            {runs.map((item) => (
              <button
                key={item.id}
                aria-pressed={run?.id === item.id}
                onClick={() => choose(item)}
              >
                <span>
                  <strong>{item.config.model.toUpperCase()}</strong>
                  <em>{statusLabels[item.status]}</em>
                </span>
                <span>
                  {item.train_samples.toLocaleString()} 张 ·{" "}
                  {item.config.epochs} 轮 · seed {item.config.seed}
                </span>
                <small>
                  {new Date(item.created_at).toLocaleString("zh-CN", {
                    timeZone: "Asia/Shanghai",
                    hour12: false,
                  })}
                </small>
                <b>
                  {item.evaluation
                    ? `测试 ${(item.evaluation.accuracy * 100).toFixed(2)}%`
                    : item.best_validation_loss === null
                      ? "尚无完整 epoch"
                      : `最低验证损失 ${format(item.best_validation_loss, 4)}`}
                </b>
              </button>
            ))}
          </div>
        ) : (
          <p className="mt-empty">
            还没有训练记录。准备数据后，选择 MLP 或 CNN 开始训练。
          </p>
        )}
      </ExperimentPanel>

      {run && (
        <>
          <ExperimentPanel
            title={`${run.config.model.toUpperCase()} · 所选实验的训练过程`}
            eyebrow={`RUN ${run.id.slice(0, 8).toUpperCase()}`}
            action={
              <span className={`mt-status ${run.status}`}>
                {statusLabels[run.status]}
              </span>
            }
          >
            <div className="mt-run-heading">
              <div>
                <strong>
                  Epoch {run.epoch} / {run.config.epochs}
                </strong>
                <span>
                  当前批次 {run.batch} / {run.batches_per_epoch} ·{" "}
                  {run.device.toUpperCase()} ·{" "}
                  {run.parameter_count.toLocaleString()} 个参数
                </span>
              </div>
              {activeStatuses.has(run.status) && (
                <button
                  className="secondary-button"
                  disabled={busy || run.status === "cancelling"}
                  onClick={() => void operation("cancel")}
                >
                  <Square size={13} />
                  取消任务
                </button>
              )}
            </div>
            <div
              className="mt-progress"
              role="progressbar"
              aria-label="所选任务训练轮数进度"
              aria-valuenow={Math.round(progress * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div style={{ width: `${progress * 100}%` }} />
            </div>
            <p className="mt-operation-notice" role="status">
              {run.message}
            </p>
            <div className="result-grid three-results">
              <Result
                label="最新完整轮 · 训练准确率"
                value={
                  latest ? `${(latest.train_accuracy * 100).toFixed(2)}%` : "—"
                }
              />
              <Result
                label="最新完整轮 · 验证准确率"
                value={
                  latest
                    ? `${(latest.validation_accuracy * 100).toFixed(2)}%`
                    : "—"
                }
              />
              <Result
                label="验证损失最佳轮"
                value={
                  run.best_epoch === null ? "—" : `Epoch ${run.best_epoch}`
                }
                hint={
                  run.best_validation_loss === null
                    ? "等待第一轮完成"
                    : `验证损失 ${format(run.best_validation_loss, 5)}`
                }
              />
            </div>
            <TrainingCurves history={run.history} />
            <p className="panel-note">
              训练指标是该轮真实训练批次的样本加权平均；验证指标是在该轮结束后的固定模型上计算。只有训练集执行梯度更新。
            </p>
            <div className="mt-report-actions">
              <button
                className="text-button"
                disabled={!run.history.length}
                onClick={() =>
                  downloadText(
                    `mnist-${run.config.model}-${run.id.slice(0, 8)}-history.csv`,
                    "\uFEFFepoch,train_loss,train_accuracy,validation_loss,validation_accuracy,seconds\n" +
                      run.history
                        .map(
                          (row) =>
                            `${row.epoch},${row.train_loss},${row.train_accuracy},${row.validation_loss},${row.validation_accuracy},${row.seconds}`,
                        )
                        .join("\n"),
                    "text/csv;charset=utf-8",
                  )
                }
              >
                <Download size={14} />
                导出训练曲线 CSV
              </button>
              {run.checkpoint_available && (
                <a
                  className="text-button"
                  href={`/api/mnist/runs/${run.id}/checkpoint`}
                  download
                >
                  <Download size={14} />
                  下载最佳模型权重
                </a>
              )}
              {run.evaluation && (
                <a
                  className="secondary-button"
                  href="#mnist-evaluation"
                  onClick={() => {
                    selectRunId(run.id);
                    onInteract();
                  }}
                >
                  查看混淆矩阵与错误样本 <ArrowRight size={14} />
                </a>
              )}
            </div>
            <ErrorMessage error={run.error} />
          </ExperimentPanel>
          <ExperimentPanel
            title="逐轮指标，保留真实记录"
            eyebrow="EPOCH METRICS"
          >
            <div className="table-scroll">
              <table className="sample-table mt-history-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>训练损失</th>
                    <th>验证损失</th>
                    <th>训练准确率</th>
                    <th>验证准确率</th>
                    <th>用时</th>
                  </tr>
                </thead>
                <tbody>
                  {run.history.map((row) => (
                    <tr key={row.epoch}>
                      <td>
                        {row.epoch}
                        {row.epoch === run.best_epoch ? " ★" : ""}
                      </td>
                      <td>{format(row.train_loss, 5)}</td>
                      <td>{format(row.validation_loss, 5)}</td>
                      <td>{(row.train_accuracy * 100).toFixed(2)}%</td>
                      <td>{(row.validation_accuracy * 100).toFixed(2)}%</td>
                      <td>{format(row.seconds, 1)} s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!run.history.length && (
              <p className="mt-empty">
                第一轮尚未完成，暂不绘制曲线或填充估算指标。
              </p>
            )}
          </ExperimentPanel>
        </>
      )}
    </div>
  );
}
