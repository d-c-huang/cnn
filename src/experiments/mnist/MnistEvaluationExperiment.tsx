import { useEffect, useRef, useState } from "react";
import { ArrowRight, Download, X } from "lucide-react";
import { ErrorMessage, ExperimentPanel, Result } from "../../components/ui";
import { format } from "../../lib/api";
import {
  mnistRequest,
  selectRunId,
  selectedRunId,
  statusLabels,
} from "../../lib/mnist";
import type { ErrorPage, ErrorSample, MnistStatus, Run } from "../../lib/mnist";
import { downloadText, useMnistResource } from "../../lib/useMnist";
import ConfusionMatrix from "./ConfusionMatrix";
import ErrorSamples from "./ErrorSamples";
import DigitImage from "./DigitImage";
import "./mnist-evaluation.css";

export default function MnistEvaluationExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const records = useMnistResource<{ runs: Run[] }>("/runs");
  const dataset = useMnistResource<MnistStatus>("/status");
  const [preparing, setPreparing] = useState(false);
  const [prepareError, setPrepareError] = useState<string | null>(null);
  const prepareRequest = useRef<AbortController | null>(null);
  const [chosenId, setChosenId] = useState<string | null>(selectedRunId);
  const [cell, setCell] = useState<{ truth: number; predicted: number } | null>(
    null,
  );
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<ErrorPage | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sample, setSample] = useState<ErrorSample | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const errorVersion = useRef(0);
  const runs = records.data?.runs || [];
  const run =
    runs.find((item) => item.id === chosenId) ||
    runs.find((item) => item.evaluation) ||
    runs[0] ||
    null;
  const runId = run?.id;
  const evaluation = run?.evaluation;
  const checkpointEpoch = evaluation?.checkpoint_epoch;
  const datasetReady = dataset.data?.dataset.status === "ready";

  useEffect(() => () => prepareRequest.current?.abort(), []);
  async function prepareImages() {
    if (prepareRequest.current) return;
    const controller = new AbortController();
    prepareRequest.current = controller;
    setPreparing(true);
    setPrepareError(null);
    try {
      await mnistRequest<MnistStatus>("/prepare", {
        body: {},
        signal: controller.signal,
      });
      if (!controller.signal.aborted) {
        void dataset.refresh();
        onInteract();
      }
    } catch (cause) {
      if (!controller.signal.aborted)
        setPrepareError(
          cause instanceof Error ? cause.message : "加载测试图像失败。",
        );
    } finally {
      if (!controller.signal.aborted) setPreparing(false);
      if (prepareRequest.current === controller) prepareRequest.current = null;
    }
  }

  useEffect(() => {
    const token = ++errorVersion.current;
    const controller = new AbortController();
    setPage(null);
    setError(null);
    if (!runId || checkpointEpoch === undefined || !datasetReady) {
      setPending(false);
      return () => controller.abort();
    }
    setPending(true);
    const query = new URLSearchParams({ offset: String(offset), limit: "12" });
    if (cell) {
      query.set("true_label", String(cell.truth));
      query.set("predicted_label", String(cell.predicted));
    }
    void mnistRequest<ErrorPage>(`/runs/${runId}/errors?${query}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (token === errorVersion.current) setPage(result);
      })
      .catch((cause) => {
        if (token === errorVersion.current && !controller.signal.aborted)
          setError(
            cause instanceof Error ? cause.message : "无法读取错误样本。",
          );
      })
      .finally(() => {
        if (token === errorVersion.current) setPending(false);
      });
    return () => {
      errorVersion.current++;
      controller.abort();
    };
  }, [
    runId,
    checkpointEpoch,
    cell?.truth,
    cell?.predicted,
    offset,
    datasetReady,
  ]);

  function choose(id: string) {
    setChosenId(id);
    selectRunId(id);
    setCell(null);
    setOffset(0);
    setSample(null);
    dialog.current?.close();
    onInteract();
  }
  function openSample(item: ErrorSample) {
    setSample(item);
    dialog.current?.showModal();
    onInteract();
  }

  return (
    <div className="me-experiment">
      <ExperimentPanel
        title="把最终模型，交给独立测试集"
        eyebrow="HELD-OUT TEST EVALUATION"
      >
        <div className="me-select-run">
          <label htmlFor="mnist-evaluation-run">选择实验记录</label>
          <select
            id="mnist-evaluation-run"
            value={run?.id || ""}
            onChange={(e) => choose(e.target.value)}
            disabled={!runs.length}
          >
            <option value="" disabled>
              尚无实验记录
            </option>
            {runs.map((item) => (
              <option key={item.id} value={item.id}>
                {item.config.model.toUpperCase()} · {item.id.slice(0, 8)} ·{" "}
                {statusLabels[item.status]} ·{" "}
                {item.train_samples.toLocaleString()} 张训练图像
              </option>
            ))}
          </select>
          <a className="text-button" href="#mnist-train">
            训练 MLP 或 CNN <ArrowRight size={14} />
          </a>
        </div>
        <p className="panel-note">
          最佳权重只根据验证损失选择。训练全部结束后，载入该权重，在从未参与参数更新和模型选择的官方测试集上计算以下结果。
        </p>
        <ErrorMessage error={records.error} />
        {records.pending && (
          <p className="me-empty">正在读取持久化的实验记录…</p>
        )}
        {!records.pending && !run && (
          <p className="me-empty">
            先训练一个模型，完成后即可查看独立测试结果。
          </p>
        )}
        {run && !evaluation && (
          <div className="me-pending-evaluation" role="status">
            <strong>{statusLabels[run.status]}</strong>
            <p>{run.error || run.message}</p>
            <p>
              只有完整训练结束、选择验证集最佳模型后，才会生成测试报告。取消或中断的任务不会显示估算测试指标。
            </p>
          </div>
        )}
        {run && evaluation && (
          <>
            <div className="result-grid me-results">
              <Result
                label="独立测试准确率"
                value={`${(evaluation.accuracy * 100).toFixed(2)}%`}
                hint={`${evaluation.total.toLocaleString()} 张官方测试图像`}
              />
              <Result
                label="测试交叉熵损失"
                value={format(evaluation.loss, 5)}
              />
              <Result
                label="分类错误样本"
                value={evaluation.error_count.toLocaleString()}
                hint="可从混淆矩阵筛选查看"
              />
              <Result
                label="本次评估的权重"
                value={`Epoch ${evaluation.checkpoint_epoch}`}
                hint={`由最低验证损失选择 · ${run.config.model.toUpperCase()}`}
              />
            </div>
            <div className="me-report-actions">
              <button
                className="text-button"
                onClick={() =>
                  downloadText(
                    `mnist-${run.config.model}-${run.id.slice(0, 8)}-report.json`,
                    JSON.stringify(run, null, 2),
                  )
                }
              >
                <Download size={14} />
                导出完整评估报告 JSON
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
            </div>
          </>
        )}
      </ExperimentPanel>

      {evaluation && (
        <>
          <ExperimentPanel
            title="模型把哪些数字认错了"
            eyebrow="CONFUSION MATRIX · TRUE × PREDICTED"
          >
            <ConfusionMatrix
              evaluation={evaluation}
              selected={cell}
              onSelect={(next) => {
                setCell(next);
                setOffset(0);
                onInteract();
              }}
            />
          </ExperimentPanel>
          <ExperimentPanel
            title={
              cell
                ? `筛选错例：真实 ${cell.truth} → 预测 ${cell.predicted}`
                : "打开每一个错误预测"
            }
            eyebrow="REAL TEST ERRORS"
            action={
              cell ? (
                <button
                  className="text-button"
                  onClick={() => {
                    setCell(null);
                    setOffset(0);
                  }}
                >
                  查看全部错例
                </button>
              ) : undefined
            }
          >
            {cell?.truth === cell?.predicted && cell && (
              <p className="panel-note">
                对角线表示正确分类，因此该筛选条件不会返回错误样本。
              </p>
            )}
            {datasetReady ? (
              <ErrorSamples
                page={page}
                pending={pending}
                error={error}
                onPage={(next) => {
                  setOffset(next);
                  onInteract();
                }}
                onSelect={openSample}
              />
            ) : (
              <div className="me-pending-evaluation">
                <p>
                  评估报告已保存；查看原始错例需要加载 MNIST
                  图像缓存。加载图像不会重新训练或重新评估模型。
                </p>
                <button
                  className="secondary-button"
                  disabled={
                    preparing ||
                    dataset.pending ||
                    dataset.data?.dataset.status === "preparing"
                  }
                  onClick={() => void prepareImages()}
                >
                  {preparing || dataset.data?.dataset.status === "preparing"
                    ? "正在加载图像…"
                    : "加载 MNIST 测试图像"}
                </button>
                <ErrorMessage
                  error={
                    prepareError ||
                    dataset.error ||
                    dataset.data?.dataset.error ||
                    null
                  }
                />
              </div>
            )}
          </ExperimentPanel>
          <ExperimentPanel
            title="分别检查每个数字的表现"
            eyebrow="PER-CLASS METRICS"
          >
            <div className="table-scroll">
              <table className="sample-table me-class-table">
                <thead>
                  <tr>
                    <th>真实类别</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>测试样本数</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluation.class_metrics.map((row) => (
                    <tr key={row.label}>
                      <td>{row.label}</td>
                      <td>{(row.precision * 100).toFixed(2)}%</td>
                      <td>{(row.recall * 100).toFixed(2)}%</td>
                      <td>{(row.f1 * 100).toFixed(2)}%</td>
                      <td>{row.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="panel-note">
              Precision：预测为该数字的样本中有多少正确；Recall：真实为该数字的样本中找回多少；F1
              综合二者。没有对应样本时指标记为 0。
            </p>
          </ExperimentPanel>
        </>
      )}

      <dialog
        ref={dialog}
        className="me-error-dialog"
        onClose={() => setSample(null)}
      >
        <div className="dialog-heading">
          <h2>测试集错例详情</h2>
          <button
            className="icon-button"
            aria-label="关闭错例详情"
            onClick={() => dialog.current?.close()}
          >
            <X size={20} />
          </button>
        </div>
        {sample && (
          <>
            <div className="me-error-overview">
              <DigitImage
                pixels={sample.pixels}
                label={`官方测试集索引 ${sample.index}，真实 ${sample.label}，预测 ${sample.prediction}`}
                size={140}
              />
              <div>
                <span className="small-label">
                  官方测试集索引 {sample.index}
                </span>
                <strong>
                  真实 {sample.label} → 预测 {sample.prediction}
                </strong>
                <p>预测类别模型分数 {(sample.confidence * 100).toFixed(2)}%</p>
              </div>
            </div>
            <div
              className="me-probabilities"
              aria-label="十个数字类别的 softmax 输出"
            >
              {sample.probabilities.map((probability, label) => (
                <div key={label}>
                  <span>{label}</span>
                  <div>
                    <i
                      style={{ width: `${probability * 100}%` }}
                      className={
                        label === sample.prediction
                          ? "predicted"
                          : label === sample.label
                            ? "truth"
                            : ""
                      }
                    />
                  </div>
                  <strong>{(probability * 100).toFixed(2)}%</strong>
                </div>
              ))}
            </div>
            <p className="panel-note">
              图像为该次测试的原始 28×28 灰度图。分数来自同一最佳 checkpoint 的
              softmax 输出，不是经过校准的可靠性保证。
            </p>
          </>
        )}
      </dialog>
    </div>
  );
}
