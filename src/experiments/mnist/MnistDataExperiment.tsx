import { useEffect, useRef, useState } from "react";
import { Check, Database, Download, RefreshCw } from "lucide-react";
import { ErrorMessage, ExperimentPanel, Result } from "../../components/ui";
import {
  loadMnistConfig,
  mnistRequest,
  storeMnistConfig,
} from "../../lib/mnist";
import type { DataSplit, MnistStatus, SplitConfig } from "../../lib/mnist";
import DigitImage from "./DigitImage";
import "./mnist-data.css";

const splitNames = {
  train: "训练集",
  validation: "验证集",
  test: "测试集",
} as const;
const datasetStates: Record<MnistStatus["dataset"]["status"], string> = {
  missing: "等待准备",
  preparing: "正在准备",
  ready: "数据已就绪",
  failed: "准备失败",
};
const number = (value: number) => value.toLocaleString("zh-CN");

export default function MnistDataExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const initial = useRef(loadMnistConfig());
  const [applied, setApplied] = useState<SplitConfig>(() => ({
    seed: initial.current.seed,
    validation_ratio: initial.current.validation_ratio,
    train_limit: initial.current.train_limit,
  }));
  const [seed, setSeed] = useState(String(initial.current.seed));
  const [ratio, setRatio] = useState(initial.current.validation_ratio);
  const [limit, setLimit] = useState<number | null>(
    initial.current.train_limit,
  );
  const [status, setStatus] = useState<MnistStatus | null>(null);
  const [statusPending, setStatusPending] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [preparePending, setPreparePending] = useState(false);
  const [prepareError, setPrepareError] = useState<string | null>(null);
  const [statusRevision, setStatusRevision] = useState(0);
  const [split, setSplit] = useState<DataSplit | null>(null);
  const [splitPending, setSplitPending] = useState(false);
  const [splitError, setSplitError] = useState<string | null>(null);
  const [splitRevision, setSplitRevision] = useState(0);
  const [inputError, setInputError] = useState<string | null>(null);
  const [gallery, setGallery] = useState<keyof typeof splitNames>("train");
  const prepareController = useRef<AbortController | null>(null);
  const prepareVersion = useRef(0);

  useEffect(() => {
    let live = true;
    const controller = new AbortController();
    let timer: number | undefined;
    async function checkStatus() {
      try {
        const result = await mnistRequest<MnistStatus>("/status", {
          signal: controller.signal,
        });
        if (!live) return;
        setStatus(result);
        setStatusError(null);
        if (
          result.dataset.status === "preparing" ||
          result.dataset.status === "ready"
        )
          setPrepareError(null);
        setStatusPending(false);
        // Schedule only after this request completes: polling never overlaps.
        if (result.dataset.status === "preparing")
          timer = window.setTimeout(() => void checkStatus(), 1000);
      } catch (cause) {
        if (!live || controller.signal.aborted) return;
        setStatusPending(false);
        setStatusError(
          cause instanceof Error ? cause.message : "无法读取 MNIST 数据状态。",
        );
      }
    }
    setStatusPending(true);
    void checkStatus();
    return () => {
      live = false;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [statusRevision]);

  useEffect(() => {
    if (status?.dataset.status !== "ready") {
      setSplit(null);
      setSplitPending(false);
      setSplitError(null);
      return;
    }
    let live = true;
    const controller = new AbortController();
    setSplit(null);
    setSplitPending(true);
    setSplitError(null);
    void mnistRequest<DataSplit>("/split", {
      body: applied,
      signal: controller.signal,
    })
      .then((result) => {
        if (live) setSplit(result);
      })
      .catch((cause) => {
        if (live && !controller.signal.aborted)
          setSplitError(
            cause instanceof Error ? cause.message : "数据划分失败，请重试。",
          );
      })
      .finally(() => {
        if (live) setSplitPending(false);
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [status?.dataset.status, applied, splitRevision]);

  useEffect(
    () => () => {
      prepareVersion.current++;
      prepareController.current?.abort();
    },
    [],
  );

  async function prepare() {
    if (prepareController.current) return;
    const token = ++prepareVersion.current;
    const controller = new AbortController();
    prepareController.current = controller;
    setPreparePending(true);
    setPrepareError(null);
    onInteract();
    try {
      const result = await mnistRequest<MnistStatus>("/prepare", {
        body: {},
        signal: controller.signal,
      });
      if (token !== prepareVersion.current) return;
      setStatus(result);
      setStatusRevision((value) => value + 1);
    } catch (cause) {
      if (token === prepareVersion.current && !controller.signal.aborted) {
        setPrepareError(
          cause instanceof Error ? cause.message : "无法开始准备数据。",
        );
        // A response timeout does not mean the background preparation stopped.
        setStatusRevision((value) => value + 1);
      }
    } finally {
      if (token === prepareVersion.current) {
        prepareController.current = null;
        setPreparePending(false);
      }
    }
  }

  function applySplit() {
    const nextSeed = Number(seed);
    if (
      !seed.trim() ||
      !Number.isInteger(nextSeed) ||
      nextSeed < 0 ||
      nextSeed > 9999
    ) {
      setInputError("随机种子请输入 0–9999 的整数。");
      return;
    }
    const next: SplitConfig = {
      seed: nextSeed,
      validation_ratio: ratio,
      train_limit: limit,
    };
    setInputError(null);
    storeMnistConfig({ ...loadMnistConfig(), ...next });
    setSplit(null);
    setSplitError(null);
    setApplied(next);
    setSplitRevision((value) => value + 1);
    onInteract();
  }

  const preparing = preparePending || status?.dataset.status === "preparing";
  const ready = status?.dataset.status === "ready";
  const dirty =
    seed !== String(applied.seed) ||
    ratio !== applied.validation_ratio ||
    limit !== applied.train_limit;
  const visibleSamples =
    split?.samples.filter((sample) => sample.split === gallery) || [];
  const allCount = split
    ? split.train_count + split.validation_count + split.test_count
    : 0;

  return (
    <div className="md-experiment">
      <ExperimentPanel
        title="把真实的手写数字，带进实验室"
        eyebrow="MNIST / 28 × 28 GRAYSCALE"
        action={
          <span className={`md-dataset-state ${ready ? "md-ready" : ""}`}>
            {ready ? <Check size={14} /> : <Database size={14} />}
            {status ? datasetStates[status.dataset.status] : "检查数据状态"}
          </span>
        }
      >
        <div className="md-runtime-row">
          <div>
            <span className="small-label">计算环境</span>
            <p>
              {status?.runtime.available
                ? `PyTorch ${status.runtime.torch_version} · CPU · ${status.runtime.threads} 线程`
                : status
                  ? "未检测到可用的 PyTorch / Torchvision"
                  : "正在检查本机计算服务…"}
            </p>
            {status?.runtime.available && (
              <small>Torchvision {status.runtime.torchvision_version}</small>
            )}
          </div>
          <button
            className={ready ? "secondary-button" : "primary-button"}
            disabled={preparing || statusPending || !status?.runtime.available}
            onClick={() => void prepare()}
          >
            {ready ? <RefreshCw size={15} /> : <Download size={15} />}
            {preparing
              ? "正在准备 MNIST…"
              : ready
                ? "重新检查数据"
                : "准备 MNIST 数据"}
          </button>
        </div>
        <p className="md-status-message" role="status">
          {status?.dataset.message || "读取本地缓存与下载状态…"}
        </p>
        <p className="panel-note">
          首次准备会下载官方 MNIST；后续复用本地缓存。数据包含 60,000
          个官方训练样本和 10,000 个独立测试样本，图片大小为 28 × 28。
        </p>
        {status && !status.runtime.available && (
          <div className="md-install-note">
            安装训练依赖后重启服务：
            <code>python -m pip install -r requirements.txt</code>
          </div>
        )}
        <ErrorMessage
          error={prepareError || statusError || status?.dataset.error || null}
        />
        {statusError && (
          <button
            className="text-button"
            onClick={() => setStatusRevision((value) => value + 1)}
          >
            <RefreshCw size={14} />
            重试状态检查
          </button>
        )}
      </ExperimentPanel>

      <div className="md-split-layout">
        <ExperimentPanel
          title="三份数据，各自承担不同的任务"
          eyebrow="STRATIFIED & REPRODUCIBLE SPLIT"
        >
          {split ? (
            <>
              <div
                className="md-split-bar"
                role="img"
                aria-label={`可用训练集 ${split.train_count} 个，验证集 ${split.validation_count} 个，独立测试集 ${split.test_count} 个`}
              >
                <div
                  className="md-train-part"
                  style={{ width: `${(split.train_count / allCount) * 100}%` }}
                />
                <div
                  className="md-validation-part"
                  style={{
                    width: `${(split.validation_count / allCount) * 100}%`,
                  }}
                />
                <div
                  className="md-test-part"
                  style={{ width: `${(split.test_count / allCount) * 100}%` }}
                />
              </div>
              <div className="md-split-legend">
                <span>
                  <i className="md-train-part" />
                  训练集 {number(split.train_count)}
                </span>
                <span>
                  <i className="md-validation-part" />
                  验证集 {number(split.validation_count)}
                </span>
                <span>
                  <i className="md-test-part" />
                  测试集 {number(split.test_count)}
                </span>
              </div>
              <div className="result-grid">
                <Result
                  label="本次实际训练"
                  value={number(split.effective_train_count)}
                  hint={`从 ${number(split.train_count)} 个可用训练样本中取用`}
                />
                <Result
                  label="每轮完整验证"
                  value={number(split.validation_count)}
                  hint="用验证损失选择最佳模型"
                />
                <Result
                  label="结束后独立测试"
                  value={number(split.test_count)}
                  hint="不参与训练或最佳轮次选择"
                />
              </div>
              <div className="md-split-explanation">
                <p>
                  种子 <strong>{split.seed}</strong>：先把官方 60,000
                  个训练样本按数字类别分层随机划分，再得到互不重叠的训练集与验证集。
                </p>
                <p>
                  {split.effective_train_count < split.train_count
                    ? `当前是训练子集实验：可用训练集有 ${number(split.train_count)} 个，本次分层抽取 ${number(split.effective_train_count)} 个实际训练；验证集与测试集仍完整保留。`
                    : "当前使用完整训练划分；训练集和验证集共同来自官方训练部分，官方测试集单独保留。"}
                </p>
              </div>
            </>
          ) : (
            <div className="md-split-empty" role="status">
              <Database size={27} strokeWidth={1.3} />
              <p>
                {splitPending
                  ? "正在读取真实标签并生成可复现划分…"
                  : ready
                    ? "应用配置，查看真实数据划分。"
                    : "准备好 MNIST 后，这里会显示真实划分与样本。"}
              </p>
            </div>
          )}
          <ErrorMessage error={splitError} />
          {splitError && (
            <button
              className="text-button"
              onClick={() => {
                setSplitRevision((value) => value + 1);
                setStatusRevision((value) => value + 1);
              }}
            >
              <RefreshCw size={14} />
              重新读取划分
            </button>
          )}
        </ExperimentPanel>

        <aside
          className="control-panel md-config-panel"
          aria-label="MNIST 数据划分配置"
        >
          <h3>配置数据划分</h3>
          <div className="md-form">
            <label htmlFor="mnist-data-seed">
              随机种子 <span>0–9999</span>
              <input
                id="mnist-data-seed"
                type="number"
                min={0}
                max={9999}
                step={1}
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
              />
            </label>
            <label htmlFor="mnist-validation-ratio">
              验证集占官方训练集
              <select
                id="mnist-validation-ratio"
                value={ratio}
                onChange={(event) => setRatio(Number(event.target.value))}
              >
                <option value={0.1}>10% · 6,000 个验证样本</option>
                <option value={0.2}>20% · 12,000 个验证样本</option>
              </select>
            </label>
            <label htmlFor="mnist-train-limit">
              实际训练规模
              <select
                id="mnist-train-limit"
                value={limit ?? "full"}
                onChange={(event) =>
                  setLimit(
                    event.target.value === "full"
                      ? null
                      : Number(event.target.value),
                  )
                }
              >
                <option value="full">使用完整训练划分</option>
                <option value={6000}>快速实验 · 6,000 个训练样本</option>
                <option value={12000}>中等实验 · 12,000 个训练样本</option>
              </select>
            </label>
          </div>
          <button
            className="primary-button full-width"
            disabled={!ready || splitPending}
            onClick={applySplit}
          >
            <Check size={15} />
            {splitPending ? "正在划分…" : "应用划分"}
          </button>
          <ErrorMessage error={inputError} />
          <p className="md-config-note" role="status">
            {dirty
              ? "有未应用的修改。预览仍显示上一次已应用的配置。"
              : "已应用配置会带入后面的 MLP / CNN 训练页面。"}
          </p>
          <p className="panel-note">
            训练子集只减少实际训练数量。官方独立测试集始终是 10,000
            个，不能用它选择学习率或最佳轮次。
          </p>
        </aside>
      </div>

      {split && (
        <>
          <ExperimentPanel
            title="每一个数字，都有自己的样本"
            eyebrow="REAL MNIST SAMPLES"
            action={
              <div
                className="segmented md-gallery-tabs"
                role="group"
                aria-label="选择样本所属数据集"
              >
                {(["train", "validation", "test"] as const).map((name) => (
                  <button
                    key={name}
                    aria-pressed={gallery === name}
                    onClick={() => {
                      setGallery(name);
                      onInteract();
                    }}
                  >
                    {splitNames[name]}
                  </button>
                ))}
              </div>
            }
          >
            <div className="md-sample-gallery">
              {visibleSamples.map((sample) => (
                <div
                  className="md-sample"
                  key={`${sample.split}-${sample.index}`}
                >
                  <DigitImage
                    pixels={sample.pixels}
                    label={`MNIST ${splitNames[sample.split]}样本 ${sample.index}，真实标签 ${sample.label}`}
                    size={62}
                  />
                  <span>
                    真实标签 <strong>{sample.label}</strong>
                  </span>
                  <small>索引 {sample.index}</small>
                </div>
              ))}
            </div>
            <p className="panel-note">
              展示当前{splitNames[gallery]}中的 {visibleSamples.length}{" "}
              个真实样本。训练/验证索引来自官方训练部分，测试索引来自独立测试部分；索引各自在对应官方文件中从
              0 开始。
            </p>
          </ExperimentPanel>
          <div className="md-detail-layout">
            <ExperimentPanel
              title="按类别核对这次划分"
              eyebrow="LABEL DISTRIBUTION"
            >
              <div className="table-scroll">
                <table className="sample-table md-distribution-table">
                  <thead>
                    <tr>
                      <th>数字</th>
                      <th>实际训练</th>
                      <th>验证</th>
                      <th>独立测试</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: 10 }, (_, label) => (
                      <tr key={label}>
                        <td>{label}</td>
                        <td>{number(split.train_class_counts[label] || 0)}</td>
                        <td>
                          {number(split.validation_class_counts[label] || 0)}
                        </td>
                        <td>{number(split.test_class_counts[label] || 0)}</td>
                      </tr>
                    ))}
                    <tr className="md-total-row">
                      <td>合计</td>
                      <td>{number(split.effective_train_count)}</td>
                      <td>{number(split.validation_count)}</td>
                      <td>{number(split.test_count)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="panel-note">
                数量来自真实标签统计。各类别样本数不完全相同，分层划分会尽量保留原有类别分布。
              </p>
            </ExperimentPanel>
            <ExperimentPanel
              title="把像素，转换成一致的模型输入"
              eyebrow="ONE PREPROCESSING PIPELINE"
            >
              <div className="md-preprocessing">
                <div>
                  <span>01</span>
                  <p>
                    原始灰度像素<strong>0 … 255</strong>
                  </p>
                </div>
                <div>
                  <span>02</span>
                  <p>
                    除以 255<strong>x = pixel / 255</strong>
                  </p>
                </div>
                <div>
                  <span>03</span>
                  <p>
                    使用预设的 MNIST 标准化参数
                    <strong>x标准 = (x − 0.1307) / 0.3081</strong>
                  </p>
                </div>
                <div>
                  <span>04</span>
                  <p>
                    保留图像形状<strong>[batch, 1, 28, 28]</strong>
                  </p>
                </div>
              </div>
              <p className="panel-note">
                训练、验证与测试采用同一套预处理，标准化后的值可以为负数。MLP
                会在模型内部展平图像；CNN 保留空间结构做卷积。
              </p>
              <div className="insight">
                验证集回答“选哪一轮模型”；独立测试集回答“选定的模型表现如何”。测试结果不会进入每轮训练曲线。
              </div>
            </ExperimentPanel>
          </div>
        </>
      )}
    </div>
  );
}
