import { MathFormula } from "../components/MathFormula";
import { useEffect, useRef, useState } from "react";
import { Download, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import {
  Chart,
  ErrorMessage,
  ExperimentPanel,
  Result,
  Slider,
} from "../components/ui";
import { calculate, format } from "../lib/api";
import { useLessonTask } from "../lib/learningProgress";
import { isXorTaskComplete } from "../lib/advancedTaskChecks";
import { flattenParameters } from "../lib/xor";
import type {
  HistoryPoint,
  ParameterUpdate,
  XorSnapshot,
  XorStepResponse,
} from "../lib/xor";
import XorNetwork from "./XorNetwork";
import XorBoundary from "./XorBoundary";
import "./xor.css";

type RecordedPoint = HistoryPoint & { learning_rate?: number };
const maxSteps = 5000;

export default function XorExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [snapshot, setSnapshot] = useState<XorSnapshot | null>(null);
  const [history, setHistory] = useState<RecordedPoint[]>([]);
  const [updates, setUpdates] = useState<ParameterUpdate[]>([]);
  const [rate, setRate] = useState(1);
  const [lastRate, setLastRate] = useState(1);
  const [seedInput, setSeedInput] = useState("7");
  const [modelSeed, setModelSeed] = useState(7);
  const [mode, setMode] = useState<"forward" | "gradient">("forward");
  const [selectedParameter, setSelectedParameter] = useState<string | null>(
    "b2",
  );
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState("正在初始化网络…");
  const current = useRef<XorSnapshot | null>(null);
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const paused = useRef(true);

  function cancel() {
    version.current++;
    request.current?.abort();
    request.current = null;
    paused.current = true;
    setRunning(false);
    setBusy(false);
  }

  function commit(result: XorSnapshot) {
    current.current = result;
    setSnapshot(result);
  }

  async function initialize(seed: number) {
    cancel();
    const token = version.current;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    setNotice("正在初始化网络…");
    current.current = null;
    setSnapshot(null);
    setHistory([]);
    setUpdates([]);
    try {
      const result = await calculate<XorSnapshot>(
        "/xor/init",
        { seed },
        controller.signal,
      );
      if (token !== version.current) return;
      commit(result);
      setModelSeed(seed);
      setHistory([{ step: 0, loss: result.loss, accuracy: result.accuracy }]);
      setNotice("网络已初始化。先观察预测，再尝试一次参数更新。");
    } catch (cause) {
      if (token === version.current && !controller.signal.aborted)
        setError(
          cause instanceof Error ? cause.message : "网络初始化失败，请重试。",
        );
    } finally {
      if (token === version.current) {
        request.current = null;
        setBusy(false);
      }
    }
  }

  useEffect(() => {
    void initialize(7);
    return () => {
      version.current++;
      request.current?.abort();
      request.current = null;
      paused.current = true;
    };
    // Initialization is owned by this mounted experiment, not by draft controls.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function train(steps: number) {
    const source = current.current;
    if (request.current || !source || source.iteration >= maxSteps) return;
    const token = ++version.current;
    const controller = new AbortController();
    request.current = controller;
    const count = Math.min(steps, maxSteps - source.iteration);
    setBusy(true);
    setError(null);
    try {
      const result = await calculate<XorStepResponse>(
        "/xor/step",
        {
          parameters: source.parameters,
          learning_rate: rate,
          steps: count,
          iteration: source.iteration,
          selected_sample: source.selected.index,
        },
        controller.signal,
      );
      if (token !== version.current) return;
      commit(result.snapshot);
      setHistory((previous) => [
        ...previous,
        ...result.history.map((row) => ({ ...row, learning_rate: rate })),
      ]);
      setUpdates(result.updates);
      setLastRate(rate);
      onInteract();
      if (result.snapshot.accuracy === 1 && result.snapshot.loss < 0.02) {
        paused.current = true;
        setRunning(false);
        setNotice(
          "四个训练样本均预测正确，平均 BCE < 0.02。自动训练已停止，仍可单步观察。",
        );
      } else if (result.snapshot.iteration >= maxSteps) {
        paused.current = true;
        setRunning(false);
        setNotice("已达到 5000 步上限，可重新初始化并比较不同学习率。");
      } else
        setNotice(
          paused.current
            ? `已完成第 ${result.snapshot.iteration} 步。图中显示更新后的真实网络。`
            : "正在训练：每次更新都使用四个样本的平均梯度。",
        );
    } catch (cause) {
      if (token === version.current && !controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "训练失败，请重试。");
        paused.current = true;
        setRunning(false);
      }
    } finally {
      if (token === version.current) {
        request.current = null;
        setBusy(false);
      }
    }
  }

  useEffect(() => {
    if (!running || busy || !snapshot || snapshot.iteration >= maxSteps) return;
    const timer = window.setTimeout(() => {
      if (!paused.current) void train(20);
    }, 250);
    return () => window.clearTimeout(timer);
    // Each request consumes the last committed model; no overlapping training requests.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, busy, snapshot, rate]);

  async function selectSample(index: number) {
    const source = current.current;
    if (!source) return;
    cancel();
    const token = version.current;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    try {
      const result = await calculate<XorSnapshot>(
        "/xor/inspect",
        {
          parameters: source.parameters,
          iteration: source.iteration,
          selected_sample: index,
        },
        controller.signal,
      );
      if (token === version.current) {
        commit(result);
        setNotice("已暂停训练并切换观察样本，模型参数保持不变。");
        onInteract();
      }
    } catch (cause) {
      if (token === version.current && !controller.signal.aborted)
        setError(cause instanceof Error ? cause.message : "样本切换失败。");
    } finally {
      if (token === version.current) {
        request.current = null;
        setBusy(false);
      }
    }
  }

  function saveFile(name: string, contents: string, type: string) {
    const url = URL.createObjectURL(new Blob([contents], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const parameterValues = snapshot
    ? flattenParameters(snapshot.parameters)
    : [];
  const gradients = snapshot
    ? new Map(
        flattenParameters(snapshot.gradients).map((p) => [p.name, p.value]),
      )
    : new Map<string, number>();
  const sampleGradients = snapshot
    ? new Map(
        flattenParameters(snapshot.sample_gradients).map((p) => [
          p.name,
          p.value,
        ]),
      )
    : new Map<string, number>();
  const selected = parameterValues.find((p) => p.name === selectedParameter);
  const lastUpdate = updates.find((p) => p.name === selectedParameter);
  const reachedLimit = (snapshot?.iteration ?? 0) >= maxSteps;
  useLessonTask("xor", isXorTaskComplete(snapshot));

  return (
    <div className="xor-experiment">
      <ExperimentPanel
        title="让一个小网络，学会 XOR"
        eyebrow="2 INPUTS → 4 TANH → 1 SIGMOID"
        action={
          <span className="live-tag">
            <i />
            NumPy 真实训练
          </span>
        }
      >
        <div className="xor-training-bar">
          <button
            className="primary-button"
            disabled={!snapshot || reachedLimit || (!running && busy)}
            onClick={() => {
              paused.current = running;
              setRunning(!running);
              if (running)
                setNotice("已暂停；正在进行的这一批更新完成后停下。");
              onInteract();
            }}
          >
            {running ? <Pause size={16} /> : <Play size={16} />}
            {running ? "暂停训练" : "自动训练"}
          </button>
          <button
            className="secondary-button"
            disabled={!snapshot || busy || running || reachedLimit}
            onClick={() => void train(1)}
          >
            <SkipForward size={15} />
            单步更新
          </button>
          <button
            className="secondary-button"
            disabled={!snapshot || busy || running || reachedLimit}
            onClick={() => void train(100)}
          >
            训练 100 步
          </button>
          <span className="xor-step-label">
            STEP <strong>{snapshot?.iteration ?? 0}</strong>
          </span>
        </div>
        <div className="xor-setup">
          <Slider
            label="学习率"
            symbol="η"
            value={rate}
            min={0.01}
            max={2}
            step={0.01}
            onChange={(value) => {
              cancel();
              setRate(value);
              setNotice("学习率已调整，将用于后续更新；已有参数与历史保留。");
              onInteract();
            }}
          />
          <div className="xor-seed-control">
            <label htmlFor="xor-seed">
              随机种子 <span>当前模型 seed {modelSeed}</span>
            </label>
            <div>
              <input
                id="xor-seed"
                type="number"
                min={0}
                max={9999}
                step={1}
                value={seedInput}
                onChange={(e) => setSeedInput(e.target.value)}
              />
              <button
                className="secondary-button"
                onClick={() => {
                  const seed = Number(seedInput);
                  if (
                    !seedInput.trim() ||
                    !Number.isInteger(seed) ||
                    seed < 0 ||
                    seed > 9999
                  ) {
                    setError("随机种子请输入 0–9999 的整数。");
                    return;
                  }
                  void initialize(seed);
                  onInteract();
                }}
              >
                <RotateCcw size={14} />
                重新初始化
              </button>
            </div>
          </div>
        </div>
        <p className="xor-run-notice" role="status">
          {notice}
          {busy ? " · 计算中…" : ""}
        </p>
        <div className="result-grid three-results">
          <Result
            label="四样本平均 BCE"
            value={snapshot ? format(snapshot.loss, 6) : "—"}
            hint="从输出 logits 稳定计算"
          />
          <Result
            label="四个训练点的准确率"
            value={snapshot ? `${snapshot.accuracy * 100}%` : "—"}
            hint="只衡量这四个教学样本"
          />
          <Result label="可训练参数" value="17" hint="权重与偏置全部参与更新" />
        </div>
        <ErrorMessage error={error} />
        {!snapshot && !busy && (
          <button
            className="secondary-button full-width"
            onClick={() => void initialize(modelSeed)}
          >
            重试初始化
          </button>
        )}
      </ExperimentPanel>

      {snapshot && (
        <>
          <div className="xor-visual-grid">
            <ExperimentPanel
              title="打开网络，观察每一层"
              eyebrow="INSPECT THE NETWORK"
              action={
                <div
                  className="segmented xor-mode"
                  role="group"
                  aria-label="网络观察模式"
                >
                  <button
                    aria-pressed={mode === "forward"}
                    onClick={() => {
                      setMode("forward");
                      onInteract();
                    }}
                  >
                    前向数值
                  </button>
                  <button
                    aria-pressed={mode === "gradient"}
                    onClick={() => {
                      setMode("gradient");
                      onInteract();
                    }}
                  >
                    样本梯度
                  </button>
                </div>
              }
            >
              <XorNetwork
                snapshot={snapshot}
                mode={mode}
                selectedParameter={selectedParameter}
                onSelectParameter={(name) => {
                  setSelectedParameter(name);
                  onInteract();
                }}
              />
              <div className="xor-parameter-inspect" aria-live="polite">
                <span className="small-label">
                  选中参数 · {selectedParameter}
                </span>
                <div>
                  <span>
                    当前值{" "}
                    <strong>
                      {selected ? format(selected.value, 5) : "—"}
                    </strong>
                  </span>
                  <span>
                    当前样本梯度{" "}
                    <strong>
                      {format(
                        sampleGradients.get(selectedParameter || "") ?? 0,
                        5,
                      )}
                    </strong>
                  </span>
                  <span>
                    四样本平均梯度{" "}
                    <strong>
                      {format(gradients.get(selectedParameter || "") ?? 0, 5)}
                    </strong>
                  </span>
                </div>
              </div>
              <details className="xor-forward-details">
                <summary>逐层展开当前样本的前向与反向计算</summary>
                <p>z₁ = xW₁ + b₁，a₁ = tanh(z₁)</p>
                <div className="table-scroll">
                  <table className="sample-table">
                    <thead>
                      <tr>
                        <th>隐藏单元</th>
                        <th>加权和 z₁</th>
                        <th>激活值 a₁</th>
                        <th>梯度 ∂L/∂z₁</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.selected.z1.map((z, j) => (
                        <tr key={j}>
                          <td>h{j + 1}</td>
                          <td>{format(z, 5)}</td>
                          <td>{format(snapshot.selected.a1[j], 5)}</td>
                          <td>{format(snapshot.sample_gradients.b1[j], 5)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>
                  z₂ = a₁W₂ + b₂ ={" "}
                  <strong>{format(snapshot.selected.z2, 5)}</strong>
                  <br />p = sigmoid(z₂) ={" "}
                  <strong>{format(snapshot.selected.output, 5)}</strong>
                  <br />
                  单样本 BCE ={" "}
                  <strong>{format(snapshot.selected.loss, 5)}</strong>
                </p>
                <p>
                  δ₂ = ∂L/∂z₂ = p − y ={" "}
                  <strong>{format(snapshot.sample_gradients.b2, 5)}</strong>
                  <br />
                  δ₁ⱼ = δ₂ × W₂ⱼ × (1 − a₁ⱼ²)
                </p>
                <small>
                  这里是所选单个样本的损失与梯度。全量训练把四个样本的梯度取平均；所有数值来自当前同一份参数。
                </small>
              </details>
              <p className="panel-note">
                当前图显示第 {snapshot.selected.index + 1}{" "}
                个样本；训练时把四个样本的梯度取平均，再更新全部参数。
              </p>
            </ExperimentPanel>
            <ExperimentPanel
              title="非线性的边界，逐渐形成"
              eyebrow="DECISION PROBABILITIES"
            >
              <XorBoundary
                snapshot={snapshot}
                onSelectSample={(index) => void selectSample(index)}
              />
            </ExperimentPanel>
          </div>

          <ExperimentPanel
            title="四个输入，四个真实预测"
            eyebrow="XOR TRUTH TABLE"
          >
            <div
              className="xor-samples"
              role="group"
              aria-label="选择 XOR 观察样本"
            >
              {snapshot.samples.map((sample, i) => (
                <button
                  key={i}
                  className={snapshot.selected.index === i ? "selected" : ""}
                  aria-pressed={snapshot.selected.index === i}
                  onClick={() => void selectSample(i)}
                  disabled={busy}
                >
                  <span className="xor-sample-input">
                    {sample.inputs.join(" ⊕ ")} <span>→ {sample.target}</span>
                  </span>
                  <span>输出 {format(sample.probability, 4)}</span>
                  <strong
                    className={
                      sample.prediction === sample.target
                        ? "xor-correct"
                        : "xor-incorrect"
                    }
                  >
                    预测 {sample.prediction} ·{" "}
                    {sample.prediction === sample.target ? "正确" : "待学习"}
                  </strong>
                </button>
              ))}
            </div>
            <p className="panel-note">
              相同输入输出 0，不同输入输出 1。分类阈值为 0.5；输出是模型的
              sigmoid 值，不代表经过校准的可靠性。
            </p>
          </ExperimentPanel>

          <div className="xor-visual-grid">
            <ExperimentPanel
              title="损失，来自每一次真实更新"
              eyebrow="FULL-BATCH TRAINING HISTORY"
              action={
                <button
                  className="text-button"
                  onClick={() =>
                    saveFile(
                      "zhiwei-xor-history.csv",
                      "\uFEFFstep,mean_bce,training_accuracy,learning_rate\n" +
                        history
                          .map(
                            (row) =>
                              `${row.step},${row.loss},${row.accuracy},${row.learning_rate ?? ""}`,
                          )
                          .join("\n"),
                      "text/csv;charset=utf-8",
                    )
                  }
                >
                  <Download size={14} />
                  导出 CSV
                </button>
              }
            >
              <Chart
                series={[
                  {
                    points: history.map((row) => ({
                      x: row.step,
                      y: row.loss,
                    })),
                  },
                ]}
                xLabel="参数更新 step"
                yLabel="平均 BCE"
                ariaLabel="XOR 网络每一步真实全量训练的平均交叉熵损失"
                markers={[
                  {
                    x: snapshot.iteration,
                    y: snapshot.loss,
                    label: "当前损失",
                  },
                ]}
              />
              <p className="panel-note">
                每个 step 使用全部 4 个样本。自动训练每次请求执行 20
                步，曲线保留其中每一步的结果。
              </p>
            </ExperimentPanel>
            <ExperimentPanel
              title="把梯度，变成参数的变化"
              eyebrow="LAST COMMITTED UPDATE"
            >
              <div className="xor-update-formula">
                <MathFormula tex={String.raw`\theta_{t+1}=\theta_t-\eta\,\overline{g}`} block />
                {lastUpdate ? (
                  <>
                    <strong>{lastUpdate.name}</strong>
                    <p>
                      {format(lastUpdate.before, 5)} − {format(lastRate)} × (
                      {format(lastUpdate.gradient, 5)})
                    </p>
                    <strong>= {format(lastUpdate.after, 5)}</strong>
                    <small>第 {snapshot.iteration} 步已应用的参数更新</small>
                  </>
                ) : (
                  <p>点击「单步更新」，查看一次真实参数变化。</p>
                )}
              </div>
              <p className="panel-note">
                下表“使用的梯度”来自最后一次更新前。上方网络与当前梯度来自更新后的参数，两者具有不同的计算时刻。
              </p>
              <button
                className="secondary-button full-width"
                onClick={() =>
                  saveFile(
                    "zhiwei-xor-experiment.json",
                    JSON.stringify(
                      {
                        format: "zhiwei-xor-v1",
                        architecture: [2, 4, 1],
                        hidden_activation: "tanh",
                        output_activation: "sigmoid",
                        loss: "mean_bce_from_logits",
                        seed: modelSeed,
                        next_learning_rate: rate,
                        snapshot,
                      },
                      null,
                      2,
                    ),
                    "application/json",
                  )
                }
              >
                <Download size={14} />
                导出当前实验 JSON
              </button>
            </ExperimentPanel>
          </div>

          <ExperimentPanel
            title="17 个参数，每一项都可以追踪"
            eyebrow="PARAMETERS & GRADIENTS"
          >
            <div className="table-scroll">
              <table className="sample-table xor-parameter-table">
                <thead>
                  <tr>
                    <th>选择参数</th>
                    <th>更新前</th>
                    <th>使用的平均梯度</th>
                    <th>更新后 / 当前值</th>
                  </tr>
                </thead>
                <tbody>
                  {parameterValues.map((parameter) => {
                    const update = updates.find(
                      (row) => row.name === parameter.name,
                    );
                    return (
                      <tr
                        key={parameter.name}
                        className={
                          selectedParameter === parameter.name
                            ? "xor-selected-row"
                            : ""
                        }
                      >
                        <td>
                          <button
                            className="xor-param-button"
                            aria-pressed={selectedParameter === parameter.name}
                            onClick={() => {
                              setSelectedParameter(parameter.name);
                              onInteract();
                            }}
                          >
                            {parameter.name}
                          </button>
                        </td>
                        <td>{update ? format(update.before, 5) : "—"}</td>
                        <td>
                          {update ? format(update.gradient, 5) : "尚未更新"}
                        </td>
                        <td>{format(parameter.value, 5)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="panel-note">
              参数索引从 0
              开始。点击参数可在网络中定位；改变学习率不会清空已有训练，重新初始化才会重置权重和曲线。
            </p>
          </ExperimentPanel>
        </>
      )}
    </div>
  );
}
