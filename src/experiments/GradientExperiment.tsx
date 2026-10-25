import { useEffect, useRef, useState } from "react";
import { Download, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import {
  Chart,
  ControlsHeading,
  ErrorMessage,
  ExperimentPanel,
  Result,
  Slider,
} from "../components/ui";
import { calculate, format, useCalculation } from "../lib/api";

type GradientResponse = {
  weight_before: number;
  weight_after: number;
  gradient: number;
  loss_before: number;
  loss_after: number;
  converged: boolean;
  diverged: boolean;
};
type Step = {
  step: number;
  weight: number;
  loss: number;
  gradient: number | null;
};

export default function GradientExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [start, setStart] = useState(-2);
  const [rate, setRate] = useState(0.1);
  const [history, setHistory] = useState<Step[]>([]);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stepError, setStepError] = useState<string | null>(null);
  const [status, setStatus] = useState("准备好，迈出第一步。");
  const [lastUpdate, setLastUpdate] = useState<GradientResponse | null>(null);
  const last = history.at(-1);
  const weight = last?.weight ?? start;
  const { data, error } = useCalculation<GradientResponse>("/gradient", {
    weight,
    learning_rate: rate,
  });
  const inFlight = useRef(false);
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const stopped =
    (last?.step ?? 0) >= 200 ||
    (last?.loss ?? 0) > 1e8 ||
    (history.length > 1 && Math.abs(weight - 2) < 0.001);

  useEffect(
    () => () => {
      version.current++;
      request.current?.abort();
    },
    [],
  );

  async function step() {
    if (inFlight.current || stopped) return;
    inFlight.current = true;
    const currentVersion = version.current;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setStepError(null);
    try {
      const result = await calculate<GradientResponse>(
        "/gradient",
        { weight, learning_rate: rate },
        controller.signal,
      );
      if (version.current !== currentVersion) return;
      setHistory((previous) => {
        const entries = previous.length
          ? previous
          : [
              {
                step: 0,
                weight: result.weight_before,
                loss: result.loss_before,
                gradient: null,
              },
            ];
        return [
          ...entries,
          {
            step: entries.length,
            weight: result.weight_after,
            loss: result.loss_after,
            gradient: result.gradient,
          },
        ];
      });
      setLastUpdate(result);
      onInteract();
      if (result.converged) {
        setRunning(false);
        setStatus("已收敛：权重接近 2，损失接近 0。");
      } else if (
        result.loss_after > 1e8 ||
        Math.abs(result.weight_after) > 1e5
      ) {
        setRunning(false);
        setStatus("损失持续增大，已停止。试着减小学习率。");
      } else if ((last?.step ?? 0) + 1 >= 200) {
        setRunning(false);
        setStatus("已完成 200 步。可以重置后比较其他学习率。");
      } else if (result.diverged)
        setStatus("损失正在增大：学习率过大，让更新越过了最低点。");
      else if (Math.abs(result.loss_after - result.loss_before) < 1e-10)
        setStatus("在最低点两侧振荡：试着减小学习率。");
      else setStatus("损失正在下降，参数逐步靠近最低点。");
    } catch (cause) {
      if (!controller.signal.aborted) {
        setStepError(
          cause instanceof Error ? cause.message : "计算失败，请重试。",
        );
        setRunning(false);
      }
    } finally {
      inFlight.current = false;
      if (version.current === currentVersion) setBusy(false);
    }
  }

  useEffect(() => {
    if (!running || busy || stopped) return;
    const timer = window.setTimeout(() => void step(), 500);
    return () => window.clearTimeout(timer);
    // Re-schedule from the last committed result. There is never more than one request in flight.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, busy, stopped, weight, rate]);

  function reset(nextStart = start) {
    version.current++;
    request.current?.abort();
    setRunning(false);
    setBusy(false);
    setHistory([]);
    setLastUpdate(null);
    setStepError(null);
    setStatus("准备好，迈出第一步。");
    setStart(nextStart);
  }

  const loss = last?.loss ?? data?.loss_before;
  const curveWeights = [...history.map((s) => s.weight), weight, -3, 5].filter(
    (w) => Math.abs(w) < 1e4,
  );
  const lo = Math.min(...curveWeights) - 1,
    hi = Math.max(...curveWeights) + 1;
  const landscape = Array.from({ length: 101 }, (_, i) => {
    const x = lo + ((hi - lo) * i) / 100;
    return { x, y: 2 * (x - 2) ** 2 };
  });
  function exportHistory() {
    const csv =
      "\uFEFFstep,weight_after_update,loss_after_update,gradient_used_for_update\n" +
      history
        .map(
          (row) =>
            `${row.step},${row.weight},${row.loss},${row.gradient ?? ""}`,
        )
        .join("\n");
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "zhiwei-gradient-experiment.csv";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="experiment-layout">
      <div className="experiment-main">
        <ExperimentPanel
          title="让参数，沿着梯度走下去"
          eyebrow="GRADIENT DESCENT"
          action={
            <span className="live-tag">
              <i />
              NumPy 单步计算
            </span>
          }
        >
          <div className="plot-legend">
            <span>
              <i className="legend-line" />
              损失曲面的一维截面
            </span>
            <span>
              <i className="legend-dot lime" />
              当前权重
            </span>
          </div>
          <Chart
            ariaLabel="权重与损失的关系，当前权重逐步靠近最低点"
            series={[{ points: landscape }]}
            xLabel="权重 w"
            yLabel="损失 L"
            markers={
              loss === undefined || Math.abs(weight) > 1e4
                ? []
                : [
                    { x: weight, y: loss, label: "当前权重" },
                    {
                      x: 2,
                      y: 0,
                      label: "最优权重",
                      color: "var(--teal-soft)",
                    },
                  ]
            }
          />
          <div className="playback-bar">
            <button
              className="primary-button"
              disabled={stopped || (!running && busy)}
              onClick={() => {
                setRunning(!running);
                onInteract();
              }}
            >
              {running ? <Pause size={16} /> : <Play size={16} />}
              {running ? "暂停" : "自动运行"}
            </button>
            <button
              className="secondary-button"
              disabled={running || busy || stopped}
              onClick={() => void step()}
            >
              <SkipForward size={16} />
              {busy && !running ? "计算中…" : "单步更新"}
            </button>
            <button
              className="icon-button"
              aria-label="重置梯度下降"
              onClick={() => {
                reset();
                onInteract();
              }}
            >
              <RotateCcw size={17} />
            </button>
            <span className="step-count">
              STEP <strong>{history.length ? history.length - 1 : 0}</strong>
            </span>
          </div>
          <p
            className={`run-status ${lastUpdate?.diverged ? "warning-text" : ""}`}
            role="status"
          >
            {status}
          </p>
          <div className="result-grid three-results">
            <Result label="当前权重 w" value={format(weight, 5)} />
            <Result
              label="当前损失 L"
              value={loss === undefined ? "—" : format(loss, 5)}
            />
            <Result
              label="当前位置的梯度"
              value={data ? format(data.gradient, 5) : "—"}
            />
          </div>
          <ErrorMessage error={stepError || error} />
        </ExperimentPanel>
        <ExperimentPanel
          title="每一步，都留下真实记录"
          eyebrow="TRAINING HISTORY"
          action={
            <button
              className="text-button"
              onClick={exportHistory}
              disabled={!history.length}
            >
              <Download size={14} />
              导出 CSV
            </button>
          }
        >
          {history.length > 1 ? (
            <Chart
              ariaLabel="每一步参数更新后的真实损失历史"
              series={[
                { points: history.map((s) => ({ x: s.step, y: s.loss })) },
              ]}
              xLabel="更新步数 step"
              yLabel="损失 L"
              markers={
                last ? [{ x: last.step, y: last.loss, label: "最新一步" }] : []
              }
            />
          ) : (
            <div className="empty-history">
              <span>0 → 1</span>
              <p>点击「单步更新」，观察第一次学习。</p>
            </div>
          )}
        </ExperimentPanel>
      </div>
      <aside className="control-panel">
        <ControlsHeading />
        <div className="control-stack">
          <Slider
            label="学习率"
            symbol="η"
            value={rate}
            min={0.01}
            max={0.6}
            step={0.01}
            onChange={(v) => {
              reset();
              setRate(v);
              onInteract();
            }}
          />
          <Slider
            label="初始权重"
            symbol="w₀"
            value={start}
            min={-4}
            max={6}
            step={0.1}
            onChange={(v) => {
              reset(v);
              onInteract();
            }}
          />
        </div>
        <div className="rate-presets">
          <button
            className={rate === 0.1 ? "is-selected" : ""}
            onClick={() => {
              reset();
              setRate(0.1);
              onInteract();
            }}
          >
            0.1 · 稳步下降
          </button>
          <button
            className={rate === 0.5 ? "is-selected" : ""}
            onClick={() => {
              reset();
              setRate(0.5);
              onInteract();
            }}
          >
            0.5 · 来回振荡
          </button>
          <button
            className={rate === 0.55 ? "is-selected" : ""}
            onClick={() => {
              reset();
              setRate(0.55);
              onInteract();
            }}
          >
            0.55 · 逐步发散
          </button>
        </div>
        <div className="formula">w新 = w旧 − η × ∂L/∂w</div>
        {lastUpdate && (
          <div className="update-detail">
            <span className="small-label">最近一次参数更新</span>
            <p>
              {format(lastUpdate.weight_before)} − {format(rate)} × (
              {format(lastUpdate.gradient)})
            </p>
            <strong>= {format(lastUpdate.weight_after, 5)}</strong>
          </div>
        )}
        <p className="panel-note">
          沿用线性模型的 5 个样本，固定 b = 0.5，只训练权重 w。
          <br />
          L(w) = 2(w − 2)²
          <br />
          ∂L/∂w = 4(w − 2)
        </p>
        <div className="insight">
          <span className="small-label">观察重点</span>
          <p>梯度告诉我们上坡的方向。减去梯度，才能让参数朝下坡更新。</p>
        </div>
      </aside>
    </div>
  );
}
