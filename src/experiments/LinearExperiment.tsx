import { MathFormula } from "../components/MathFormula";
import { useMemo, useState } from "react";
import { RotateCcw, WandSparkles } from "lucide-react";
import {
  Chart,
  ControlsHeading,
  ErrorMessage,
  ExperimentPanel,
  Result,
  Slider,
} from "../components/ui";
import { calculationStatus, format, useCalculation } from "../lib/api";
import { linearPreview } from "../lib/foundationPreviews";
import { useLessonTask } from "../lib/learningProgress";

type LinearResponse = {
  points: { x: number; y: number; prediction: number; error: number }[];
  mse: number;
};

export default function LinearExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [weight, setWeight] = useState(0.5);
  const [bias, setBias] = useState(0);
  const preview = useMemo(() => linearPreview(weight, bias), [weight, bias]);
  const { data: displayedData, error, pending } = useCalculation<LinearResponse>("/linear", {
    weight,
    bias,
  }, preview);
  useLessonTask("linear", Boolean(displayedData && displayedData.points.length === 5 && displayedData.mse < 0.001));
  const change = (setter: (n: number) => void, n: number) => {
    setter(n);
    onInteract();
  };
  return (
    <div className="experiment-layout">
      <div className="experiment-main">
        <ExperimentPanel
          title="找到穿过数据的那条线"
          eyebrow="LINEAR REGRESSION"
          action={
            <span className="live-tag" title={calculationStatus(error, pending)}>
              <i />
              实时预览
            </span>
          }
        >
          <div className="plot-legend">
            <span>
              <i className="legend-dot" />
              真实样本
            </span>
            <span>
              <i className="legend-line" />
              模型预测
            </span>
            <span>
              <i className="legend-line dashed" />
              预测误差
            </span>
          </div>
          {displayedData ? (
            <Chart
              ariaLabel="线性模型预测与五个真实样本的对比图"
              xLabel="输入 x"
              yLabel="输出 y"
              xDomain={[-2.5, 2.5]}
              series={[
                {
                  points: [
                    { x: -2.5, y: weight * -2.5 + bias },
                    { x: 2.5, y: weight * 2.5 + bias },
                  ],
                },
                ...displayedData.points.map((p) => ({
                  points: [
                    { x: p.x, y: p.y },
                    { x: p.x, y: p.prediction },
                  ],
                  dashed: true,
                  color: "var(--orange)",
                })),
              ]}
              markers={displayedData.points.map((p) => ({
                x: p.x,
                y: p.y,
                label: `样本 x=${p.x}`,
                color: "var(--teal-soft)",
              }))}
            />
          ) : (
            <div className="chart-placeholder">
              {pending ? "正在计算模型预测…" : "计算服务未连接"}
            </div>
          )}
          <div className="result-grid" aria-live="polite">
            <Result
              label="当前模型"
              value={`ŷ = ${format(weight)}x ${bias < 0 ? "−" : "+"} ${format(Math.abs(bias))}`}
            />
            <Result
              label="均方误差 MSE"
              value={displayedData ? format(displayedData.mse, 4) : "—"}
              hint="越接近 0，拟合越准确"
            />
          </div>
          <ErrorMessage error={error} />
        </ExperimentPanel>
        <ExperimentPanel
          title="每一个预测，都可以拆开来看"
          eyebrow="SAMPLE DETAILS"
        >
          <div className="table-scroll">
            <table className="sample-table">
              <thead>
                <tr>
                  <th>输入 x</th>
                  <th>真实标签 y</th>
                  <th>模型预测 ŷ</th>
                  <th>误差 ŷ − y</th>
                </tr>
              </thead>
              <tbody>
                {displayedData?.points.map((p) => (
                  <tr key={p.x}>
                    <td>{format(p.x)}</td>
                    <td>{format(p.y)}</td>
                    <td>{format(p.prediction)}</td>
                    <td
                      className={
                        Math.abs(p.error) < 0.001 ? "positive-text" : ""
                      }
                    >
                      {format(p.error)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="panel-note">
            5 个教学样本由 y = 2x + 0.5 构造。这里先用简单数据理解参数的作用。
          </p>
        </ExperimentPanel>
      </div>
      <aside className="control-panel">
        <ControlsHeading
          action={
            <button
              className="icon-button"
              aria-label="重置线性模型"
              onClick={() => {
                setWeight(0.5);
                setBias(0);
                onInteract();
              }}
            >
              <RotateCcw size={15} />
            </button>
          }
        />
        <div className="control-stack">
          <Slider
            label="权重"
            symbol="w"
            value={weight}
            min={-3}
            max={4}
            step={0.1}
            onChange={(v) => change(setWeight, v)}
          />
          <Slider
            label="偏置"
            symbol="b"
            value={bias}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) => change(setBias, v)}
          />
        </div>
        <MathFormula className="formula" tex={String.raw`\hat{y} = wx + b`} block />
        <p className="panel-note">
          权重决定直线的倾斜程度，偏置让整条线向上或向下移动。
        </p>
        <button
          className="secondary-button full-width"
          onClick={() => {
            setWeight(2);
            setBias(0.5);
            onInteract();
          }}
        >
          <WandSparkles size={15} />
          看看最佳拟合
        </button>
        <div className="insight">
          <span className="small-label">想一想</span>
          <p>只改变偏置，能让一条斜率不对的直线穿过所有样本吗？</p>
        </div>
      </aside>
    </div>
  );
}
