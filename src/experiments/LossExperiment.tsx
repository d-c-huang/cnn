import { useState } from "react";
import { RotateCcw } from "lucide-react";
import {
  Chart,
  ControlsHeading,
  ErrorMessage,
  ExperimentPanel,
  Result,
  Slider,
} from "../components/ui";
import { format, useCalculation } from "../lib/api";

type LossResponse = {
  loss: number;
  gradient: number;
  curve: { x: number; y: number }[];
};

export default function LossExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [prediction, setPrediction] = useState(-1);
  const [target, setTarget] = useState(2);
  const { data, error, pending } = useCalculation<LossResponse>("/loss", {
    prediction,
    target,
  });
  return (
    <div className="experiment-layout">
      <div className="experiment-main">
        <ExperimentPanel
          title="把“错了多少”变成一个数字"
          eyebrow="SQUARED ERROR"
          action={
            <span className="live-tag">
              <i />
              实时计算
            </span>
          }
        >
          <div className="plot-legend">
            <span>
              <i className="legend-line" />
              平方误差曲线
            </span>
            <span>
              <i className="legend-dot lime" />
              当前预测
            </span>
          </div>
          {data ? (
            <Chart
              ariaLabel="平方误差随预测值变化的曲线，最低点是真实标签"
              series={[{ points: data.curve }]}
              xLabel="预测值 ŷ"
              yLabel="损失 L"
              markers={[
                { x: prediction, y: data.loss, label: "当前预测" },
                {
                  x: target,
                  y: 0,
                  label: "损失最低点",
                  color: "var(--teal-soft)",
                },
              ]}
            />
          ) : (
            <div className="chart-placeholder">
              {pending ? "正在计算损失…" : "计算服务未连接"}
            </div>
          )}
          <div className="result-grid" aria-live="polite">
            <Result
              label="平方误差"
              value={data ? format(data.loss, 4) : "—"}
              hint="L = (ŷ − y)²"
            />
            <Result
              label="当前梯度"
              value={data ? format(data.gradient) : "—"}
              hint="∂L/∂ŷ = 2(ŷ − y)"
            />
          </div>
          <ErrorMessage error={error} />
        </ExperimentPanel>
        <div className="explanation-strip">
          <span className="explanation-index">L</span>
          <div>
            <h3>误差有方向，平方误差始终 ≥ 0</h3>
            <p>
              预测偏大和偏小都会产生损失。曲线最低点对应 ŷ =
              y，此时损失和梯度都等于 0。
            </p>
          </div>
        </div>
      </div>
      <aside className="control-panel">
        <ControlsHeading
          action={
            <button
              className="icon-button"
              aria-label="重置损失实验"
              onClick={() => {
                setPrediction(-1);
                setTarget(2);
                onInteract();
              }}
            >
              <RotateCcw size={15} />
            </button>
          }
        />
        <div className="control-stack">
          <Slider
            label="预测值"
            symbol="ŷ"
            value={prediction}
            min={-5}
            max={5}
            step={0.1}
            onChange={(v) => {
              setPrediction(v);
              onInteract();
            }}
          />
          <Slider
            label="真实标签"
            symbol="y"
            value={target}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) => {
              setTarget(v);
              onInteract();
            }}
          />
        </div>
        <div className="formula">
          L = ({format(prediction)} − {format(target)})²
          <br />
          <strong>= {data ? format(data.loss, 4) : "…"}</strong>
        </div>
        <p className="panel-note">
          这里计算单个样本的平方误差。上一节的
          MSE，是把多个样本的平方误差取平均。
        </p>
        <div className="insight">
          <span className="small-label">试一试</span>
          <p>把预测值移到真实标签两侧相同的距离，损失会有什么变化？梯度呢？</p>
        </div>
      </aside>
    </div>
  );
}
