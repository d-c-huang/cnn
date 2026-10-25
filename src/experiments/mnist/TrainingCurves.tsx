import { useEffect, useRef, useState } from "react";
import type { EpochPoint } from "../../lib/mnist";
import { format } from "../../lib/api";
import "./mnist-visuals.css";

type PlotProps = {
  history: EpochPoint[];
  metric: "loss" | "accuracy";
  selectedEpoch: number | null;
  onSelect: (epoch: number) => void;
};

function EpochPlot({ history, metric, selectedEpoch, onSelect }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(430);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(entries[0].contentRect.width),
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const height = 260,
    left = 51,
    right = 18,
    top = 23,
    bottom = 45;
  const plotWidth = Math.max(15, width - left - right);
  const plotHeight = height - top - bottom;
  const first = history[0].epoch,
    last = history[history.length - 1].epoch;
  const xMin = history.length === 1 ? first - 0.5 : first;
  const xMax = history.length === 1 ? last + 0.5 : last;
  const train = (row: EpochPoint) =>
    metric === "loss" ? row.train_loss : row.train_accuracy * 100;
  const validation = (row: EpochPoint) =>
    metric === "loss" ? row.validation_loss : row.validation_accuracy * 100;
  const yMax =
    metric === "accuracy"
      ? 100
      : Math.max(
          0.001,
          ...history.flatMap((row) => [train(row), validation(row)]),
        ) * 1.13;
  const lossTickDigits = Math.max(1, 1 - Math.floor(Math.log10(yMax)));
  const x = (value: number) =>
    left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const y = (value: number) => top + (1 - value / yMax) * plotHeight;
  const path = (getter: (row: EpochPoint) => number) =>
    history
      .map(
        (row, index) => `${index ? "L" : "M"}${x(row.epoch)},${y(getter(row))}`,
      )
      .join(" ");
  const maxTicks = width < 340 ? 3 : 5;
  const epochs = Array.from(new Set(history.map((row) => row.epoch)));
  const ticks =
    epochs.length <= maxTicks
      ? epochs
      : Array.from(
          { length: maxTicks },
          (_, index) =>
            epochs[Math.round((index * (epochs.length - 1)) / (maxTicks - 1))],
        );
  const selected =
    history.find((row) => row.epoch === selectedEpoch) ||
    history[history.length - 1];
  return (
    <div className="mv-epoch-plot" ref={ref}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${metric === "loss" ? "损失" : "准确率"}训练曲线：横轴为完整训练轮次，实线训练集、虚线验证集，共 ${history.length} 个真实轮次结果`}
      >
        {[0, 1, 2, 3, 4].map((i) => {
          const value = (yMax * i) / 4;
          return (
            <g key={i}>
              <line
                className="mv-grid-line"
                x1={left}
                x2={left + plotWidth}
                y1={y(value)}
                y2={y(value)}
              />
              <text
                x={left - 9}
                y={y(value) + 4}
                textAnchor="end"
                className="mv-axis-text"
              >
                {metric === "accuracy"
                  ? `${Math.round(value)}%`
                  : format(value, lossTickDigits)}
              </text>
            </g>
          );
        })}
        {ticks.map((epoch, index) => (
          <text
            key={epoch}
            x={x(epoch)}
            y={height - 23}
            textAnchor={
              history.length === 1
                ? "middle"
                : index === 0
                  ? "start"
                  : index === ticks.length - 1
                    ? "end"
                    : "middle"
            }
            className="mv-axis-text"
          >
            {epoch}
          </text>
        ))}
        <text x={left} y={12} className="mv-axis-title">
          {metric === "loss" ? "交叉熵损失" : "准确率"}
        </text>
        <text
          x={left + plotWidth / 2}
          y={height - 3}
          textAnchor="middle"
          className="mv-axis-title"
        >
          完整轮次 epoch
        </text>
        <path d={path(train)} className="mv-train-line" />
        <path d={path(validation)} className="mv-validation-line" />
        <line
          className="mv-epoch-guide"
          x1={x(selected.epoch)}
          x2={x(selected.epoch)}
          y1={top}
          y2={top + plotHeight}
        />
        {history.flatMap((row) => [
          <circle
            key={`train-${row.epoch}`}
            cx={x(row.epoch)}
            cy={y(train(row))}
            r={row.epoch === selected.epoch ? 4.5 : 3}
            className="mv-train-point"
          >
            <title>
              第 {row.epoch} 轮训练集：
              {metric === "accuracy"
                ? `${format(train(row), 2)}%`
                : format(train(row), 5)}
            </title>
          </circle>,
          <circle
            key={`val-${row.epoch}`}
            cx={x(row.epoch)}
            cy={y(validation(row))}
            r={row.epoch === selected.epoch ? 4.5 : 3}
            className="mv-validation-point"
          >
            <title>
              第 {row.epoch} 轮验证集：
              {metric === "accuracy"
                ? `${format(validation(row), 2)}%`
                : format(validation(row), 5)}
            </title>
          </circle>,
        ])}
        <rect
          x={left}
          y={top}
          width={plotWidth}
          height={plotHeight}
          fill="transparent"
          onPointerMove={(event) => {
            const bounds =
              event.currentTarget.ownerSVGElement!.getBoundingClientRect();
            const cursor =
              xMin +
              ((((event.clientX - bounds.left) * width) / bounds.width - left) /
                plotWidth) *
                (xMax - xMin);
            const nearest = history.reduce((best, row) =>
              Math.abs(row.epoch - cursor) < Math.abs(best.epoch - cursor)
                ? row
                : best,
            );
            onSelect(nearest.epoch);
          }}
        />
      </svg>
      <div className="mv-curve-values">
        <span>第 {selected.epoch} 轮</span>
        <span>
          训练{" "}
          <strong>
            {metric === "accuracy"
              ? `${format(train(selected), 2)}%`
              : format(train(selected), 5)}
          </strong>
        </span>
        <span>
          验证{" "}
          <strong>
            {metric === "accuracy"
              ? `${format(validation(selected), 2)}%`
              : format(validation(selected), 5)}
          </strong>
        </span>
      </div>
    </div>
  );
}

export default function TrainingCurves({ history }: { history: EpochPoint[] }) {
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);
  const valid = history.filter((row) =>
    [
      row.epoch,
      row.train_loss,
      row.validation_loss,
      row.train_accuracy,
      row.validation_accuracy,
    ].every(Number.isFinite),
  );
  if (!valid.length)
    return (
      <div className="mv-empty-curves">
        <span>epoch 0 → 1</span>
        <p>尚无完整训练轮次。第一轮结束后，将显示真实的训练集和验证集曲线。</p>
      </div>
    );
  const effectiveSelection = valid.some((row) => row.epoch === selectedEpoch)
    ? selectedEpoch
    : null;
  return (
    <div className="mv-training-curves">
      <div className="mv-curve-toolbar">
        <div className="mv-curve-legend">
          <span>
            <i className="mv-train-swatch" />
            训练集
          </span>
          <span>
            <i className="mv-validation-swatch" />
            验证集
          </span>
        </div>
        <label>
          查看轮次
          <select
            value={effectiveSelection ?? "latest"}
            onChange={(e) =>
              setSelectedEpoch(
                e.target.value === "latest" ? null : Number(e.target.value),
              )
            }
          >
            <option value="latest">最近完整轮次</option>
            {valid.map((row) => (
              <option key={row.epoch} value={row.epoch}>
                第 {row.epoch} 轮
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="mv-curve-grid">
        <EpochPlot
          history={valid}
          metric="loss"
          selectedEpoch={effectiveSelection}
          onSelect={setSelectedEpoch}
        />
        <EpochPlot
          history={valid}
          metric="accuracy"
          selectedEpoch={effectiveSelection}
          onSelect={setSelectedEpoch}
        />
      </div>
      <p className="mv-note">
        每个点对应一个完整
        epoch。验证集仅用于选择模型；独立测试集不参与训练曲线。
      </p>
    </div>
  );
}
