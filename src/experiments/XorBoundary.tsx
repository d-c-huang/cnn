import { useEffect, useRef, useState } from "react";
import type { XorSnapshot } from "../lib/xor";
import { format } from "../lib/api";
import "./xor-visuals.css";

export default function XorBoundary({
  snapshot,
  onSelectSample,
}: {
  snapshot: XorSnapshot;
  onSelectSample: (index: number) => void;
}) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(412);
  useEffect(() => {
    if (!plotRef.current) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(entries[0].contentRect.width),
    );
    observer.observe(plotRef.current);
    return () => observer.disconnect();
  }, []);
  const { grid } = snapshot;
  const left = 44,
    top = 24,
    side = Math.max(100, width - left - 20),
    height = top + side + 54;
  const x = (value: number) =>
    left + ((value - grid.x_min) / (grid.x_max - grid.x_min)) * side;
  const y = (value: number) =>
    top + ((grid.y_max - value) / (grid.y_max - grid.y_min)) * side;
  const cell = side / grid.size;
  function color(probability: number) {
    const p = Math.max(0, Math.min(1, probability));
    const low = [125, 176, 165],
      high = [205, 235, 137];
    return `rgb(${low.map((v, i) => Math.round(v + (high[i] - v) * p)).join(",")})`;
  }
  return (
    <div className="xv-boundary-shell">
      <div className="xv-boundary-plot" ref={plotRef}>
        <svg
          className="xv-boundary-svg"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="XOR 二维输出概率图，横轴输入 x1，纵轴输入 x2；四个训练样本可通过下方按钮选择"
        >
          <rect x={left} y={top} width={side} height={side} fill="#f2f5e9" />
          {grid.probabilities.map((probability, index) => {
            const row = Math.floor(index / grid.size),
              column = index % grid.size;
            return (
              <rect
                key={index}
                x={left + column * cell}
                y={top + (grid.size - 1 - row) * cell}
                width={cell + 0.05}
                height={cell + 0.05}
                fill={color(probability)}
              >
                <title>模型输出 {format(probability, 3)}</title>
              </rect>
            );
          })}
          {[0, 0.5, 1].map((value) => (
            <g key={value}>
              <line
                x1={x(value)}
                x2={x(value)}
                y1={top}
                y2={top + side}
                stroke="#fff"
                strokeOpacity=".3"
                strokeDasharray="3 4"
              />
              <line
                x1={left}
                x2={left + side}
                y1={y(value)}
                y2={y(value)}
                stroke="#fff"
                strokeOpacity=".3"
                strokeDasharray="3 4"
              />
              <text
                x={x(value)}
                y={top + side + 22}
                className="xv-axis-label"
                textAnchor="middle"
              >
                {value}
              </text>
              <text
                x={left - 14}
                y={y(value) + 4}
                className="xv-axis-label"
                textAnchor="end"
              >
                {value}
              </text>
            </g>
          ))}
          {snapshot.samples.map((sample, index) => (
            <g key={index}>
              <circle
                cx={x(sample.inputs[0])}
                cy={y(sample.inputs[1])}
                r={snapshot.selected.index === index ? 12 : 9}
                fill={sample.target ? "#ecf8d7" : "#31594e"}
                stroke={snapshot.selected.index === index ? "#26351c" : "#fff"}
                strokeWidth={snapshot.selected.index === index ? 3 : 2}
              />
              <text
                x={x(sample.inputs[0])}
                y={y(sample.inputs[1]) + (sample.inputs[1] ? 28 : -19)}
                textAnchor="middle"
                className="xv-point-label"
              >
                {sample.inputs.join("")} → {sample.target}
              </text>
            </g>
          ))}
          <text
            x={left + side / 2}
            y={height - 5}
            className="xv-axis-label"
            textAnchor="middle"
          >
            输入 x₁
          </text>
          <text x="12" y="17" className="xv-axis-label">
            输入 x₂
          </text>
        </svg>
        <div
          className="xv-map-actions"
          role="group"
          aria-label="在概率图上选择训练样本"
        >
          {snapshot.samples.map((sample, index) => (
            <button
              key={index}
              type="button"
              className="xv-map-sample"
              style={{
                left: `${(x(sample.inputs[0]) / width) * 100}%`,
                top: `${(y(sample.inputs[1]) / height) * 100}%`,
              }}
              aria-label={`选择样本 ${sample.inputs.join("")}，标签 ${sample.target}，概率 ${format(sample.probability)}`}
              aria-pressed={snapshot.selected.index === index}
              onClick={() => onSelectSample(index)}
            />
          ))}
        </div>
      </div>
      <div className="xv-probability-legend">
        <span>类 0</span>
        <div
          style={{
            background: `linear-gradient(90deg, ${color(0)}, ${color(1)})`,
          }}
        />
        <span>类 1</span>
      </div>
      <div className="xv-boundary-selection" aria-live="polite">
        <span>
          当前样本 {snapshot.selected.inputs.join("")} · 标签{" "}
          {snapshot.selected.target}
        </span>
        <strong>P(类 1) = {format(snapshot.selected.output, 4)}</strong>
      </div>
      <p className="xv-note">
        颜色表示模型输出 P(类 1)，以 0.5 为分类阈值。只训练并评估这 4 个 XOR
        点，训练准确率不代表泛化能力。
      </p>
    </div>
  );
}
