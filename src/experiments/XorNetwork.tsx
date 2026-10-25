import { useId } from "react";
import { ArrowRight } from "lucide-react";
import type { XorSnapshot } from "../lib/xor";
import { flattenParameters } from "../lib/xor";
import { format } from "../lib/api";
import "./xor-visuals.css";

type Props = {
  snapshot: XorSnapshot;
  mode: "forward" | "gradient";
  selectedParameter: string | null;
  onSelectParameter: (name: string) => void;
};

export default function XorNetwork({
  snapshot,
  mode,
  selectedParameter,
  onSelectParameter,
}: Props) {
  const markerId = useId().replace(/:/g, "");
  const values = flattenParameters(snapshot.parameters);
  const sampleGradients = flattenParameters(snapshot.sample_gradients);
  const weights = new Map(values.map((entry) => [entry.name, entry.value]));
  const gradients = new Map(
    sampleGradients.map((entry) => [entry.name, entry.value]),
  );
  const activeValue = selectedParameter
    ? weights.get(selectedParameter)
    : undefined;
  const activeGradient = selectedParameter
    ? gradients.get(selectedParameter)
    : undefined;
  const inputY = [128, 252];
  const hiddenY = [54, 148, 242, 336];
  const edges = [
    ...snapshot.parameters.w1.flatMap((row, i) =>
      row.map((weight, j) => ({
        name: `w1[${i},${j}]`,
        weight,
        from: [100, inputY[i]],
        to: [322, hiddenY[j]],
      })),
    ),
    ...snapshot.parameters.w2.map((weight, j) => ({
      name: `w2[${j}]`,
      weight,
      from: [398, hiddenY[j]],
      to: [612, 195],
    })),
  ];
  return (
    <div className="xv-network-shell">
      <div className="xv-network-caption">
        <span>
          样本 {snapshot.selected.inputs.join("")} → 标签{" "}
          {snapshot.selected.target}
        </span>
        <span>
          {mode === "forward"
            ? "节点：前向激活值 · 边：权重"
            : "节点：前向激活值 · 边：单样本参数梯度"}
        </span>
      </div>
      <div className="xv-network-desktop">
        <svg
          className="xv-network-svg"
          viewBox="0 0 720 440"
          role="img"
          aria-label={`XOR 两个输入、四个隐藏单元、一个输出的神经网络，当前输出 ${format(snapshot.selected.output)}`}
        >
          <defs>
            <marker
              id={markerId}
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0 0L7 3.5L0 7Z" fill="#8b9b78" />
            </marker>
          </defs>
          {edges.map((edge) => {
            const value =
              mode === "forward"
                ? edge.weight
                : (gradients.get(edge.name) ?? 0);
            const active = selectedParameter === edge.name;
            return (
              <g key={edge.name}>
                {active && (
                  <line
                    x1={edge.from[0]}
                    y1={edge.from[1]}
                    x2={edge.to[0]}
                    y2={edge.to[1]}
                    stroke="#c9ef6b"
                    strokeWidth="9"
                    opacity=".65"
                  />
                )}
                <line
                  x1={edge.from[0]}
                  y1={edge.from[1]}
                  x2={edge.to[0]}
                  y2={edge.to[1]}
                  stroke={value < 0 ? "#80a79b" : "#4e5f41"}
                  strokeWidth={
                    active
                      ? 3
                      : Math.min(
                          4,
                          1 + Math.abs(value) * (mode === "gradient" ? 7 : 0.6),
                        )
                  }
                  strokeDasharray={value < 0 ? "6 4" : undefined}
                  markerEnd={`url(#${markerId})`}
                  opacity={active ? 1 : 0.54}
                >
                  <title>
                    {edge.name}，{mode === "forward" ? "权重" : "单样本梯度"}{" "}
                    {format(value, 5)}
                  </title>
                </line>
              </g>
            );
          })}
          {snapshot.selected.inputs.map((value, i) => (
            <g key={`input-${i}`}>
              <circle
                cx="64"
                cy={inputY[i]}
                r="35"
                className="xv-input-circle"
              />
              <text
                x="64"
                y={inputY[i] - 7}
                className="xv-node-label"
                textAnchor="middle"
              >
                x{i + 1}
              </text>
              <text
                x="64"
                y={inputY[i] + 15}
                className="xv-node-value"
                textAnchor="middle"
              >
                {format(value)}
              </text>
            </g>
          ))}
          {snapshot.selected.a1.map((value, j) => (
            <g key={`hidden-${j}`}>
              <circle
                cx="360"
                cy={hiddenY[j]}
                r="37"
                className={`xv-hidden-circle ${selectedParameter === `b1[${j}]` ? "xv-selected-bias" : ""}`}
              />
              <text
                x="360"
                y={hiddenY[j] - 8}
                className="xv-node-label"
                textAnchor="middle"
              >
                h{j + 1} · tanh
              </text>
              <text
                x="360"
                y={hiddenY[j] + 14}
                className="xv-node-value"
                textAnchor="middle"
              >
                {format(value)}
              </text>
              <text
                x="360"
                y={hiddenY[j] + 54}
                className="xv-bias-label"
                textAnchor="middle"
              >
                b₁[{j}] = {format(snapshot.parameters.b1[j], 2)}
              </text>
            </g>
          ))}
          <g>
            <circle
              cx="654"
              cy="195"
              r="41"
              className={`xv-output-circle ${selectedParameter === "b2" ? "xv-selected-bias" : ""}`}
            />
            <text x="654" y="185" className="xv-node-label" textAnchor="middle">
              σ(z₂)
            </text>
            <text x="654" y="210" className="xv-node-value" textAnchor="middle">
              {format(snapshot.selected.output)}
            </text>
            <text x="654" y="259" className="xv-bias-label" textAnchor="middle">
              b₂ = {format(snapshot.parameters.b2, 2)}
            </text>
          </g>
          <text x="64" y="430" className="xv-layer-label" textAnchor="middle">
            输入层 · 2
          </text>
          <text x="360" y="430" className="xv-layer-label" textAnchor="middle">
            隐藏层 · 4
          </text>
          <text x="654" y="430" className="xv-layer-label" textAnchor="middle">
            输出层 · 1
          </text>
        </svg>
      </div>
      <div className="xv-network-compact" aria-label="神经网络各层的前向值">
        <div className="xv-compact-head">
          <span>输入层 2</span>
          <ArrowRight size={14} />
          <span>tanh 隐藏层 4</span>
          <ArrowRight size={14} />
          <span>输出层 1</span>
        </div>
        <div className="xv-compact-columns">
          <div>
            {snapshot.selected.inputs.map((value, i) => (
              <div className="xv-compact-node" key={i}>
                <small>x{i + 1}</small>
                <strong>{format(value)}</strong>
              </div>
            ))}
          </div>
          <div>
            {snapshot.selected.a1.map((value, i) => (
              <div className="xv-compact-node" key={i}>
                <small>h{i + 1}</small>
                <strong>{format(value)}</strong>
                <small>b = {format(snapshot.parameters.b1[i], 2)}</small>
              </div>
            ))}
          </div>
          <div>
            <div className="xv-compact-node xv-compact-output">
              <small>σ(z₂)</small>
              <strong>{format(snapshot.selected.output)}</strong>
              <small>b = {format(snapshot.parameters.b2, 2)}</small>
            </div>
          </div>
        </div>
      </div>
      <div className="xv-selected-parameter" aria-live="polite">
        {selectedParameter &&
        activeValue !== undefined &&
        activeGradient !== undefined ? (
          <>
            <strong>{selectedParameter}</strong>
            <span>
              当前权重 / 偏置 <b>{format(activeValue, 5)}</b>
            </span>
            <span>
              当前样本梯度 <b>{format(activeGradient, 5)}</b>
            </span>
          </>
        ) : (
          <span>选择下方一个参数，在网络中查看它的位置。</span>
        )}
      </div>
      <div
        className="xv-parameter-picker"
        role="group"
        aria-label="选择网络参数"
      >
        {values.map((entry) => (
          <button
            type="button"
            key={entry.name}
            aria-pressed={selectedParameter === entry.name}
            onClick={() => onSelectParameter(entry.name)}
          >
            {entry.name}
          </button>
        ))}
      </div>
      <p className="xv-note">
        实线表示正值，虚线表示负值。
        {mode === "gradient"
          ? "这里显示选中样本的参数梯度；训练更新使用四个样本梯度的平均值。"
          : "连线粗细表示权重绝对值，节点中的数字是选中样本的前向激活值。"}
      </p>
    </div>
  );
}
