import { useId } from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { format } from "../lib/api";
import "./graph.css";

export type GraphRequest = {
  inputs: [number, number];
  weights: [number, number];
  bias: number;
  target: number;
  learning_rate: number;
};

export type GraphNode = {
  id: string;
  label: string;
  op: string;
  value: number;
  gradient: number;
  expression: string;
  trainable: boolean;
};

export type GraphResponse = {
  nodes: GraphNode[];
  edges: {
    source: string;
    target: string;
    local_derivative: number;
    contribution: number;
  }[];
  forward_order: string[];
  backward_order: string[];
  output: number;
  loss: number;
  output_after: number;
  loss_after: number;
  updates: { name: string; before: number; gradient: number; after: number }[];
  next: GraphRequest;
};

export const initialGraph: GraphRequest = {
  inputs: [0.6, 0.9],
  weights: [0.8, -0.4],
  bias: 0.2,
  target: 0.8,
  learning_rate: 0.1,
};

const positions: Record<string, [number, number]> = {
  x1: [65, 65],
  w1: [65, 155],
  p1: [220, 110],
  x2: [65, 335],
  w2: [65, 425],
  p2: [220, 380],
  s: [375, 245],
  b: [375, 425],
  z: [530, 245],
  a: [685, 245],
  target: [685, 425],
  e: [840, 245],
  loss: [840, 75],
};
const operationLabels: Record<string, string> = {
  multiply: "乘法",
  add: "求和",
  tanh: "tanh 激活",
  target: "真实标签",
  subtract: "相减",
  square: "平方损失",
};

function endpoints(a: [number, number], b: [number, number]) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const scale = Math.min(
    dx ? 57 / Math.abs(dx) : Infinity,
    dy ? 46 / Math.abs(dy) : Infinity,
  );
  return [a[0] + dx * scale, a[1] + dy * scale];
}

export default function ComputationGraph({
  data,
  selected,
  onSelect,
  mode = "forward",
  activeNode,
  revealedValues,
  revealedGradients,
}: {
  data: GraphResponse;
  selected: string;
  onSelect: (id: string) => void;
  mode?: "forward" | "backward";
  activeNode?: string | null;
  revealedValues?: string[];
  revealedGradients?: string[];
}) {
  const markerId = useId().replace(/:/g, "");
  const valueVisible = (id: string) =>
    !revealedValues || revealedValues.includes(id);
  const gradientVisible = (id: string) =>
    !revealedGradients || revealedGradients.includes(id);
  const nodeContent = (node: GraphNode) => (
    <>
      <span className="gz-node-label">
        {node.label}
        {node.trainable && <i title="可训练参数">θ</i>}
      </span>
      <strong>
        {mode === "backward"
          ? gradientVisible(node.id)
            ? format(node.gradient, 4)
            : "…"
          : valueVisible(node.id)
            ? format(node.value, 4)
            : "…"}
      </strong>
      <small>
        {mode === "backward"
          ? "∂L / ∂节点"
          : node.op === "input" || node.op === "parameter"
            ? "给定数值"
            : operationLabels[node.op] || node.op}
      </small>
    </>
  );
  function nodeClass(node: GraphNode) {
    return `gz-node ${selected === node.id ? "gz-selected" : ""} ${activeNode === node.id ? "gz-active" : ""} ${node.trainable ? "gz-parameter" : ""} ${(mode === "forward" && !valueVisible(node.id)) || (mode === "backward" && !gradientVisible(node.id)) ? "gz-unrevealed" : ""}`;
  }
  return (
    <div className="gz-graph">
      <div className="gz-direction">
        <span>
          {mode === "forward" ? (
            <ArrowRight size={15} />
          ) : (
            <ArrowLeft size={15} />
          )}
          {mode === "forward"
            ? "前向：沿依赖关系计算数值"
            : "反向：从损失向输入传递梯度"}
        </span>
        <small>θ 可训练参数 · 点击节点查看细节</small>
      </div>
      <div className="gz-desktop">
        <svg
          viewBox="0 0 1000 490"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <marker
              id={markerId}
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
            >
              <path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" />
            </marker>
          </defs>
          {data.edges.map((edge) => {
            const start =
              positions[mode === "backward" ? edge.target : edge.source];
            const end =
              positions[mode === "backward" ? edge.source : edge.target];
            if (!start || !end) return null;
            const [x1, y1] = endpoints(start, end),
              [x2, y2] = endpoints(end, start);
            const active =
              mode === "backward"
                ? edge.source === activeNode
                : edge.target === activeNode;
            const selectedEdge =
              edge.source === selected || edge.target === selected;
            return (
              <path
                key={`${edge.source}-${edge.target}`}
                d={`M${x1},${y1} L${x2},${y2}`}
                markerEnd={`url(#${markerId})`}
                className={`gz-edge ${active ? "gz-edge-active" : selectedEdge ? "gz-edge-selected" : ""}`}
              />
            );
          })}
        </svg>
        {data.nodes.map((node) => {
          const position = positions[node.id];
          return position ? (
            <button
              type="button"
              key={node.id}
              className={nodeClass(node)}
              style={{
                left: `${position[0] / 10}%`,
                top: `${position[1] / 4.9}%`,
              }}
              aria-pressed={selected === node.id}
              aria-label={`节点 ${node.label}，${mode === "backward" ? "梯度" : "数值"} ${mode === "backward" ? (gradientVisible(node.id) ? format(node.gradient, 4) : "尚未传播") : valueVisible(node.id) ? format(node.value, 4) : "尚未计算"}`}
              onClick={() => onSelect(node.id)}
            >
              {nodeContent(node)}
            </button>
          ) : null;
        })}
      </div>
      <div
        className="gz-mobile"
        role="group"
        aria-label="按计算依赖排列的可选节点"
      >
        {data.forward_order
          .map((id) => data.nodes.find((node) => node.id === id))
          .filter((node): node is GraphNode => Boolean(node))
          .map((node, index) => (
            <button
              key={node.id}
              type="button"
              className={nodeClass(node)}
              aria-pressed={selected === node.id}
              onClick={() => onSelect(node.id)}
            >
              <span className="gz-node-index">
                {String(index + 1).padStart(2, "0")}
              </span>
              {nodeContent(node)}
            </button>
          ))}
      </div>
    </div>
  );
}

export function GraphNodeDetails({
  data,
  selected,
  revealedValues,
  revealedGradients,
}: {
  data: GraphResponse;
  selected: string;
  revealedValues?: string[];
  revealedGradients?: string[];
}) {
  const node = data.nodes.find((item) => item.id === selected);
  if (!node) return null;
  const valueVisible = !revealedValues || revealedValues.includes(selected);
  const gradientVisible =
    !revealedGradients || revealedGradients.includes(selected);
  const edges = data.edges.filter((edge) => edge.source === selected);
  const label = (id: string) =>
    data.nodes.find((item) => item.id === id)?.label || id;
  return (
    <div className="gz-node-detail" aria-live="polite">
      <div className="gz-detail-heading">
        <span className="small-label">选中节点 · {node.label}</span>
        <span>{node.trainable ? "可训练参数" : "计算图节点"}</span>
      </div>
      <div className="gz-expression">{node.expression}</div>
      <div className="gz-detail-values">
        <span>
          前向值{" "}
          <strong>{valueVisible ? format(node.value, 5) : "尚未计算"}</strong>
        </span>
        <span>
          梯度 ∂L/∂{node.label}{" "}
          <strong>
            {gradientVisible ? format(node.gradient, 5) : "尚未传播"}
          </strong>
        </span>
      </div>
      {gradientVisible ? (
        <div className="gz-chain-list">
          {edges.length ? (
            <>
              <p>沿每条出边收集梯度贡献，再相加得到该节点的梯度：</p>
              {edges.map((edge) => {
                const upstream =
                  data.nodes.find((item) => item.id === edge.target)
                    ?.gradient ?? 0;
                return (
                  <div key={`${edge.source}-${edge.target}`}>
                    <span>
                      {label(edge.target)} → {node.label}
                    </span>
                    <code>
                      {format(upstream, 4)} × {format(edge.local_derivative, 4)}{" "}
                      = {format(edge.contribution, 5)}
                    </code>
                    <small>
                      上游梯度 × 局部导数 ∂{label(edge.target)}/∂{node.label}
                    </small>
                  </div>
                );
              })}
            </>
          ) : (
            <p>损失对自身的导数为 1，这是反向传播的起点。</p>
          )}
        </div>
      ) : (
        <p className="gz-wait-note">
          先完成前向计算，再从损失节点开始反向传播。
        </p>
      )}
    </div>
  );
}
