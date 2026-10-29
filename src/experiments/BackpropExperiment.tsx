import { MathFormula } from "../components/MathFormula";
import { useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, RotateCcw } from "lucide-react";
import {
  ControlsHeading,
  ErrorMessage,
  ExperimentPanel,
  Result,
  Slider,
} from "../components/ui";
import { calculationStatus, format, useCalculation } from "../lib/api";
import { graphPreview } from "../lib/foundationPreviews";
import { useLessonTask } from "../lib/learningProgress";
import ComputationGraph, {
  GraphNodeDetails,
  initialGraph,
} from "./ComputationGraph";
import type { GraphRequest, GraphResponse } from "./ComputationGraph";

const operations = ["p1", "p2", "s", "z", "a", "e", "loss"];

export default function BackpropExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [parameters, setParameters] = useState<GraphRequest>(initialGraph);
  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState("p1");
  const [applied, setApplied] = useState(0);
  const [lastApplied, setLastApplied] = useState<{
    before: number;
    after: number;
  } | null>(null);
  const appliedKey = useRef<string | null>(null);
  const preview = useMemo(() => graphPreview(parameters), [parameters]);
  const { data, confirmedData, error, pending } = useCalculation<GraphResponse>(
    "/graph",
    parameters,
    preview,
  );
  useLessonTask("backprop", applied > 0);
  const backwardOrder = data?.backward_order || [];
  const updateCursor = operations.length + backwardOrder.length + 1;
  const phase =
    cursor <= operations.length
      ? "forward"
      : cursor < updateCursor
        ? "backward"
        : "update";
  const currentNode =
    phase === "forward"
      ? operations[cursor - 1] || null
      : phase === "backward"
        ? backwardOrder[cursor - operations.length - 1] || null
        : null;
  const sources =
    data?.nodes
      .filter((node) => !operations.includes(node.id))
      .map((node) => node.id) || [];
  const revealedValues =
    phase === "forward"
      ? [...sources, ...operations.slice(0, cursor)]
      : data?.forward_order || [];
  const revealedGradients =
    phase === "forward"
      ? []
      : phase === "update"
        ? backwardOrder
        : backwardOrder.slice(0, cursor - operations.length);
  const current = data?.nodes.find((node) => node.id === currentNode);

  function move(next: number) {
    const clamped = Math.min(Math.max(0, next), updateCursor);
    setCursor(clamped);
    const node =
      clamped <= operations.length
        ? operations[clamped - 1]
        : clamped < updateCursor
          ? backwardOrder[clamped - operations.length - 1]
          : null;
    if (node) setSelected(node);
    onInteract();
  }
  function change(next: GraphRequest) {
    appliedKey.current = null;
    setParameters(next);
    setCursor(0);
    setSelected("p1");
    setLastApplied(null);
    onInteract();
  }
  function reset() {
    appliedKey.current = null;
    setParameters(initialGraph);
    setCursor(0);
    setSelected("p1");
    setApplied(0);
    setLastApplied(null);
    onInteract();
  }
  function applyUpdate() {
    const key = JSON.stringify(parameters);
    if (!confirmedData || pending || phase !== "update" || appliedKey.current === key) return;
    appliedKey.current = key;
    setLastApplied({ before: confirmedData.loss, after: confirmedData.loss_after });
    setParameters(confirmedData.next);
    setApplied((previous) => previous + 1);
    setCursor(0);
    setSelected("p1");
    onInteract();
  }

  return (
    <div className="experiment-layout gz-experiment">
      <div className="experiment-main">
        <ExperimentPanel
          title="跟着一次训练，走完前向与反向"
          eyebrow="ONE COMPLETE TRAINING STEP"
          action={
            <span className="live-tag" title={calculationStatus(error, pending)}>
              <i />
              解析梯度 · 实时预览
            </span>
          }
        >
          <div className="gz-stages" role="group" aria-label="训练过程阶段">
            <button
              aria-pressed={phase === "forward"}
              disabled={!data}
              onClick={() => move(1)}
            >
              <span>01</span>前向计算
            </button>
            <button
              aria-pressed={phase === "backward"}
              disabled={!data}
              onClick={() => move(operations.length + 1)}
            >
              <span>02</span>反向传播
            </button>
            <button
              aria-pressed={phase === "update"}
              disabled={!data}
              onClick={() => move(updateCursor)}
            >
              <span>03</span>参数更新
            </button>
          </div>
          {data ? (
            <>
              <ComputationGraph
                data={data}
                selected={selected}
                onSelect={(id) => {
                  setSelected(id);
                  onInteract();
                }}
                mode={phase === "forward" ? "forward" : "backward"}
                activeNode={currentNode}
                revealedValues={revealedValues}
                revealedGradients={revealedGradients}
              />
              <div className="gz-player">
                <button
                  className="secondary-button"
                  disabled={cursor === 0}
                  onClick={() => move(cursor - 1)}
                >
                  <ArrowLeft size={15} />
                  上一步
                </button>
                <span>
                  {phase === "forward"
                    ? `前向 ${cursor} / ${operations.length}`
                    : phase === "backward"
                      ? `反向 ${cursor - operations.length} / ${backwardOrder.length}`
                      : "等待应用更新"}
                </span>
                <button
                  className="primary-button"
                  disabled={phase === "update"}
                  onClick={() => move(cursor + 1)}
                >
                  下一步
                  <ArrowRight size={15} />
                </button>
              </div>
              <div className="gz-step-explanation" role="status">
                <span className="small-label">
                  {phase === "forward"
                    ? "前向计算"
                    : phase === "backward"
                      ? "链式法则"
                      : "优化器更新"}
                </span>
                <p>
                  {phase === "forward"
                    ? current
                      ? `${current.expression}，得到 ${format(current.value, 5)}。`
                      : "输入、标签和参数已给定。点击「下一步」，从第一路乘法开始计算。"
                    : phase === "backward"
                      ? current?.id === "loss"
                        ? "损失对自身的导数为 1。把这个梯度作为起点，沿计算依赖反向传播。"
                        : `${current?.label || "当前节点"} 的梯度来自后续运算：上游梯度 × 局部导数，存在多条路径时把贡献相加。`
                      : "反向传播已经算出梯度。下面预览 SGD 更新；点击「应用参数更新」才会让新参数进入下一次计算。"}
                </p>
              </div>
              <GraphNodeDetails
                data={data}
                selected={selected}
                revealedValues={revealedValues}
                revealedGradients={revealedGradients}
              />
              {phase === "update" && (
                <div className="gz-update-section">
                  <h3>SGD：θ新 = θ旧 − η × 梯度</h3>
                  <div className="table-scroll">
                    <table className="sample-table gz-update-table">
                      <thead>
                        <tr>
                          <th>参数</th>
                          <th>更新前</th>
                          <th>梯度</th>
                          <th>更新后</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.updates.map((row) => (
                          <tr key={row.name}>
                            <td>{row.name}</td>
                            <td>{format(row.before, 5)}</td>
                            <td>{format(row.gradient, 5)}</td>
                            <td>{format(row.after, 5)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="result-grid">
                    <Result label="更新前损失" value={format(data.loss, 6)} />
                    <Result
                      label="更新后损失（预览）"
                      value={format(data.loss_after, 6)}
                    />
                  </div>
                  <button
                    className="primary-button gz-apply-button"
                    onClick={applyUpdate}
                    disabled={pending || !confirmedData}
                  >
                    <Check size={16} />
                    应用参数更新
                  </button>
                  <p className="panel-note">只有后端确认当前参数的梯度后，才可应用更新。</p>
                </div>
              )}
            </>
          ) : (
            <div className="gz-loading">
              {pending ? "正在计算前向值与解析梯度…" : "计算服务未连接"}
            </div>
          )}
          <ErrorMessage error={error} />
        </ExperimentPanel>
        {lastApplied && (
          <div className="gz-applied-message" role="status">
            <Check size={17} />
            <p>
              已应用第 {applied} 次参数更新：损失{" "}
              {format(lastApplied.before, 6)} → {format(lastApplied.after, 6)}
              。图中的参数已替换为更新后的值，可以重新播放。
            </p>
          </div>
        )}
        <div className="explanation-strip">
          <span className="explanation-index">∇</span>
          <div>
            <h3>反向传播计算梯度，优化器更新参数</h3>
            <p>
              前向值和梯度都属于更新前的同一份参数。应用更新后，旧梯度不再用于新参数；下一次训练要重新执行前向和反向计算。
            </p>
            <p>
              本节 L = ½(a − y)²；系数 ½ 让 ∂L/∂a = a −
              y，和第一期没有系数的平方误差相差一倍。
            </p>
          </div>
        </div>
      </div>
      <aside className="control-panel">
        <ControlsHeading
          action={
            <button
              className="icon-button"
              aria-label="重置反向传播实验"
              onClick={reset}
            >
              <RotateCcw size={16} />
            </button>
          }
        />
        <div className="control-stack">
          <Slider
            label="学习率"
            symbol="η"
            value={parameters.learning_rate}
            min={0.01}
            max={1}
            step={0.01}
            onChange={(v) => change({ ...parameters, learning_rate: v })}
          />
          <Slider
            label="真实标签"
            symbol="y"
            value={parameters.target}
            min={-1}
            max={1}
            step={0.1}
            onChange={(v) => change({ ...parameters, target: v })}
          />
        </div>
        <div className="gz-current-parameters">
          <span className="small-label">当前参与计算的参数</span>
          <dl>
            <div>
              <dt>w₁</dt>
              <dd>{format(parameters.weights[0], 5)}</dd>
            </div>
            <div>
              <dt>w₂</dt>
              <dd>{format(parameters.weights[1], 5)}</dd>
            </div>
            <div>
              <dt>b</dt>
              <dd>{format(parameters.bias, 5)}</dd>
            </div>
          </dl>
          <span className="small-label">已应用更新：{applied} 次</span>
        </div>
        <MathFormula className="formula" tex={String.raw`\begin{aligned}\frac{\partial L}{\partial w_1} &= \frac{\partial L}{\partial a} \\ &\quad\cdot\frac{\partial a}{\partial z}\cdot x_1\end{aligned}`} block />
        <p className="panel-note">
          固定输入 x₁ = 0.6、x₂ =
          0.9。偏导数是在其他量不变时，一个量对损失的影响。
        </p>
        <div className="insight">
          先看 a 节点的梯度，再看 z 和 w₁。注意 tanh 的局部导数 1 − a²
          如何改变梯度的大小。
        </div>
      </aside>
    </div>
  );
}
