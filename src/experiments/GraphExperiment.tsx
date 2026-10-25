import { useState } from "react";
import { RotateCcw } from "lucide-react";
import {
  ControlsHeading,
  ErrorMessage,
  ExperimentPanel,
  Result,
  Slider,
} from "../components/ui";
import { format, useCalculation } from "../lib/api";
import ComputationGraph, {
  GraphNodeDetails,
  initialGraph,
} from "./ComputationGraph";
import type { GraphRequest, GraphResponse } from "./ComputationGraph";

export default function GraphExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [parameters, setParameters] = useState<GraphRequest>(initialGraph);
  const [selected, setSelected] = useState("a");
  const [mode, setMode] = useState<"forward" | "backward">("forward");
  const { data, error, pending } = useCalculation<GraphResponse>(
    "/graph",
    parameters,
  );
  function change(next: GraphRequest) {
    setParameters(next);
    onInteract();
  }
  return (
    <div className="experiment-layout gz-experiment">
      <div className="experiment-main">
        <ExperimentPanel
          title="把一次预测，展开成一张计算图"
          eyebrow="A GRAPH OF OPERATIONS"
          action={
            <div
              className="segmented gz-mode"
              role="group"
              aria-label="计算图显示模式"
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
                aria-pressed={mode === "backward"}
                onClick={() => {
                  setMode("backward");
                  onInteract();
                }}
              >
                反向梯度
              </button>
            </div>
          }
        >
          {data ? (
            <>
              <ComputationGraph
                data={data}
                mode={mode}
                selected={selected}
                onSelect={(id) => {
                  setSelected(id);
                  onInteract();
                }}
              />
              <div className="result-grid">
                <Result
                  label="神经元输出 a"
                  value={format(data.output, 5)}
                  hint="a = tanh(x₁w₁ + x₂w₂ + b)"
                />
                <Result
                  label="当前损失 L"
                  value={format(data.loss, 5)}
                  hint="L = ½(a − y)²"
                />
              </div>
              <GraphNodeDetails data={data} selected={selected} />
            </>
          ) : (
            <div className="gz-loading">
              {pending ? "正在展开当前参数的计算图…" : "计算服务未连接"}
            </div>
          )}
          <ErrorMessage error={error} />
        </ExperimentPanel>
        <div className="explanation-strip">
          <span className="explanation-index">∂</span>
          <div>
            <h3>图中的每一步，都是一个可求导的运算</h3>
            <p>
              连线表示数值依赖。前向沿箭头求值，反向把上游梯度乘以这一步的局部导数。多条路径汇合时，要把各条路径的梯度贡献相加。
            </p>
            <p>
              本节使用 L = ½(a − y)²，系数 ½ 让导数简化为 a −
              y；第一期使用的平方误差没有这个系数。
            </p>
          </div>
        </div>
      </div>
      <aside className="control-panel">
        <ControlsHeading
          action={
            <button
              className="icon-button"
              aria-label="重置计算图"
              onClick={() => {
                setParameters(initialGraph);
                setSelected("a");
                setMode("forward");
                onInteract();
              }}
            >
              <RotateCcw size={16} />
            </button>
          }
        />
        <div className="control-stack">
          <Slider
            label="输入 1"
            symbol="x₁"
            value={parameters.inputs[0]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) =>
              change({ ...parameters, inputs: [v, parameters.inputs[1]] })
            }
          />
          <Slider
            label="输入 2"
            symbol="x₂"
            value={parameters.inputs[1]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) =>
              change({ ...parameters, inputs: [parameters.inputs[0], v] })
            }
          />
          <Slider
            label="权重 1"
            symbol="w₁"
            value={parameters.weights[0]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) =>
              change({ ...parameters, weights: [v, parameters.weights[1]] })
            }
          />
          <Slider
            label="权重 2"
            symbol="w₂"
            value={parameters.weights[1]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) =>
              change({ ...parameters, weights: [parameters.weights[0], v] })
            }
          />
          <Slider
            label="偏置"
            symbol="b"
            value={parameters.bias}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) => change({ ...parameters, bias: v })}
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
        <div className="insight">
          选择权重
          w₁，再切换到「反向梯度」，沿着损失到权重的路径，查看链式法则如何把每一步连接起来。
        </div>
        <p className="panel-note">
          输入与标签在这次训练中保持不变。只有带 θ 标记的 w₁、w₂、b
          会由优化器更新。
        </p>
      </aside>
    </div>
  );
}
