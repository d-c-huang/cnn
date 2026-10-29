import { MathFormula } from "../components/MathFormula";
import { useMemo, useState } from "react";
import { ArrowDown, ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import { calculationStatus, format, useCalculation } from "../lib/api";
import { useLessonTask } from "../lib/learningProgress";
import { neuronActivationCurve, neuronOutputCurve, neuronValue } from "../lib/neuron";
import type { Activation, Pair } from "../lib/neuron";
import {
  Slider,
  ExperimentPanel,
  Result,
  ErrorMessage,
  Chart,
} from "../components/ui";
import "./neuron.css";

interface NeuronResponse {
  products: [number, number];
  z: number;
  output: number;
  curve: Array<{ x: number; y: number }>;
}

interface ExperimentProps {
  onInteract: () => void;
}

const ACTIVATIONS: Activation[] = ["relu", "sigmoid", "tanh", "linear"];
const ACTIVATION_LABELS: Record<Activation, string> = {
  relu: "ReLU",
  sigmoid: "Sigmoid",
  tanh: "Tanh",
  linear: "线性",
};
const ACTIVATION_FORMULAS: Record<Activation, string> = {
  relu: String.raw`a = \max(0,z)`,
  sigmoid: String.raw`a = \frac{1}{1+e^{-z}}`,
  tanh: String.raw`a = \tanh(z)`,
  linear: "a = z",
};

function display(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(3);
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : "−"} ${Math.abs(value).toFixed(1)}`;
}

export default function NeuronExperiment({ onInteract }: ExperimentProps) {
  const [inputs, setInputs] = useState<[number, number]>([0.6, 0.9]);
  const [weights, setWeights] = useState<[number, number]>([0.8, -0.4]);
  const [bias, setBias] = useState(0.2);
  const [activation, setActivation] = useState<Activation>("relu");
  const [curveAxis, setCurveAxis] = useState<0 | 1>(0);
  const [reference, setReference] = useState<{ weights: Pair; bias: number }>({ weights: [0.8, -0.4], bias: 0.2 });
  const payload = useMemo(
    () => ({ inputs, weights, bias, activation }),
    [inputs, weights, bias, activation],
  );
  const preview = useMemo(() => {
    const value = neuronValue(inputs, weights, bias, activation);
    return { ...value, curve: neuronActivationCurve(value.z, activation) };
  }, [inputs, weights, bias, activation]);
  const { data, error, pending } = useCalculation<NeuronResponse>(
    "/neuron",
    payload,
    preview,
  );
  const current = data ?? preview;
  const activationCurve = current.curve;
  useLessonTask("neuron", activation === "sigmoid" && weights[0] < 0 && curveAxis === 0);
  const outputCurve = useMemo(() => neuronOutputCurve(inputs, weights, bias, activation, curveAxis), [inputs, weights, bias, activation, curveAxis]);
  const referenceCurve = useMemo(() => neuronOutputCurve(inputs, reference.weights, reference.bias, activation, curveAxis), [inputs, reference, activation, curveAxis]);
  const fixedAxis = curveAxis === 0 ? 1 : 0;

  function changeInput(index: 0 | 1, value: number) {
    setInputs((current) =>
      index === 0 ? [value, current[1]] : [current[0], value],
    );
    onInteract();
  }

  function changeWeight(index: 0 | 1, value: number) {
    setWeights((current) =>
      index === 0 ? [value, current[1]] : [current[0], value],
    );
    onInteract();
  }

  function reset() {
    setInputs([0.6, 0.9]);
    setWeights([0.8, -0.4]);
    setBias(0.2);
    setActivation("relu");
    setCurveAxis(0);
    setReference({ weights: [0.8, -0.4], bias: 0.2 });
    onInteract();
  }

  return (
    <div className="experiment-layout neuron-workspace">
      <div className="neuron-interactive">
        <ExperimentPanel title="参数如何改变输出曲线" eyebrow="01 / OBSERVE" className="neuron-output-panel" action={<span className="live-tag" title={calculationStatus(error, pending)}><i />实时预览</span>}>
          <div className="neuron-curve-toolbar">
            <div className="segmented" role="group" aria-label="输出曲线横轴">
              {([0, 1] as const).map(axis => <button key={axis} type="button" aria-pressed={curveAxis === axis}
                onClick={() => { setCurveAxis(axis); onInteract(); }}>
                输入 {axis + 1} · x{axis === 0 ? "₁" : "₂"}
              </button>)}
            </div>
            <button className="text-button neuron-reference-button" type="button" onClick={() => {
              setReference({ weights: [...weights], bias });
              onInteract();
            }} title="保存当前权重和偏置，用虚线对比后续调整">设为参考</button>
          </div>
          <p className="panel-note neuron-curve-context">
            固定输入 x{fixedAxis === 0 ? "₁" : "₂"} = {inputs[fixedAxis].toFixed(1)}，
            扫描 x{curveAxis === 0 ? "₁" : "₂"} 从 −3 到 3。当前激活：{ACTIVATION_LABELS[activation]}。
          </p>
          <div className="plot-legend">
            <span><i className="legend-line" />当前参数</span>
            <span><i className="legend-line dashed" />参考参数</span>
            <span><i className="legend-dot" />当前输出</span>
          </div>
          <Chart
            series={[{ points: outputCurve, color: "#7ed2c4" }, { points: referenceCurve, color: "#c89974", dashed: true }]}
            xLabel={`输入 x${curveAxis === 0 ? "₁" : "₂"}`}
            yLabel="输出 a"
            xDomain={[-3, 3]}
            height={240}
            yDomain={activation === "sigmoid" ? [-0.08, 1.08] : activation === "tanh" ? [-1.12, 1.12] : undefined}
            markers={[{ x: inputs[curveAxis], y: current.output, label: "当前输入的输出", color: "#c2eaa1" }]}
            ariaLabel={`神经元输出曲线，横轴为输入 ${curveAxis + 1}，固定输入 ${fixedAxis + 1} 为 ${inputs[fixedAxis]}，激活函数 ${ACTIVATION_LABELS[activation]}，当前输出 ${display(current.output)}`}
          />
          <div className="neuron-curve-metrics" aria-live="polite">
            <Result label={`当前输入 x${curveAxis === 0 ? "₁" : "₂"}`} value={inputs[curveAxis].toFixed(1)} />
            <Result label="加权和 z" value={format(current.z)} />
            <Result label="最终输出 a" value={format(current.output)} />
          </div>
          <ErrorMessage error={error} />
          <details className="neuron-curve-notes">
            <summary>公式、参考参数与观察提示<ChevronDown size={15} aria-hidden="true" /></summary>
          <MathFormula tex={String.raw`a=f\!\left(${weights[curveAxis].toFixed(1)}x_{${curveAxis + 1}}+(${weights[fixedAxis].toFixed(1)})\cdot(${inputs[fixedAxis].toFixed(1)})+(${bias.toFixed(1)})\right)`} block />
          <div className="neuron-curve-reference">
            <span>参考：w₁ = {reference.weights[0].toFixed(1)} · w₂ = {reference.weights[1].toFixed(1)} · b = {reference.bias.toFixed(1)}</span>
            <span>两条曲线使用相同的激活函数和固定输入</span>
          </div>
          <p className="panel-note">
            {weights[curveAxis] === 0
              ? "当前横轴对应的权重为 0，输出不随这个输入变化，曲线是一条水平线。"
              : `调整 w${curveAxis === 0 ? "₁" : "₂"}，观察曲线的陡缓与方向；当前权重${weights[curveAxis] > 0 ? "为正，曲线随输入增大而上升（饱和区或 ReLU 零区可能保持平坦）" : "为负，曲线随输入增大而下降（饱和区或 ReLU 零区可能保持平坦）"}。`}
            改变偏置 b，观察曲线的位置变化；改变横轴对应的输入值，只会移动标记点。
          </p>
          </details>
        </ExperimentPanel>
      <aside className="control-panel neuron-controls" aria-label="神经元实验控制">
        <div className="neuron-controls-heading">
          <h2><SlidersHorizontal size={16} aria-hidden="true" />调整参数</h2>
          <button className="text-button" type="button" onClick={reset}><RotateCcw size={14} aria-hidden="true" />重置</button>
        </div>
        <div className="neuron-control-groups">
          <fieldset className="neuron-control-group neuron-activation-group">
            <legend>激活函数</legend>
            <div
              className="segmented activation-options"
              role="group"
              aria-label="激活函数"
            >
              {ACTIVATIONS.map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={activation === value}
                  onClick={() => {
                    setActivation(value);
                    onInteract();
                  }}
                >
                  {ACTIVATION_LABELS[value]}
                </button>
              ))}
            </div>
          </fieldset>
          <fieldset className="neuron-control-group neuron-weights-group">
            <legend>权重与偏置</legend>
            <div className="neuron-slider-pair">
          <Slider
            label="权重 1"
            symbol="w₁"
            value={weights[0]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(value) => changeWeight(0, value)}
          />
          <Slider
            label="权重 2"
            symbol="w₂"
            value={weights[1]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(value) => changeWeight(1, value)}
          />
            </div>
            <div className="neuron-bias-control">
          <Slider
            label="偏置"
            symbol="b"
            value={bias}
            min={-3}
            max={3}
            step={0.1}
            onChange={(value) => {
              setBias(value);
              onInteract();
            }}
          />
            </div>
          </fieldset>
          <fieldset className="neuron-control-group neuron-inputs-group">
            <legend>输入特征</legend>
            <div className="neuron-slider-pair">
          <Slider
            label="输入 1"
            symbol="x₁"
            value={inputs[0]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(value) => changeInput(0, value)}
          />
          <Slider
            label="输入 2"
            symbol="x₂"
            value={inputs[1]}
            min={-3}
            max={3}
            step={0.1}
            onChange={(value) => changeInput(1, value)}
          />
            </div>
            <p className="panel-note">横轴对应的输入控制标记点，另一个输入固定曲线的切片位置。</p>
          </fieldset>
        </div>
      </aside>
      </div>
      <details className="neuron-explanation">
        <summary>
          <span><span className="eyebrow">02 / UNDERSTAND</span><strong>计算过程与激活函数</strong></span>
          <ChevronDown size={18} aria-hidden="true" />
        </summary>
        <div className="neuron-explanation-body">
        <ExperimentPanel
          title="一个神经元如何计算"
          eyebrow="FORWARD PASS"
        >
          <div className="neuron-diagram" aria-busy={pending}>
            <svg
              className="neuron-desktop"
              viewBox="0 0 720 280"
              role="img"
              aria-labelledby="neuron-graph-title neuron-graph-description"
            >
              <title id="neuron-graph-title">
                两个输入经过加权求和，再通过激活函数得到输出
              </title>
              <desc id="neuron-graph-description">
                输入 x1 为 {inputs[0]}，权重 w1 为 {weights[0]}；输入 x2 为{" "}
                {inputs[1]}，权重 w2 为 {weights[1]}。偏置为 {bias}，激活函数为{" "}
                {ACTIVATION_LABELS[activation]}。加权和为 {display(current.z)}
                ，输出为 {display(current.output)}。
              </desc>
              {[0, 1].map((index) => {
                const y = index === 0 ? 65 : 205;
                const weight = weights[index] ?? 0;
                return (
                  <g key={index}>
                    <path
                      d={`M 122 ${y} L 285 135`}
                      fill="none"
                      stroke={weight < 0 ? "#858b7d" : "#252a23"}
                      strokeWidth={1.5 + Math.abs(weight) * 2}
                      strokeDasharray={weight < 0 ? "6 5" : undefined}
                    />
                    <circle
                      cx="80"
                      cy={y}
                      r="40"
                      fill="#f1efe8"
                      stroke="#bfc3b6"
                    />
                    <text
                      x="80"
                      y={y - 5}
                      textAnchor="middle"
                      className="diagram-label"
                    >
                      x{index + 1}
                    </text>
                    <text
                      x="80"
                      y={y + 15}
                      textAnchor="middle"
                      className="diagram-value"
                    >
                      {inputs[index]?.toFixed(1)}
                    </text>
                    <text
                      x="180"
                      y={index === 0 ? 55 : 210}
                      textAnchor="middle"
                      className="diagram-label"
                    >
                      w{index + 1} = {weight.toFixed(1)}（
                      {weight < 0 ? "负" : weight > 0 ? "正" : "零"}）
                    </text>
                    <text
                      x="190"
                      y={index === 0 ? 76 : 230}
                      textAnchor="middle"
                      className="diagram-detail"
                    >
                      乘积 {display(current.products[index])}
                    </text>
                  </g>
                );
              })}
              <path d="M 345 135 L 413 135" stroke="#252a23" strokeWidth="2" />
              <path d="M 535 135 L 592 135" stroke="#252a23" strokeWidth="2" />
              <circle cx="315" cy="135" r="35" fill="#c9ef6b" />
              <text x="315" y="144" textAnchor="middle" className="diagram-sum">
                Σ
              </text>
              <text
                x="315"
                y="194"
                textAnchor="middle"
                className="diagram-label"
              >
                偏置 b {signed(bias)}
              </text>
              <text
                x="378"
                y="117"
                textAnchor="middle"
                className="diagram-label"
              >
                z
              </text>
              <text
                x="378"
                y="161"
                textAnchor="middle"
                className="diagram-detail"
              >
                {display(current.z)}
              </text>
              <rect
                x="413"
                y="96"
                width="122"
                height="78"
                rx="12"
                fill="#252a23"
              />
              <text
                x="474"
                y="129"
                textAnchor="middle"
                className="diagram-activation"
              >
                {ACTIVATION_LABELS[activation]}
              </text>
              <text
                x="474"
                y="151"
                textAnchor="middle"
                className="diagram-activation-detail"
              >
                激活函数
              </text>
              <circle cx="635" cy="135" r="43" fill="#c9ef6b" />
              <text
                x="635"
                y="125"
                textAnchor="middle"
                className="diagram-label"
              >
                输出 a
              </text>
              <text
                x="635"
                y="149"
                textAnchor="middle"
                className="diagram-value"
              >
                {display(current.output)}
              </text>
            </svg>
            <div
              className="neuron-compact"
              role="group"
              aria-label="神经元计算流程：输入乘以权重、加上偏置、激活、得到输出"
            >
              <div className="neuron-compact-inputs">
                {([0, 1] as const).map((index) => (
                  <div className="neuron-compact-input" key={index}>
                    <span className="neuron-compact-label">
                      输入 {index + 1} × 权重 {index + 1}
                    </span>
                    <div className="neuron-compact-product">
                      <span>
                        x{index === 0 ? "₁" : "₂"} {inputs[index].toFixed(1)} ×
                        w{index === 0 ? "₁" : "₂"} ({weights[index].toFixed(1)})
                      </span>
                      <strong>= {display(current.products[index])}</strong>
                    </div>
                  </div>
                ))}
              </div>
              <ArrowDown className="neuron-flow-arrow" aria-hidden="true" />
              <div className="neuron-compact-stage neuron-compact-sum">
                <span className="neuron-compact-symbol" aria-hidden="true">
                  Σ
                </span>
                <div>
                  <span className="neuron-compact-label">
                    加权求和 · 偏置 b = {bias.toFixed(1)}
                  </span>
                  <p>
                    {display(current.products[0])} + ({display(current.products[1])}
                    ) {signed(bias)}
                  </p>
                  <strong>z = {display(current.z)}</strong>
                </div>
              </div>
              <ArrowDown className="neuron-flow-arrow" aria-hidden="true" />
              <div className="neuron-compact-stage neuron-compact-activation">
                <span className="neuron-compact-label">
                  应用激活函数 · {ACTIVATION_LABELS[activation]}
                </span>
                <MathFormula tex={ACTIVATION_FORMULAS[activation]} />
              </div>
              <ArrowDown className="neuron-flow-arrow" aria-hidden="true" />
              <div className="neuron-compact-stage neuron-compact-output">
                <span className="neuron-compact-label">最终输出</span>
                <strong>a = {display(current.output)}</strong>
              </div>
            </div>
          </div>
          <p className="axis-note">
            <span className="neuron-desktop-note">
              线越粗，权重绝对值越大；实线表示非负权重，虚线表示负权重。
            </span>
            <span className="neuron-mobile-note">
              将两路乘积与偏置相加，再应用激活函数。
            </span>
            当前数值即时更新，并由后端同步校验。
          </p>
          <div className="formula-row neuron-formulas">
            <div>
              <div className="small-label">第一步 · 加权求和</div>
              <MathFormula className="formula" tex="z=x_1w_1+x_2w_2+b" block />
            </div>
            <div>
              <div className="small-label">第二步 · 激活</div>
              <MathFormula className="formula" tex={ACTIVATION_FORMULAS[activation]} block />
            </div>
          </div>
          <p className="neuron-substitution">
            z = ({inputs[0].toFixed(1)} × {weights[0].toFixed(1)}) + (
            {inputs[1].toFixed(1)} × {weights[1].toFixed(1)}) {signed(bias)} ={" "}
            <strong>{display(current.z)}</strong>
          </p>
          <div className="result-grid" aria-live="polite">
            <Result
              label="第一路乘积 x₁w₁"
              value={display(current.products[0])}
            />
            <Result
              label="第二路乘积 x₂w₂"
              value={display(current.products[1])}
            />
            <Result
              label="最终输出 a"
              value={display(current.output)}
              hint={`通过 ${ACTIVATION_LABELS[activation]} 变换`}
            />
          </div>
        </ExperimentPanel>

        <ExperimentPanel title="激活函数改变了什么" eyebrow="NONLINEARITY">
          <Chart
            series={[{ points: activationCurve, color: "#7ed2c4" }]}
            xLabel="加权和 z"
            yLabel="激活输出 a"
            markers={[{ x: current.z, y: current.output, label: "当前输出", color: "#c2eaa1" }]}
            ariaLabel={`${ACTIVATION_LABELS[activation]} 激活函数曲线，当前加权和 ${display(current.z)}，输出 ${display(current.output)}`}
          />
          <p className="panel-note">
            {activation === "linear"
              ? "线性激活不改变加权和。多个纯线性层叠加，仍可合并成一个线性变换。"
              : activation === "relu"
                ? "ReLU 保留正值，把负值变为 0。将偏置向左移动，观察当前输出何时归零。"
                : activation === "sigmoid"
                  ? "Sigmoid 将输出压缩到 0–1。这个数在本实验中只是激活值，不自动代表可信的分类概率。"
                  : "Tanh 将输出压缩到 −1 到 1；输入靠近 0 时，输出变化更明显。"}
          </p>
        </ExperimentPanel>
        </div>
      </details>
    </div>
  );
}
