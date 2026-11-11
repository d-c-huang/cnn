import { MathFormula } from "../components/MathFormula";
import { useMemo, useState } from "react";
import { ArrowDown, ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import { calculationStatus, format, useCalculation } from "../lib/api";
import { useLessonTask } from "../lib/learningProgress";
import { neuronActivationCurve, neuronOutputCurve, neuronValue } from "../lib/neuron";
import type { Activation, Pair } from "../lib/neuron";
import { ACTIVATIONS, activationInfo } from "../lib/activations";
import ActivationExplorer from "./ActivationExplorer";
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
  derivative: number;
  differentiable: boolean;
  curve: Array<{ x: number; y: number }>;
}

interface ExperimentProps {
  onInteract: () => void;
}

const ACTIVATION_LABELS = Object.fromEntries(ACTIVATIONS.map(value => [value, activationInfo[value].label])) as Record<Activation, string>;
const ACTIVATION_FORMULAS = Object.fromEntries(ACTIVATIONS.map(value => [value, activationInfo[value].formula])) as Record<Activation, string>;

function display(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(3);
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : "−"} ${Math.abs(value).toFixed(1)}`;
}

export default function NeuronExperiment({ onInteract }: ExperimentProps) {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [comparison, setComparison] = useState<string | null>(null);
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

  function changeActivation(value: Activation) {
    setActivation(value);
    onInteract();
  }

  function activationPreset(z: number) {
    setInputs([z / 3, 0]);
    setWeights([3, 0]);
    setBias(0);
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
        <aside className="prediction-prompt"><strong>先预判，再运行</strong><p>保留当前输入和偏置，选择 Sigmoid 并把 w₁ 设为 −0.8。与当前权重下的 Sigmoid 输出相比会怎样？</p><div role="group" aria-label="预判神经元输出">{["下降", "上升", "不变"].map(option => <button key={option} className="secondary-button" aria-pressed={prediction === option} onClick={() => { setPrediction(option); setComparison(null); }}>{option}</button>)}</div><button className="primary-button" disabled={!prediction} onClick={() => {
          const before = neuronValue(inputs, weights, bias, "sigmoid").output;
          const after = neuronValue(inputs, [-0.8, weights[1]], bias, "sigmoid").output;
          setActivation("sigmoid"); setWeights([-0.8, weights[1]]); onInteract();
          setComparison(`你的预判：${prediction}；本次教学计算：${Math.abs(after - before) < 1e-10 ? "不变" : after < before ? "下降" : "上升"}，${before.toFixed(4)} → ${after.toFixed(4)}。注意结果还取决于当前输入的符号。`);
        }}>验证这组参数</button>{comparison && <p role="status">{comparison}</p>}</aside>
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
            series={[{ points: outputCurve, color: "var(--fwd)" }, { points: referenceCurve, color: "var(--bwd)", dashed: true }]}
            xLabel={`输入 x${curveAxis === 0 ? "₁" : "₂"}`}
            yLabel="输出 a"
            xDomain={[-3, 3]}
            height={240}
            yDomain={activation === "sigmoid" ? [-0.08, 1.08] : activation === "tanh" ? [-1.12, 1.12] : undefined}
            markers={[{ x: inputs[curveAxis], y: current.output, label: "当前输入的输出", color: "var(--fwd)" }]}
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
              : activation === "silu"
                ? "SiLU 不是全程单调函数，曲线方向由权重与局部导数的乘积决定。到下方激活实验查看导数的符号。"
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
                  onClick={() => changeActivation(value)}
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
      <ActivationExplorer activation={activation} z={current.z} output={current.output}
        derivative={current.derivative} differentiable={current.differentiable}
        weight={weights[curveAxis]} axis={curveAxis} onActivation={changeActivation}
        onPreset={activationPreset} onInteract={onInteract} />
      <details className="neuron-explanation">
        <summary>
          <span><span className="eyebrow">03 / UNDERSTAND</span><strong>展开神经元的计算过程</strong></span>
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
                      stroke={weight < 0 ? "var(--bwd)" : "var(--fwd)"}
                      strokeWidth={1.5 + Math.abs(weight) * 2}
                      strokeDasharray={weight < 0 ? "6 5" : undefined}
                    />
                    <circle
                      cx="80"
                      cy={y}
                      r="40"
                      fill="var(--surface-3)"
                      stroke="var(--border)"
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
              <path d="M 345 135 L 413 135" stroke="var(--text-2)" strokeWidth="2" />
              <path d="M 535 135 L 592 135" stroke="var(--text-2)" strokeWidth="2" />
              <circle cx="315" cy="135" r="35" fill="var(--fwd)" />
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
                fill="var(--text-2)"
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
              <circle cx="635" cy="135" r="43" fill="var(--output-node)" />
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

        </div>
      </details>
    </div>
  );
}
