import { useMemo, useState } from "react";
import { ArrowDown } from "lucide-react";
import { useCalculation } from "../lib/api";
import {
  Slider,
  ExperimentPanel,
  Result,
  ErrorMessage,
  Chart,
} from "../components/ui";
import "./neuron.css";

type Activation = "relu" | "sigmoid" | "tanh" | "linear";

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
  relu: "a = max(0, z)",
  sigmoid: "a = 1 / (1 + e⁻ᶻ)",
  tanh: "a = tanh(z)",
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
  const payload = useMemo(
    () => ({ inputs, weights, bias, activation }),
    [inputs, weights, bias, activation],
  );
  const { data, error, pending } = useCalculation<NeuronResponse>(
    "/neuron",
    payload,
  );

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
    onInteract();
  }

  return (
    <div className="experiment-layout">
      <div className="experiment-main">
        <ExperimentPanel
          title="一个神经元如何计算"
          eyebrow="02 / INPUT → WEIGHT → OUTPUT"
          action={
            <button className="text-button" type="button" onClick={reset}>
              重置实验
            </button>
          }
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
                {ACTIVATION_LABELS[activation]}。加权和为 {display(data?.z)}
                ，输出为 {display(data?.output)}。
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
                      乘积 {display(data?.products[index])}
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
                {display(data?.z)}
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
                {display(data?.output)}
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
                      <strong>= {display(data?.products[index])}</strong>
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
                    {display(data?.products[0])} + ({display(data?.products[1])}
                    ) {signed(bias)}
                  </p>
                  <strong>z = {display(data?.z)}</strong>
                </div>
              </div>
              <ArrowDown className="neuron-flow-arrow" aria-hidden="true" />
              <div className="neuron-compact-stage neuron-compact-activation">
                <span className="neuron-compact-label">
                  应用激活函数 · {ACTIVATION_LABELS[activation]}
                </span>
                <strong>{ACTIVATION_FORMULAS[activation]}</strong>
              </div>
              <ArrowDown className="neuron-flow-arrow" aria-hidden="true" />
              <div className="neuron-compact-stage neuron-compact-output">
                <span className="neuron-compact-label">最终输出</span>
                <strong>a = {display(data?.output)}</strong>
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
            {pending ? "后端正在计算当前参数…" : ""}
          </p>
          <div className="formula-row neuron-formulas">
            <div>
              <div className="small-label">第一步 · 加权求和</div>
              <div className="formula">z = x₁w₁ + x₂w₂ + b</div>
            </div>
            <div>
              <div className="small-label">第二步 · 激活</div>
              <div className="formula">{ACTIVATION_FORMULAS[activation]}</div>
            </div>
          </div>
          <p className="neuron-substitution">
            z = ({inputs[0].toFixed(1)} × {weights[0].toFixed(1)}) + (
            {inputs[1].toFixed(1)} × {weights[1].toFixed(1)}) {signed(bias)} ={" "}
            <strong>{display(data?.z)}</strong>
          </p>
          <div className="result-grid" aria-live="polite">
            <Result
              label="第一路乘积 x₁w₁"
              value={display(data?.products[0])}
            />
            <Result
              label="第二路乘积 x₂w₂"
              value={display(data?.products[1])}
            />
            <Result
              label="最终输出 a"
              value={display(data?.output)}
              hint={`通过 ${ACTIVATION_LABELS[activation]} 变换`}
            />
          </div>
          <ErrorMessage error={error} />
        </ExperimentPanel>

        <ExperimentPanel title="激活函数改变了什么" eyebrow="NONLINEARITY">
          {data ? (
            <Chart
              series={[{ points: data.curve, color: "#5d7629" }]}
              xLabel="加权和 z"
              yLabel="激活输出 a"
              markers={[
                {
                  x: data.z,
                  y: data.output,
                  label: "当前输出",
                  color: "#252a23",
                },
              ]}
              ariaLabel={`${ACTIVATION_LABELS[activation]} 激活函数曲线，当前加权和 ${display(data.z)}，输出 ${display(data.output)}`}
            />
          ) : (
            <div className="chart-placeholder">
              {error
                ? "计算失败，请检查上方错误信息。"
                : "等待后端返回激活函数曲线…"}
            </div>
          )}
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

      <aside className="control-panel" aria-label="神经元实验控制">
        <div className="control-stack">
          <div className="small-label">输入特征</div>
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
          <div className="small-label">模型参数</div>
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
          <div>
            <div className="small-label">激活函数</div>
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
          </div>
          <div className="insight">
            试一试：让一个权重变为负数，再增大对应输入。观察它如何拉低加权和。
          </div>
          <p className="panel-note">
            本节手动调整参数来理解前向计算；后续训练会根据损失自动更新参数。
          </p>
        </div>
      </aside>
    </div>
  );
}
