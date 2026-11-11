import { useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import { MathFormula } from "../components/MathFormula";
import { Chart, ControlsHeading, ErrorMessage, ExperimentPanel, Result, Slider } from "../components/ui";
import { calculationStatus, format, useCalculation } from "../lib/api";
import { useLessonTask } from "../lib/learningProgress";
import { lossPreview } from "../lib/loss";
import type { LossKind } from "../lib/loss";
import "./loss.css";

const definitions = {
  mse: {
    name: "MSE · 平方误差", short: "MSE", color: "var(--series-green)",
    formula: String.raw`L=e^2,\quad \frac{\partial L}{\partial\hat y}=2e`,
    hint: "L = (ŷ − y)²", derivative: "∂L/∂ŷ = 2(ŷ − y)",
    title: "误差越大，惩罚增长越快",
    description: "平方误差适合连续数值预测。误差放大两倍，损失变为四倍；大误差样本也会带来更大的梯度，因此异常值可能明显影响训练。",
    task: "点击“大误差”，再打开曲线对比，观察 MSE 的损失和梯度如何超过另外两种函数。",
  },
  mae: {
    name: "MAE · 绝对误差", short: "MAE", color: "var(--series-amber)",
    formula: String.raw`L=|e|,\quad \frac{\partial L}{\partial\hat y}=\operatorname{sign}(e)\ (e\ne0)`,
    hint: "L = |ŷ − y|", derivative: "非零误差处：∂L/∂ŷ = sign(ŷ − y)",
    title: "误差线性增长，对极端偏差更温和",
    description: "MAE 适合希望减弱异常值影响的回归任务。非零误差处的梯度大小始终为 1；零误差处有尖角，普通导数不存在，本实验选用次梯度 0。",
    task: "从“小误差”切换到“大误差”：损失变大了，梯度大小是否也变大？再点击“预测准确”观察尖角。",
  },
  huber: {
    name: "Huber · 平滑过渡", short: "Huber", color: "var(--series-blue)",
    formula: String.raw`L=\begin{cases}\frac12e^2&|e|\le\delta\\\delta(|e|-\frac12\delta)&|e|>\delta\end{cases}`,
    hint: "小误差用 ½e²，大误差改为线性增长", derivative: "∂L/∂ŷ = clip(ŷ − y, −δ, δ)",
    title: "兼顾小误差的平滑与大误差的稳健",
    description: "Huber 在阈值 δ 内使用半平方误差，超过阈值后线性增长，连接处的梯度连续。减小 δ 会让更多误差进入线性区，同时降低梯度的最大幅度。",
    task: "保持当前预测不变，拖动 δ，观察损失和梯度的变化。注意：二次区域是 ½e²，与这里的 MSE 相差一个系数。",
  },
  bce: {
    name: "BCE · 二分类交叉熵", short: "BCE", color: "var(--series-violet)",
    formula: String.raw`p=\sigma(z),\quad L=-y\ln p-(1-y)\ln(1-p)`,
    hint: "从 logits 稳定计算二分类交叉熵", derivative: "∂L/∂z = sigmoid(z) − y = p − y",
    title: "对自信却错误的预测施加更大惩罚",
    description: "用于是/否等二分类任务。原始分数 z 经 sigmoid 转为类别 1 的概率 p。本图横轴是 z，显示的梯度也是对 z 求导。标签为 0 或 1 时，损失随正确方向上的 z 趋向无穷而趋近 0，没有有限的零损失点。",
    task: "比较“不确定”“自信且正确”“自信却错误”，再切换真实类别。观察损失怎样区分预测的把握程度。",
  },
};
const regressionKinds: LossKind[] = ["mse", "mae", "huber"];

export default function LossExperiment({ onInteract }: { onInteract: () => void }) {
  const [kind, setKind] = useState<LossKind>("mse");
  const [prediction, setPrediction] = useState(-1);
  const [target, setTarget] = useState(2);
  const [logit, setLogit] = useState(-2);
  const [binaryTarget, setBinaryTarget] = useState(1);
  const [delta, setDelta] = useState(1);
  const [compare, setCompare] = useState(false);
  const binary = kind === "bce";
  const input = binary ? logit : prediction;
  const label = binary ? binaryTarget : target;
  const definition = definitions[kind];
  const preview = useMemo(() => lossPreview(input, label, kind, delta), [input, label, kind, delta]);
  const { data, error, pending } = useCalculation("/loss", { prediction: input, target: label, kind, delta }, preview);
  const current = data ?? preview;
  const comparisons = useMemo(() => regressionKinds.map(value => ({ kind: value, ...lossPreview(prediction, target, value, delta) })), [prediction, target, delta]);
  const probability = 1 / (1 + Math.exp(-logit));
  useLessonTask("loss", !binary && Math.abs(prediction - target) < 1e-12 && current.loss < 1e-12 && Math.abs(current.gradient) < 1e-12);

  return (
    <div className="experiment-layout loss-experiment">
      <div className="experiment-main">
        <ExperimentPanel title="同样的预测，不同的惩罚方式" eyebrow="LOSS FUNCTIONS" action={
          <span className="live-tag" title={calculationStatus(error, pending)}><i />实时预览</span>
        }>
          <div className="loss-picker" role="group" aria-label="选择损失函数">
            {(Object.keys(definitions) as LossKind[]).map(value => (
              <button key={value} type="button" aria-pressed={kind === value} onClick={() => { setKind(value); onInteract(); }}>
                <strong>{definitions[value].short}</strong><span>{definitions[value].name.split(" · ")[1]}</span>
              </button>
            ))}
          </div>
          <div className="loss-toolbar">
            <div className="plot-legend">
              {(compare && !binary ? [kind, ...regressionKinds.filter(value => value !== kind)] : [kind]).map(value => (
                <span key={value}><i className="legend-line" style={{ borderColor: definitions[value].color, background: definitions[value].color }} />{definitions[value].short}{value === kind ? " · 当前" : ""}</span>
              ))}
            </div>
            {!binary && <label className="normalize-toggle"><input type="checkbox" checked={compare} onChange={event => { setCompare(event.target.checked); onInteract(); }} />对比三种回归损失</label>}
          </div>
          <Chart
            ariaLabel={`${definition.name}随${binary ? "原始分数 z" : "预测值"}变化的曲线${compare && !binary ? "，同时对比三种回归损失" : ""}`}
            series={[{ points: current.curve, color: definition.color }, ...(compare && !binary ? comparisons.filter(item => item.kind !== kind).map(item => ({ points: item.curve, color: definitions[item.kind].color, dashed: true })) : [])]}
            xLabel={binary ? "原始分数 z（不是概率）" : "预测值 ŷ"} yLabel="损失 L"
            markers={[{ x: input, y: current.loss, label: "当前预测" }, ...(!binary ? [{ x: target, y: 0, label: "损失最低点", color: "var(--fwd)" }] : [])]}
          />
          <div className="result-grid" aria-live="polite">
            <Result label={definition.name} value={format(current.loss, 4)} hint={definition.hint} />
            <Result label={!current.differentiable ? "次梯度（选取 0）" : binary ? "对 z 的梯度" : "对预测值的梯度"} value={format(current.gradient, 4)} hint={!current.differentiable ? "零误差处不可导，次梯度范围为 [−1, 1]" : definition.derivative} />
            {binary && <Result label="预测为类别 1 的概率" value={`${format(probability * 100, 2)}%`} hint={`p = sigmoid(z) · 真实类别 ${binaryTarget}`} />}
          </div>
          <ErrorMessage error={error} />
        </ExperimentPanel>
        <div className="explanation-strip"><span className="explanation-index">L</span><div><h3>{definition.title}</h3><p>{definition.description}</p></div></div>
        {!binary && compare && <ExperimentPanel title="同一误差下，损失与梯度各是多少？" eyebrow="COMPARE">
          <div className="loss-table-wrap"><table className="loss-table"><thead><tr><th>函数</th><th>当前损失</th><th>梯度 / 次梯度</th><th>大误差时</th></tr></thead><tbody>
            {comparisons.map(item => <tr key={item.kind} aria-current={item.kind === kind ? "true" : undefined}><th>{definitions[item.kind].short}{item.kind === "huber" ? `（δ = ${format(delta)}）` : ""}</th><td>{format(item.loss, 4)}</td><td>{format(item.gradient, 4)}{!item.differentiable ? "（次梯度）" : ""}</td><td>{item.kind === "mse" ? "平方增长" : "线性增长"}</td></tr>)}
          </tbody></table></div>
          <p className="panel-note">这里比较同一个样本，e = ŷ − y = {format(prediction - target)}。不同损失的数值尺度不同，数值更小不代表模型更好；应结合任务与独立评估选择。</p>
        </ExperimentPanel>}
      </div>
      <aside className="control-panel">
        <ControlsHeading action={<button className="icon-button" aria-label="重置损失实验" onClick={() => { setKind("mse"); setPrediction(-1); setTarget(2); setLogit(-2); setBinaryTarget(1); setDelta(1); setCompare(false); onInteract(); }}><RotateCcw size={15} /></button>} />
        <div className="control-stack">
          <Slider label={binary ? "原始分数" : "预测值"} symbol={binary ? "z" : "ŷ"} value={input} min={binary ? -8 : -5} max={binary ? 8 : 5} step={0.1} onChange={value => { if (binary) setLogit(value); else setPrediction(value); onInteract(); }} />
          {binary ? <div><span className="small-label">真实类别 y</span><div className="segmented" role="group" aria-label="真实类别">{[0, 1].map(value => <button key={value} aria-pressed={binaryTarget === value} onClick={() => { setBinaryTarget(value); onInteract(); }}>类别 {value}</button>)}</div></div> : <Slider label="真实标签" symbol="y" value={target} min={-3} max={3} step={0.1} onChange={value => { setTarget(value); onInteract(); }} />}
          {!binary && (kind === "huber" || compare) && <Slider label="Huber 阈值" symbol="δ" value={delta} min={0.1} max={3} step={0.1} onChange={value => { setDelta(value); onInteract(); }} />}
        </div>
        <div className="loss-presets" role="group" aria-label="观察示例">
          {(binary ? ["不确定", "自信且正确", "自信却错误"] : ["预测准确", "小误差", "大误差"]).map((name, index) => <button key={name} type="button" onClick={() => {
            if (binary) setLogit(index === 0 ? 0 : (binaryTarget === 1 ? 1 : -1) * (index === 1 ? 6 : -6));
            else setPrediction(index === 0 ? target : index === 1 ? target + 0.5 : target > 0 ? -5 : 5);
            onInteract();
          }}>{name}</button>)}
        </div>
        <MathFormula className="formula loss-formula" tex={definition.formula} block />
        <p className="panel-note">{binary ? "本实验与 XOR 训练都从 logits 计算 BCE，避免概率取对数时的数值问题。MNIST 十分类使用多分类交叉熵。" : "这里展示单个样本的损失；MSE / MAE 中的 Mean 表示多样本时再取平均。回归误差 e = ŷ − y。"}</p>
        <div className="insight"><span className="small-label">试一试</span><p>{definition.task}</p></div>
      </aside>
    </div>
  );
}
