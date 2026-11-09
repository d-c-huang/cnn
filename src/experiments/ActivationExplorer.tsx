import { useMemo, useState } from "react";
import { Chart, ExperimentPanel, Result } from "../components/ui";
import { MathFormula } from "../components/MathFormula";
import { format } from "../lib/api";
import { ACTIVATIONS, activationInfo } from "../lib/activations";
import { activateNeuron, activationDerivative, neuronActivationCurve, neuronDerivativeCurves } from "../lib/neuron";
import type { Activation } from "../lib/neuron";

export default function ActivationExplorer({ activation, z, output, derivative, differentiable, weight, axis, onActivation, onPreset, onInteract }: {
  activation: Activation; z: number; output: number; derivative: number; differentiable: boolean; weight: number; axis: 0 | 1;
  onActivation: (value: Activation) => void; onPreset: (value: number) => void; onInteract: () => void;
}) {
  const [compare, setCompare] = useState(false);
  const info = activationInfo[activation];
  const visible = compare ? [activation, ...ACTIVATIONS.filter(value => value !== activation)] : [activation];
  const curves = useMemo(() => Object.fromEntries(ACTIVATIONS.map(value => [value, {
    output: neuronActivationCurve(z, value), derivative: neuronDerivativeCurves(z, value),
  }])), [z]);
  return <ExperimentPanel title="激活函数怎样改变输出与梯度" eyebrow="02 / ACTIVATION LAB" className="activation-lab">
    <div className="activation-lab-toolbar">
      <label className="activation-select">观察函数<select value={activation} onChange={event => onActivation(event.target.value as Activation)}>{ACTIVATIONS.map(value => <option key={value} value={value}>{activationInfo[value].label}</option>)}</select></label>
      <label className="normalize-toggle"><input type="checkbox" checked={compare} onChange={event => { setCompare(event.target.checked); onInteract(); }} />对比全部激活函数</label>
      <div className="activation-presets" role="group" aria-label="激活函数观察示例">
        {[{ label: "负值区 · z = −3", value: -3 }, { label: "零点 · z = 0", value: 0 }, { label: "大正值 · z = 6", value: 6 }].map(item => <button key={item.value} type="button" onClick={() => onPreset(item.value)}>{item.label}</button>)}
      </div>
    </div>
    <p className="panel-note">上图横轴是输入 x，这里的横轴是加权和 z。示例会设置 w₁ = 3、w₂ = b = x₂ = 0、x₁ = z / 3；当前 z = {format(z)}。</p>
    <div className="plot-legend activation-legend">{visible.map(value => <span key={value}><i className="legend-line" style={{ background: activationInfo[value].color, borderColor: activationInfo[value].color }} />{activationInfo[value].label}{value === activation ? " · 当前" : ""}</span>)}</div>
    <div className="activation-charts">
      <div><h3>前向：得到怎样的输出？</h3><Chart
        series={visible.map(value => ({ points: curves[value].output, color: activationInfo[value].color, dashed: value !== activation }))}
        xLabel="加权和 z" yLabel="激活输出 f(z)" height={255}
        markers={[{ x: z, y: output, label: `${info.label} 当前输出` }]}
        ariaLabel={`${info.label} 激活输出曲线${compare ? "及八种函数对比" : ""}`} /></div>
      <div><h3>反向：局部梯度如何变化？</h3><Chart
        series={visible.flatMap(value => curves[value].derivative.map(points => ({ points, color: activationInfo[value].color, dashed: value !== activation })))}
        xLabel="加权和 z" yLabel="局部导数 f′(z)" height={255}
        markers={[{ x: z, y: derivative, label: differentiable ? "当前局部导数" : "零点约定反传值" }]}
        ariaLabel={`${info.label} 局部导数曲线，ReLU 与 Leaky ReLU 在零点断开${compare ? "，对比八种函数" : ""}`} /></div>
    </div>
    <div className="activation-metrics" aria-live="polite">
      <Result label="激活输出" value={format(output, 4)} hint={`理论值域：${info.range}`} />
      <Result label={differentiable ? "局部导数 f′(z)" : "零点约定反传值"} value={format(derivative, 4)} hint={differentiable ? "传回 z 的梯度 = 上游梯度 × f′(z)" : "普通导数不存在；当前值用于演示反向传播"} />
      <Result label={`输出对输入 x${axis + 1} 的${differentiable ? "导数" : "约定反传值"}`} value={format(weight * derivative, 4)} hint={`w${axis + 1} × f′(z) = ${format(weight)} × ${format(derivative, 4)}`} />
    </div>
    <div className="activation-description">
      <div><h3>{info.label} · {info.use}</h3><p>{info.description}</p><p className="panel-note">试一试：{info.observation}</p></div>
      <div className="activation-equations"><MathFormula tex={info.formula} block /><MathFormula tex={info.derivative} block /></div>
    </div>
    <p className="panel-note">局部导数不是损失对参数的完整梯度，还需乘以上游梯度和对应输入。ReLU / Leaky ReLU 的零点均不可导，本实验分别选用 0 / 0.1 作为反传值；导数图在零点分开画出两侧分支。Softmax 需要多个分数共同归一化，在后续多分类实验中观察。</p>
    {compare && <div className="activation-table-wrap"><table className="activation-table"><caption>同一个 z = {format(z)}，比较输出、局部导数与用途</caption><thead><tr><th>函数</th><th>输出</th><th>导数 / 约定值</th><th>常见用途</th></tr></thead><tbody>{ACTIVATIONS.map(value => {
      const slope = activationDerivative(z, value);
      return <tr key={value} aria-current={value === activation ? "true" : undefined}><th>{activationInfo[value].label}</th><td>{format(activateNeuron(z, value), 4)}</td><td>{format(slope.derivative, 4)}{!slope.differentiable ? "（约定）" : ""}</td><td>{activationInfo[value].use}</td></tr>;
    })}</tbody></table></div>}
  </ExperimentPanel>;
}
