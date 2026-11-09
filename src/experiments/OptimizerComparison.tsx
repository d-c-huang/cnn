import { useEffect, useRef, useState } from "react";
import { Download, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { Chart, ErrorMessage, ExperimentPanel, Result, Slider } from "../components/ui";
import { MathFormula } from "../components/MathFormula";
import { calculate, format } from "../lib/api";
import { comparisonCsv, comparisonRow, optimizerInfo, optimizerNames } from "../lib/optimizers";
import type { ComparisonResult, Objective, Optimizer, OptimizerConfig, Vector } from "../lib/optimizers";
import "./optimizers.css";

const initialRates = () => ({ sgd: 0.1, momentum: 0.1, adagrad: 0.1, rmsprop: 0.1, adam: 0.1 });

export default function OptimizerComparison({ onInteract }: { onInteract: () => void }) {
  const [objective, setObjective] = useState<Objective>("quadratic");
  const [start, setStart] = useState<Vector>([-2, 2]);
  const [rate, setRate] = useState(0.1);
  const [steps, setSteps] = useState(80);
  const [individual, setIndividual] = useState(false);
  const [rates, setRates] = useState(initialRates);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [selected, setSelected] = useState<Optimizer>("sgd");
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logScale, setLogScale] = useState(true);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const lastStep = result ? Math.max(...result.runs.map(run => run.history.length - 1)) : 0;
  useEffect(() => {
    if (!playing || !result) return;
    if (cursor >= lastStep) { setPlaying(false); return; }
    const timer = window.setTimeout(() => setCursor(value => Math.min(value + 1, lastStep)), 180);
    return () => window.clearTimeout(timer);
  }, [playing, cursor, lastStep, result]);

  function invalidate() {
    request.current?.abort(); request.current = null;
    setBusy(false); setResult(null); setPlaying(false); setCursor(0); setError(null);
    onInteract();
  }
  function preset(next: Objective, learningRate: number) {
    invalidate(); setObjective(next); setStart([-2, 2]); setRate(learningRate);
    setSteps(next === "valley" ? 120 : 80); setIndividual(false); setRates(initialRates());
  }
  async function runComparison() {
    if (request.current) return;
    const controller = new AbortController(); request.current = controller;
    setBusy(true); setError(null); setPlaying(false);
    const config: OptimizerConfig = { objective, start, steps, learning_rate: rate, rates: individual ? rates : {} };
    try {
      const response = await calculate<ComparisonResult>("/optimizers/compare", config, controller.signal);
      if (controller.signal.aborted || request.current !== controller) return;
      setResult(response); setCursor(Math.max(...response.runs.map(run => run.history.length - 1))); onInteract();
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "优化器对比计算失败。");
    } finally {
      if (request.current === controller) { request.current = null; setBusy(false); }
    }
  }
  function download() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([comparisonCsv(result)], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "zhiwei-optimizer-comparison.csv"; anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const run = result?.runs.find(value => value.optimizer === selected);
  const row = run ? comparisonRow(run, cursor) : null;
  const previous = run && row ? comparisonRow(run, Math.max(0, row.step - 1)) : null;
  const info = optimizerInfo[selected];
  const ordered = result ? [...result.runs].sort((a, b) => Number(b.optimizer === selected) - Number(a.optimizer === selected)) : [];
  const trace = (name: Optimizer) => result?.runs.find(value => value.optimizer === name)?.history.filter(value => value.step <= cursor) ?? [];
  const vector = (value: Vector | null) => value ? (objective === "quadratic" ? format(value[0], 5) : `(${value.map(x => format(x, 5)).join(", ")})`) : "—";
  const contours = objective === "valley" ? [1, 5, 20, 80].map(level => ({
    points: Array.from({ length: 101 }, (_, i) => ({ x: 2 + Math.sqrt(2 * level) * Math.cos(i * Math.PI / 50), y: -1 + Math.sqrt(level / 10) * Math.sin(i * Math.PI / 50) })),
    color: "#294641", dashed: true,
  })) : [];
  function trajectoryDomain(axis: 0 | 1): [number, number] {
    const values = ordered.flatMap(item => trace(item.optimizer).map(value => value.position[axis]));
    values.push(axis === 0 ? 2 : -1, start[axis]);
    const min = Math.min(...values), max = Math.max(...values);
    const padding = Math.max(0.5, (max - min) * 0.1);
    return [min - padding, max + padding];
  }

  return <section id="optimizer-comparison" className="optimizer-comparison" aria-label="五种优化器对比">
    <ExperimentPanel title="相同起点，五种不同的更新方式" eyebrow="OPTIMIZER COMPARISON">
      <p className="optimizer-intro">比较 SGD、Momentum、AdaGrad、RMSProp 与 Adam。先用相同学习率看更新规则，再分别调参，观察结论是否改变。</p>
      <fieldset className="optimizer-config" disabled={busy}>
        <legend>设置共同的实验条件</legend>
        <label className="optimizer-select">损失函数<select value={objective} onChange={event => { invalidate(); setObjective(event.target.value as Objective); }}>
          <option value="quadratic">一维平方损失 · 沿用基础实验</option><option value="valley">二维狭长谷底 · 比较不同方向</option>
        </select></label>
        <div className="optimizer-presets" role="group" aria-label="优化器对比预设">
          <button type="button" onClick={() => preset("quadratic", 0.1)}>一维 · η = 0.1</button>
          <button type="button" onClick={() => preset("valley", 0.03)}>狭长谷底 · η = 0.03</button>
          <button type="button" onClick={() => preset("quadratic", 0.55)}>大学习率 · η = 0.55</button>
        </div>
        <div className="optimizer-fields">
          <Slider label={objective === "quadratic" ? "初始权重 w₀" : "初始位置 u₀"} value={start[0]} min={-4} max={6} step={0.1} onChange={value => { invalidate(); setStart([value, start[1]]); }} />
          {objective === "valley" && <Slider label="初始位置 v₀" value={start[1]} min={-4} max={4} step={0.1} onChange={value => { invalidate(); setStart([start[0], value]); }} />}
          {!individual && <Slider label="共同学习率 η" value={rate} min={0.001} max={1} step={0.001} onChange={value => { invalidate(); setRate(value); }} />}
          <Slider label="更新步数预算" value={steps} min={10} max={200} step={10} onChange={value => { invalidate(); setSteps(value); }} />
        </div>
        <label className="normalize-toggle"><input type="checkbox" checked={individual} onChange={event => { invalidate(); setIndividual(event.target.checked); setRates(Object.fromEntries(optimizerNames.map(name => [name, rate])) as Record<Optimizer, number>); }} />分别设置各优化器的学习率</label>
        {individual && <div className="optimizer-rate-grid">{optimizerNames.map(name => <Slider key={name} label={optimizerInfo[name].label} value={rates[name]} min={0.001} max={1} step={0.001} onChange={value => { invalidate(); setRates(previousRates => ({ ...previousRates, [name]: value })); }} />)}</div>}
      </fieldset>
      <div className="optimizer-objective"><MathFormula tex={objective === "quadratic" ? String.raw`L(w)=2(w-2)^2,\quad g=4(w-2)` : String.raw`L(u,v)=\tfrac12[(u-2)^2+20(v+1)^2],\quad\nabla L=(u-2,20(v+1))`} block />
        <p>{objective === "quadratic" ? "最低点 w = 2。基础 SGD 在 0 < η < 0.5 时收敛；这个范围不适用于所有函数或优化器。" : "最低点 (2, −1)。v 方向曲率是 u 方向的 20 倍。基础 SGD 的稳定学习率范围为 0 < η < 0.1。"}</p>
      </div>
      <div className="optimizer-actions">
        <button className="primary-button" disabled={busy} onClick={() => void runComparison()}><Play size={16} />{busy ? "正在计算真实轨迹…" : result ? "重新运行对比" : "运行五种优化器对比"}</button>
        <button className="secondary-button" onClick={() => { preset("quadratic", 0.1); setSelected("sgd"); setLogScale(true); }}><RotateCcw size={15} />重置对比</button>
        <button className="text-button" disabled={!result || busy} onClick={download}><Download size={15} />导出完整 CSV</button>
      </div>
      <ErrorMessage error={error} />
      <p className="panel-note">后端 NumPy 按各自更新规则计算全部轨迹，回放只展示已计算记录。五种方法使用相同目标、起点和步数预算，内部状态均从零开始；全部使用精确全量梯度，没有随机批次。相同学习率不代表各方法都已调到最佳，也不能据此判断实际神经网络训练的普遍优劣。</p>
    </ExperimentPanel>

    {result ? <ExperimentPanel title="对比损失变化与参数轨迹" eyebrow="COMPUTED TRAJECTORIES">
      <div className="optimizer-playback">
        <button className="secondary-button" onClick={() => { if (playing) setPlaying(false); else { if (cursor >= lastStep) setCursor(0); setPlaying(true); } }}>
          {playing ? <Pause size={15} /> : <Play size={15} />}{playing ? "暂停回放" : cursor >= lastStep ? "从头播放" : "继续回放"}</button>
        <button className="secondary-button" disabled={playing || cursor >= lastStep} onClick={() => setCursor(value => value + 1)}><SkipForward size={15} />查看下一步</button>
        <label>查看第 {cursor} / {lastStep} 步<input aria-label="优化器回放步数" type="range" min={0} max={lastStep} step={1} value={cursor} onChange={event => { setPlaying(false); setCursor(Number(event.target.value)); }} /></label>
        <label className="normalize-toggle"><input type="checkbox" checked={logScale} onChange={event => setLogScale(event.target.checked)} />损失用 log₁₀ 显示</label>
      </div>
      <div className="optimizer-legend" role="group" aria-label="选择重点观察的优化器">{optimizerNames.map(name => <button key={name} aria-pressed={selected === name} onClick={() => setSelected(name)}><i style={{ background: optimizerInfo[name].color }} />{optimizerInfo[name].label}</button>)}</div>
      <div className="optimizer-charts"><div><h3>损失随更新步数变化</h3><Chart height={270}
        series={ordered.map(item => ({ points: trace(item.optimizer).map(value => ({ x: value.step, y: logScale ? Math.log10(Math.max(value.loss, 1e-12)) : value.loss })), color: optimizerInfo[item.optimizer].color, dashed: item.optimizer !== selected }))}
        markers={ordered.map(item => { const current = comparisonRow(item, cursor); return { x: current.step, y: logScale ? Math.log10(Math.max(current.loss, 1e-12)) : current.loss, color: optimizerInfo[item.optimizer].color, label: optimizerInfo[item.optimizer].label }; })}
        xDomain={[0, result.config.steps]} xLabel="实际更新步数" yLabel={logScale ? "log₁₀(L)，下限 −12" : "损失 L"} ariaLabel="五种优化器的后端计算损失轨迹" />
        <p className="panel-note">{logScale ? "−2 表示损失 0.01；−6 表示 0.000001。仅绘图时将小于 10⁻¹² 的值放在 −12，表格保留真实值。" : "线性坐标便于看绝对差距；某条轨迹发散时，可切回对数坐标观察其他方法。"}</p></div>
        <div><h3>{objective === "quadratic" ? "权重怎样靠近最低点" : "在狭长谷底中怎样移动"}</h3><Chart height={270}
          series={[...ordered.map(item => ({ points: trace(item.optimizer).map(value => ({ x: objective === "quadratic" ? value.step : value.position[0], y: objective === "quadratic" ? value.position[0] : value.position[1] })), color: optimizerInfo[item.optimizer].color, dashed: item.optimizer !== selected })), ...contours,
            ...(objective === "quadratic" ? [{ points: [{ x: 0, y: 2 }, { x: result.config.steps, y: 2 }], color: "#647d70", dashed: true }] : [])]}
          markers={ordered.map(item => { const current = comparisonRow(item, cursor); return { x: objective === "quadratic" ? current.step : current.position[0], y: objective === "quadratic" ? current.position[0] : current.position[1], label: optimizerInfo[item.optimizer].label, color: optimizerInfo[item.optimizer].color }; }).concat(objective === "valley" ? [{ x: 2, y: -1, label: "最低点", color: "#ffffff" }] : [])}
          xDomain={objective === "quadratic" ? [0, result.config.steps] : trajectoryDomain(0)} yDomain={objective === "valley" ? trajectoryDomain(1) : undefined} xLabel={objective === "quadratic" ? "实际更新步数" : "参数 u"} yLabel={objective === "quadratic" ? "权重 w" : "参数 v"} ariaLabel="五种优化器的真实参数轨迹" />
          <p className="panel-note">{objective === "quadratic" ? "水平参考线为最优权重 w = 2。越过参考线不一定意味着发散，应结合后续损失观察。" : "灰色椭圆表示损失等高线 L = 1、5、20、80；白点为最低点。图中两个坐标轴独立缩放。"}</p></div></div>
      <div className="optimizer-table-wrap"><table className="optimizer-table"><caption>第 {cursor} 步的记录 · 点击上方名称查看更新细节</caption><thead><tr><th>优化器</th><th>学习率</th><th>实际步数</th><th>当前参数</th><th>当前损失</th><th>首次 L ≤ 10⁻⁶</th><th>状态</th></tr></thead><tbody>{result.runs.map(item => {
        const current = comparisonRow(item, cursor);
        const limited = item.status === "limit" && cursor >= item.history.length - 1;
        return <tr key={item.optimizer} aria-current={selected === item.optimizer ? "true" : undefined}><th>{optimizerInfo[item.optimizer].label}</th><td>{format(item.learning_rate)}</td><td>{current.step}</td><td>{vector(current.position)}</td><td>{format(current.loss, 6)}</td><td>{item.first_hit !== null && item.first_hit <= cursor ? `第 ${item.first_hit} 步` : "尚未达到"}</td><td>{limited ? "数值上限，已停止" : current.loss <= result.threshold ? "当前达标" : "当前未达标"}</td></tr>;
      })}</tbody></table></div>
      <p className="panel-note">首次达到阈值后仍继续计算，便于观察动量越过最低点或再次振荡。“首次达标”不等于后续始终达标。损失超过 10¹² 或参数绝对值超过 10⁶ 时停止该方法，其他方法继续。</p>
    </ExperimentPanel> : <div className="optimizer-empty"><strong>先运行一次共同条件下的对比</strong><p>随后查看五条真实轨迹，拖动步数滑块，逐步比较每种方法的参数变化。</p></div>}

    <ExperimentPanel title="梯度相同，更新规则为什么不同？" eyebrow="UPDATE RULES">
      <div className="optimizer-legend" role="group" aria-label="选择优化器原理">{optimizerNames.map(name => <button key={name} aria-pressed={selected === name} onClick={() => { setSelected(name); onInteract(); }}><i style={{ background: optimizerInfo[name].color }} />{optimizerInfo[name].label}</button>)}</div>
      <div className="optimizer-explanation"><div><h3>{info.label}</h3><p>{info.description}</p><p className="panel-note">记忆：{info.memory}。公式中的平方、平方根和除法均按坐标计算；ε = 10⁻⁸，放在平方根外。gₜ 使用更新前的参数计算。</p></div><MathFormula tex={info.formula} block /></div>
      {row && previous && <><div className="optimizer-update-grid">
        <Result label={`第 ${row.step} 步 · 更新前参数`} value={vector(previous.position)} />
        <Result label="本次使用的梯度 gₜ" value={vector(row.gradient)} hint={row.step === 0 ? "第 0 步为初始状态，尚未更新" : "来自更新前的参数"} />
        <Result label="实际更新量 Δθ" value={vector(row.update)} hint="更新后参数 = 更新前参数 + Δθ" />
        <Result label="更新后参数" value={vector(row.position)} />
      </div><p className="panel-note">{selected === "momentum" ? `动量缓冲 b = ${vector(row.first)}` : selected === "adagrad" ? `累计梯度平方 s = ${vector(row.second)}` : selected === "rmsprop" ? `滑动平均 v = ${vector(row.second)}` : selected === "adam" ? `未校正 m = ${vector(row.first)}；未校正 v = ${vector(row.second)}。实际更新使用公式中的偏差校正值。` : "SGD 不保存梯度历史；这里的 SGD 指基础更新规则，实验使用全量梯度。"}</p></>}
      <div className="optimizer-takeaways"><div><h3>看速度，也看稳定性</h3><p>先观察损失与参数是否稳定接近最低点，再比较达到同一阈值需要多少步。一步损失增加不一定是发散，短暂触及最低点也不代表已经稳定。</p></div><div><h3>步数不是耗时</h3><p>这里比较参数更新次数。不同优化器保存的状态和每步计算量不同，本实验不把浏览器播放速度当成训练耗时。</p></div></div>
    </ExperimentPanel>
  </section>;
}
