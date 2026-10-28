import { useEffect, useRef, useState } from "react";
import { ArrowRight, Award, Check, CheckCircle2, ChevronRight, CircleHelp, Flag, Lightbulb, Network, Pause, Play, RefreshCw, RotateCcw, ScanLine, Target, Trophy, X } from "lucide-react";
import { calculate, format } from "../../lib/api";
import { analyzeNetwork, readSavedNetwork } from "../../lib/networkBuilder";
import type { NetworkLayer } from "../../lib/networkBuilder";
import { ACTIVE_CHALLENGE_STORAGE_KEY, architectureChallenge, CHALLENGE_IDS, CHALLENGE_STORAGE_KEY, EDGE_INPUT, edgeChallenge, linearChallenge, readActiveChallenge, readChallengeProgress, xorChallenge } from "../../lib/challenges";
import type { ChallengeId } from "../../lib/challenges";
import type { XorSnapshot, XorStepResponse } from "../../lib/xor";
import "./challenge-hub.css";

const missions: { id: ChallengeId; name: string; tag: string; description: string; goal: string; icon: typeof Target; hints: string[] }[] = [
  { id: "linear-fit", name: "找到那条直线", tag: "01 / FOUNDATIONS", description: "让五个样本的误差接近零。", goal: "调整权重与偏置，使 5 个样本的均方误差 MSE ≤ 0.01。", icon: Target, hints: ["先观察 x = 0 时的目标值：这能帮助你确定偏置。", "偏置决定截距；权重决定 x 每增加 1 时，预测增加多少。", "目标样本来自 y = 2x + 0.5。先尝试 b = 0.5，再逐渐把 w 调向 2。"] },
  { id: "edge-detector", name: "唤醒边缘探测器", tag: "02 / CONVOLUTION", description: "让卷积核看见明暗交界。", goal: "在当前 8 × 8 输入上，均匀区域最大绝对响应 ≤ 0.05，边界区域最大绝对响应 ≥ 1。", icon: ScanLine, hints: ["明暗均匀时，边缘探测器应接近零。考虑让卷积核元素之和为零。", "左侧与右侧使用相反的权重，可以比较两侧亮度；响应的正负都算有效。", "可以把每行设为 [−1, 0, 1]。在当前输入上，均匀区域响应为 0，边界响应为 3。"] },
  { id: "cnn-architect", name: "设计你的第一张 CNN", tag: "03 / ARCHITECTURE", description: "把有效结构连成 10 类分类器。", goal: "在搭建器中保存包含 Conv2D、ReLU、MaxPool2D、Flatten、Linear 的有效网络，最终输出为 [10]。", icon: Network, hints: ["先到网络搭建器添加层，结构会保存在当前浏览器中。返回后点击重新检查。", "卷积和池化接收 [C,H,W]；Linear 接收向量。使用 Flatten 连接这两种形状。", "一个可行顺序是 Conv2D → ReLU → MaxPool2D → Flatten → Linear(10)。每层尺寸都必须有效。"] },
  { id: "xor-master", name: "让 XOR 全部答对", tag: "04 / REAL TRAINING", description: "用真实更新攻克非线性问题。", goal: "训练 2→4→1 网络，使四个 XOR 样本全部预测正确，且平均 BCE ≤ 0.15。", icon: Trophy, hints: ["先初始化网络，再运行 20 步，观察四个预测与损失。相同输入的目标为 0，不同输入为 1。", "默认种子 7、学习率 1 通常能有效学习。损失还高时，继续训练几组更新。", "全部分类正确后仍需检查 BCE。低置信度的正确预测，可能还没有达到 0.15 的损失目标。"] },
];

type Props = { onInteract?: () => void; onOpenTool?: (tool: "convolution" | "network-builder") => void };

export default function ChallengeHub({ onInteract, onOpenTool }: Props) {
  const [selected, setSelected] = useState<ChallengeId>(readActiveChallenge);
  const [completed, setCompleted] = useState<ChallengeId[]>(readChallengeProgress);
  const [storageWarning, setStorageWarning] = useState(false);
  const [hints, setHints] = useState<Record<ChallengeId, number>>({ "linear-fit": 0, "edge-detector": 0, "cnn-architect": 0, "xor-master": 0 });
  const [notice, setNotice] = useState<string | null>(null);
  const [resetConfirm, setResetConfirm] = useState(false);
  const [weight, setWeight] = useState(0);
  const [bias, setBias] = useState(0);
  const [kernel, setKernel] = useState([0, 0, 0, 0, 1, 0, 0, 0, 0]);
  const [architecture, setArchitecture] = useState<NetworkLayer[] | null>(() => readSavedNetwork());
  const [architectureChecked, setArchitectureChecked] = useState(false);
  const [snapshot, setSnapshot] = useState<XorSnapshot | null>(null);
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);
  const [seed, setSeed] = useState("7");
  const [rate, setRate] = useState(1);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const currentSnapshot = useRef<XorSnapshot | null>(null);
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const shouldRun = useRef(false);
  const mission = missions.find(item => item.id === selected)!;
  const linear = linearChallenge(weight, bias);
  const edge = edgeChallenge([kernel.slice(0, 3), kernel.slice(3, 6), kernel.slice(6, 9)]);
  const architectureAnalysis = architecture ? analyzeNetwork(architecture) : null;
  const architectureResult = architectureChallenge(architecture, architectureAnalysis);
  const livePassed = selected === "linear-fit" ? linear.passed : selected === "edge-detector" ? edge.passed : selected === "cnn-architect" ? architectureChecked && architectureResult.passed : xorChallenge(snapshot);

  useEffect(() => {
    try { localStorage.setItem(CHALLENGE_STORAGE_KEY, JSON.stringify({ version: 1, completed })); setStorageWarning(false); }
    catch { setStorageWarning(true); }
  }, [completed]);

  useEffect(() => {
    try { localStorage.setItem(ACTIVE_CHALLENGE_STORAGE_KEY, selected); }
    catch { setStorageWarning(true); }
  }, [selected]);

  useEffect(() => () => { version.current++; request.current?.abort(); shouldRun.current = false; }, []);

  function earn(id: ChallengeId) {
    setCompleted(previous => previous.includes(id) ? previous : [...previous, id]);
  }

  function stop() {
    version.current++;
    request.current?.abort();
    request.current = null;
    shouldRun.current = false;
    setRunning(false);
    setBusy(false);
  }

  function selectMission(id: ChallengeId) {
    if (id === "cnn-architect") { setArchitecture(readSavedNetwork()); setArchitectureChecked(false); }
    stop(); setSelected(id); setNotice(null); setResetConfirm(false); onInteract?.();
  }

  function verify() {
    onInteract?.();
    let passed = livePassed;
    if (selected === "cnn-architect") {
      const saved = readSavedNetwork();
      setArchitecture(saved); setArchitectureChecked(true);
      passed = architectureChallenge(saved, saved ? analyzeNetwork(saved) : null).passed;
    }
    if (passed) { earn(selected); setNotice("目标已达成，挑战徽章已记录。可以继续调整参数，观察结果如何变化。"); }
    else setNotice("还差一点。查看当前数值与目标条件，或者展开一条提示再试试。");
  }

  async function initialize() {
    if (!/^\d{1,4}$/.test(seed) || Number(seed) > 9999) { setTrainingError("随机种子需要是 0–9999 的整数。"); return; }
    stop();
    const token = version.current;
    const controller = new AbortController(); request.current = controller;
    setBusy(true); setTrainingError(null); setNotice(null); setSnapshot(null); currentSnapshot.current = null; setLossHistory([]);
    try {
      const result = await calculate<XorSnapshot>("/xor/init", { seed: Number(seed) }, controller.signal);
      if (token !== version.current) return;
      currentSnapshot.current = result; setSnapshot(result); setLossHistory([{ step: 0, loss: result.loss }]); onInteract?.();
    } catch (error) {
      if (token === version.current && !controller.signal.aborted) setTrainingError(error instanceof Error ? error.message : "无法初始化网络，请检查计算服务。");
    } finally { if (token === version.current) { request.current = null; setBusy(false); } }
  }

  async function trainBatch() {
    const source = currentSnapshot.current;
    if (!source || request.current || source.iteration >= 3000) return;
    const token = version.current;
    const controller = new AbortController(); request.current = controller;
    setBusy(true); setTrainingError(null);
    try {
      const result = await calculate<XorStepResponse>("/xor/step", { parameters: source.parameters, iteration: source.iteration, learning_rate: rate, steps: Math.min(20, 3000 - source.iteration), selected_sample: 0 }, controller.signal);
      if (token !== version.current) return;
      currentSnapshot.current = result.snapshot; setSnapshot(result.snapshot);
      setLossHistory(previous => [...previous, ...result.history.map(point => ({ step: point.step, loss: point.loss }))]); onInteract?.();
      if (xorChallenge(result.snapshot)) {
        shouldRun.current = false; setRunning(false); earn("xor-master"); setNotice("四个 XOR 样本全部正确，BCE 已达标。真实训练挑战完成！");
      } else if (result.snapshot.iteration >= 3000) {
        shouldRun.current = false; setRunning(false); setNotice("已达到本关 3000 步上限。可调整学习率或重新初始化再试。");
      }
    } catch (error) {
      if (token === version.current && !controller.signal.aborted) {
        setTrainingError(error instanceof Error ? error.message : "训练请求失败，请重试。"); shouldRun.current = false; setRunning(false);
      }
    } finally { if (token === version.current) { request.current = null; setBusy(false); } }
  }

  useEffect(() => {
    if (!running || busy || !snapshot || snapshot.iteration >= 3000) return;
    const timer = window.setTimeout(() => { if (shouldRun.current) void trainBatch(); }, 180);
    return () => window.clearTimeout(timer);
    // Each batch uses the last committed parameters; no overlapping requests.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, busy, snapshot, rate]);

  function resetCurrent() {
    stop(); setNotice(null); setTrainingError(null);
    if (selected === "linear-fit") { setWeight(0); setBias(0); }
    if (selected === "edge-detector") setKernel([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    if (selected === "cnn-architect") { setArchitecture(readSavedNetwork()); setArchitectureChecked(false); }
    if (selected === "xor-master") { setSnapshot(null); currentSnapshot.current = null; setLossHistory([]); setSeed("7"); setRate(1); }
    onInteract?.();
  }

  const currentHintCount = hints[selected];
  const lossMaximum = Math.max(0.2, ...lossHistory.map(point => point.loss));
  const lossPath = lossHistory.map((point, i) => `${i ? "L" : "M"}${8 + point.step / Math.max(1, snapshot?.iteration ?? 1) * 544},${110 - point.loss / lossMaximum * 98}`).join(" ");

  return <div className="challenge-hub">
    <div className="challenge-overview">
      <div className="challenge-overview-icon"><Trophy size={25} /></div>
      <div><span className="challenge-eyebrow">LEARN BY SOLVING</span><h2>把理解，变成可以验证的成果。</h2><p>每一枚徽章，都来自一次真实计算。挑战记录保存在当前浏览器。</p></div>
      <div className="challenge-overview-progress"><strong>{completed.length}<span> / 04</span></strong><span>关卡已达成</span></div>
    </div>
    <div className="challenge-missions" role="group" aria-label="选择实验挑战">
      {missions.map(item => { const Icon = item.icon; const done = completed.includes(item.id); return <button type="button" key={item.id} className={`challenge-mission ${selected === item.id ? "is-active" : ""} ${done ? "is-earned" : ""}`} onClick={() => selectMission(item.id)} aria-pressed={selected === item.id}>
        <div className="challenge-mission-top"><Icon size={19} /><span>{item.tag}</span>{done ? <CheckCircle2 size={15} className="challenge-earned-icon" /> : <ChevronRight size={14} />}</div><h3>{item.name}</h3><p>{item.description}</p><span className="challenge-mission-state">{done ? "已获得徽章" : selected === item.id ? "正在挑战" : "随时开始"}</span>
      </button>; })}
    </div>
    <section className="challenge-workspace" aria-labelledby="challenge-title">
      <div className="challenge-workspace-header"><div><span className="challenge-eyebrow">MISSION / {mission.tag}</span><h2 id="challenge-title">{mission.name}</h2></div><button className="challenge-quiet-button" type="button" onClick={resetCurrent}><RotateCcw size={14} />重置本关参数</button></div>
      <div className="challenge-goal"><Flag size={17} /><div><strong>过关目标</strong><p>{mission.goal}</p></div></div>

      {selected === "linear-fit" && <div className="challenge-task-grid">
        <div className="challenge-observation"><div className="challenge-panel-title"><Target size={15} /><h3>真实样本与当前预测</h3><span>y = 2x + 0.5</span></div><div className="challenge-table-scroll"><table className="challenge-table"><thead><tr><th>输入 x</th><th>目标 y</th><th>当前预测</th><th>误差</th></tr></thead><tbody>{[-2, -1, 0, 1, 2].map(x => <tr key={x}><td>{x}</td><td>{format(2 * x + .5)}</td><td>{format(weight * x + bias)}</td><td>{format(weight * x + bias - (2 * x + .5))}</td></tr>)}</tbody></table></div><div className="challenge-metrics"><div><span>均方误差 MSE</span><strong>{linear.mse.toFixed(4)}</strong></div><div><span>达标条件</span><strong className={linear.passed ? "challenge-green" : ""}>≤ 0.0100</strong></div></div></div>
        <aside className="challenge-controls"><h3>调整直线 ŷ = wx + b</h3><label className="challenge-slider"><span>权重 w <output>{weight.toFixed(2)}</output></span><input aria-label="挑战权重 w" type="range" min={-3} max={3} step={.05} value={weight} onChange={event => { setWeight(Number(event.target.value)); setNotice(null); onInteract?.(); }} /><small>−3<span>3</span></small></label><label className="challenge-slider"><span>偏置 b <output>{bias.toFixed(2)}</output></span><input aria-label="挑战偏置 b" type="range" min={-2} max={2} step={.05} value={bias} onChange={event => { setBias(Number(event.target.value)); setNotice(null); onInteract?.(); }} /><small>−2<span>2</span></small></label><p className="challenge-note">所有结果由上方 5 个样本即时计算。MSE = Σ(预测 − 目标)² / 5。</p></aside>
      </div>}

      {selected === "edge-detector" && <div className="challenge-task-grid challenge-edge-grid">
        <div className="challenge-observation"><div className="challenge-panel-title"><ScanLine size={15} /><h3>固定输入 → 实际特征图</h3></div><div className="challenge-matrix-pair"><div><span className="challenge-small-label">8 × 8 明暗边界</span><div className="challenge-image-grid" aria-label="固定输入，左半黑色，右半白色">{EDGE_INPUT.flat().map((value, index) => <span key={index} style={{ background: value ? "#c5e9db" : "#06110d" }} />)}</div></div><ArrowRight size={18} /><div><span className="challenge-small-label">6 × 6 互相关输出</span><div className="challenge-output-grid" aria-label="卷积核计算后的特征图">{edge.output.flat().map((value, index) => <span key={index} title={`行 ${Math.floor(index / 6) + 1}，列 ${index % 6 + 1}：${value.toFixed(3)}`} className={index % 6 === 2 || index % 6 === 3 ? "is-edge" : ""}>{format(value, 2)}</span>)}</div></div></div><div className="challenge-metrics"><div><span>均匀区域最大 |响应|</span><strong className={edge.uniformResponse <= .05 ? "challenge-green" : ""}>{edge.uniformResponse.toFixed(3)}</strong><small>目标 ≤ 0.05</small></div><div><span>边界区域最大 |响应|</span><strong className={edge.edgeResponse >= 1 ? "challenge-green" : ""}>{edge.edgeResponse.toFixed(3)}</strong><small>目标 ≥ 1</small></div></div><p className="challenge-note">输出第 3、4 列是边界区域；其余列是均匀区域。核不翻转，stride = 1，padding = 0；本关不使用 ReLU。</p></div>
        <aside className="challenge-controls"><h3>编辑 3 × 3 卷积核</h3><div className="challenge-kernel">{kernel.map((value, index) => <input key={index} aria-label={`挑战卷积核第 ${Math.floor(index / 3) + 1} 行第 ${index % 3 + 1} 列`} type="number" min={-3} max={3} step={.1} value={value} onChange={event => { const parsed = Number(event.target.value); const next = Number.isFinite(parsed) ? Math.max(-3, Math.min(3, parsed)) : 0; setKernel(previous => previous.map((old, i) => i === index ? next : old)); setNotice(null); onInteract?.(); }} />)}</div><div className="challenge-preset-buttons"><button type="button" onClick={() => { setKernel(Array(9).fill(1 / 9)); setNotice(null); }}>均值核</button><button type="button" onClick={() => { setKernel(Array(9).fill(0)); setNotice(null); }}>清空</button></div><p className="challenge-note">允许 −3 到 3 的数值。尝试让左右权重相反；有效边缘响应可以为正，也可以为负。</p>{onOpenTool && <button className="challenge-link-button" type="button" onClick={() => onOpenTool("convolution")}>去卷积实验观察计算<ArrowRight size={13} /></button>}</aside>
      </div>}

      {selected === "cnn-architect" && <div className="challenge-architecture"><div><h3>读取你的网络搭建器存档</h3><p className="challenge-note">从当前浏览器的搭建器配置检查尺寸、层类型与最终输出。需要先在搭建器里添加并保存结构。</p><div className="challenge-layer-checks">{architectureResult.requirements.map(item => <span key={item.type} className={item.met ? "is-met" : ""}>{item.met ? <Check size={13} /> : <X size={13} />}{({ conv: "Conv2D", relu: "ReLU", pool: "MaxPool2D", flatten: "Flatten", linear: "Linear" })[item.type]}</span>)}</div>{architecture ? <><div className="challenge-architecture-chain">{architecture.length ? architecture.map((layer, index) => <span key={layer.id}>{index > 0 && <ArrowRight size={11} />}<span>{({ conv: "Conv2D", relu: "ReLU", pool: "MaxPool2D", flatten: "Flatten", linear: "Linear" })[layer.type]}</span></span>) : <span>当前保存的结构为空</span>}</div><div className="challenge-metrics"><div><span>最终输出（不含 batch）</span><strong>{architectureAnalysis?.output ? `[${architectureAnalysis.output.join(", ")}]` : "无法计算"}</strong></div><div><span>结构尺寸</span><strong className={architectureAnalysis?.valid ? "challenge-green" : ""}>{architectureAnalysis?.valid ? "有效 · 10 类" : "待修正"}</strong></div></div>{Boolean(architectureAnalysis?.errors.length) && <ul className="challenge-errors">{architectureAnalysis!.errors.map((error, index) => <li key={`${index}-${error}`}>{error}</li>)}</ul>}</> : <div className="challenge-empty"><Network size={32} /><p>还没有可读取的有效存档。去搭建器开始设计，再回来检查。</p></div>}</div><div className="challenge-architecture-actions">{onOpenTool && <button type="button" className="challenge-primary-button" onClick={() => onOpenTool("network-builder")}>前往网络搭建器<ArrowRight size={15} /></button>}<button className="challenge-secondary-button" type="button" onClick={verify}><RefreshCw size={14} />重新检查存档</button><p className="challenge-note">通过关卡代表结构尺寸有效；模型准确率需要在真实数据上训练和评估。</p></div></div>}

      {selected === "xor-master" && <div className="challenge-task-grid"><div className="challenge-observation"><div className="challenge-panel-title"><Network size={15} /><h3>四个样本 · 真实模型预测</h3><span>2 → 4 → 1</span></div><div className="challenge-xor-samples">{(snapshot?.samples ?? [0, 1, 2, 3].map(index => ({ inputs: [Math.floor(index / 2), index % 2], target: index === 1 || index === 2 ? 1 : 0, prediction: null, probability: null }))).map((sample, index) => <div key={index} className={sample.prediction === null ? "" : sample.prediction === sample.target ? "is-correct" : "is-wrong"}><span>{sample.inputs.join(" ⊕ ")}</span><strong>{sample.probability === null ? "—" : sample.probability.toFixed(3)}</strong><span>目标 {sample.target} · 预测 {sample.prediction ?? "—"}</span></div>)}</div><div className="challenge-metrics"><div><span>平均 BCE</span><strong>{snapshot ? snapshot.loss.toFixed(4) : "—"}</strong><small>目标 ≤ 0.15</small></div><div><span>正确样本</span><strong>{snapshot ? `${snapshot.samples.filter(sample => sample.prediction === sample.target).length} / 4` : "—"}</strong><small>目标 4 / 4</small></div><div><span>真实更新次数</span><strong>{snapshot?.iteration ?? 0}</strong><small>本关最多 3000 步</small></div></div>{lossHistory.length > 1 && <div className="challenge-loss-chart"><div className="challenge-small-label">BCE / 参数更新 step</div><svg viewBox="0 0 560 125" role="img" aria-label={`真实训练损失曲线，当前 ${snapshot?.iteration ?? 0} 步，损失 ${snapshot?.loss.toFixed(4)}`}><path d="M8 12V110H552" stroke="#294535" fill="none" /><path d={lossPath} fill="none" stroke="#a9de85" strokeWidth="2" /><line x1="8" x2="552" y1={110 - .15 / lossMaximum * 98} y2={110 - .15 / lossMaximum * 98} stroke="#5fa88e" strokeDasharray="4 5" /></svg><div className="challenge-chart-caption"><span>0</span><span>虚线为目标 BCE = 0.15</span><span>{snapshot?.iteration}</span></div></div>}<p className="challenge-note">概率经过 sigmoid，分类阈值为 0.5。每一批请求使用全部四个样本，实际执行 20 次参数更新。</p></div><aside className="challenge-controls"><h3>训练控制</h3><label className="challenge-field">初始化随机种子<input type="number" aria-label="挑战 XOR 随机种子" min={0} max={9999} step={1} value={seed} disabled={busy || running} onChange={event => setSeed(event.target.value)} /></label><label className="challenge-slider"><span>学习率 η <output>{rate.toFixed(2)}</output></span><input aria-label="挑战 XOR 学习率" type="range" min={.05} max={2} step={.05} value={rate} disabled={busy || running} onChange={event => setRate(Number(event.target.value))} /><small>0.05<span>2.00</span></small></label><button className="challenge-secondary-button" type="button" disabled={busy || running} onClick={() => void initialize()}><RotateCcw size={14} />{snapshot ? "重新初始化网络" : "初始化网络"}</button><div className="challenge-training-buttons"><button className="challenge-secondary-button" type="button" disabled={!snapshot || busy || running || snapshot.iteration >= 3000} onClick={() => void trainBatch()}>运行 20 步</button>{running ? <button className="challenge-primary-button" type="button" onClick={stop}><Pause size={14} />停止训练</button> : <button className="challenge-primary-button" type="button" disabled={!snapshot || busy || snapshot.iteration >= 3000 || xorChallenge(snapshot)} onClick={() => { shouldRun.current = true; setRunning(true); }}><Play size={14} />自动训练</button>}</div><p className="challenge-note">此关调用本机 NumPy API，服务不可用时不能训练。自动训练会在达标或达到步数上限时停止。</p>{busy && <span className="challenge-training-state" role="status">正在计算真实网络…</span>}{trainingError && <div className="challenge-error" role="alert">{trainingError}</div>}</aside></div>}

      <div className="challenge-verification"><div className={`challenge-current-status ${livePassed ? "is-met" : ""}`}>{livePassed ? <CheckCircle2 size={17} /> : <CircleHelp size={17} />}<div><strong>{livePassed ? "当前结果已满足目标" : "当前结果尚未满足全部目标"}</strong><span>{completed.includes(selected) ? "已拥有本关历史徽章；当前结果仍独立检查。" : "点击检查后，系统按真实数值记录达成结果。"}</span></div></div><button className="challenge-primary-button" type="button" onClick={verify} disabled={selected === "xor-master" && (busy || !snapshot)}><Flag size={14} />{selected === "cnn-architect" ? "检查网络并领取徽章" : "检查目标并领取徽章"}</button></div>
      {notice && <div className={`challenge-notice ${livePassed ? "is-success" : ""}`} role="status">{livePassed ? <Award size={17} /> : <Lightbulb size={17} />}<p>{notice}</p></div>}
      <div className="challenge-hints"><div><Lightbulb size={16} /><h3>需要一点线索？</h3><span>{currentHintCount} / {mission.hints.length}</span></div>{currentHintCount > 0 && <ol>{mission.hints.slice(0, currentHintCount).map(hint => <li key={hint}>{hint}</li>)}</ol>}<button type="button" className="challenge-link-button" disabled={currentHintCount >= mission.hints.length} onClick={() => setHints(previous => ({ ...previous, [selected]: Math.min(mission.hints.length, previous[selected] + 1) }))}>{currentHintCount === 0 ? "展开第一条提示" : currentHintCount >= mission.hints.length ? "所有提示已展开" : "再看一条提示"}<ChevronRight size={13} /></button></div>
    </section>
    <div className="challenge-footer"><span>{storageWarning ? "当前浏览器无法保存挑战记录。" : "挑战徽章独立保存，不会更改 13 节课程的完成进度。"}</span>{resetConfirm ? <div className="challenge-reset-confirm"><span>清空全部挑战徽章？</span><button type="button" onClick={() => { stop(); setCompleted([]); setResetConfirm(false); setNotice("挑战徽章已清空，课程学习进度保持不变。"); }}>确认清空</button><button type="button" onClick={() => setResetConfirm(false)}>取消</button></div> : <button type="button" className="challenge-quiet-button" disabled={!completed.length} onClick={() => setResetConfirm(true)}><RefreshCw size={12} />清空挑战记录</button>}</div>
    <div className="challenge-achievements" aria-label="挑战徽章收藏">{missions.map(item => <div key={item.id} className={completed.includes(item.id) ? "is-earned" : ""}><Award size={20} /><span>{item.name}</span><small>{completed.includes(item.id) ? "已达成" : "等待解锁"}</small></div>)}</div>
    <span className="challenge-sr-only">共 {CHALLENGE_IDS.length} 个挑战，已完成 {completed.length} 个。</span>
  </div>;
}
