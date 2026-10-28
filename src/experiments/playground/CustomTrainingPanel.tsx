import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, CheckCircle2, Cpu, Download, FlaskConical, Play, RefreshCw, Square, TriangleAlert } from "lucide-react";
import { ErrorMessage } from "../../components/ui";
import { format } from "../../lib/api";
import { activeStatuses, mnistRequest, modelLabel, selectedRunId, selectRunId, statusLabels, storeMnistConfig } from "../../lib/mnist";
import type { MnistStatus, Run, RunConfig } from "../../lib/mnist";
import { analyzeNetwork } from "../../lib/networkBuilder";
import type { NetworkLayer } from "../../lib/networkBuilder";
import { preferInferenceRun } from "../../lib/inference";
import { useMnistResource } from "../../lib/useMnist";
import TrainingCurves from "../mnist/TrainingCurves";
import "./custom-training.css";

type LessonTarget = "mnist-train" | "mnist-evaluation" | "handwriting";
type Props = { layers: NetworkLayer[]; onOpenLesson?: (id: LessonTarget) => void };
const architectureKey = (layers: NetworkLayer[] | undefined) => JSON.stringify(layers?.map(layer => layer.type === "conv" ? [layer.type, layer.outChannels, layer.kernel, layer.stride, layer.padding] : layer.type === "linear" ? [layer.type, layer.units] : [layer.type]) ?? null);
const dataMatch = (a: RunConfig, b: RunConfig) => a.seed === b.seed && a.validation_ratio === b.validation_ratio && a.train_limit === b.train_limit;

export default function CustomTrainingPanel({ layers, onOpenLesson }: Props) {
  const analysis = useMemo(() => analyzeNetwork(layers), [layers]);
  const [config, setConfig] = useState<RunConfig>({ model: "custom", seed: 7, validation_ratio: .1, train_limit: 6000, epochs: 1, batch_size: 128, learning_rate: .001 });
  const [chosenId, setChosenId] = useState<string | null>(selectedRunId);
  const [freshRun, setFreshRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState("准备数据后，点击开始。提交时会保存当前网络结构与训练配置的独立快照。");
  const status = useMnistResource<MnistStatus>("/status");
  const records = useMnistResource<{ runs: Run[] }>("/runs");
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const runs = records.data?.runs ?? [];
  const customRuns = runs.filter(item => item.config.model === "custom");
  const run = customRuns.find(item => item.id === chosenId) ?? (freshRun?.id === chosenId ? freshRun : null) ?? customRuns.find(item => activeStatuses.has(item.status)) ?? customRuns[0] ?? null;
  const anyActive = Boolean(status.data?.active_run_id || runs.some(item => activeStatuses.has(item.status)) || (freshRun && activeStatuses.has(freshRun.status) && !runs.some(item => item.id === freshRun.id)));
  const currentKey = architectureKey(layers);
  const changedSinceRun = Boolean(run && architectureKey(run.config.architecture) !== currentKey);
  const comparison = run ? runs.filter(item => item.id !== run.id && item.config.model !== "custom" && item.status === "completed" && item.evaluation && dataMatch(item.config, run.config)) : [];
  const excluded = run ? runs.filter(item => item.config.model !== "custom" && item.status === "completed" && item.evaluation && !dataMatch(item.config, run.config)).length : 0;
  const latest = run?.history.at(-1);
  const progress = run ? (run.status === "completed" ? 100 : Math.max(0, Math.min(99, (Math.max(0, run.epoch - 1) + (run.train_samples ? Math.min(1, run.processed_samples / run.train_samples) : 0)) / run.config.epochs * 100))) : 0;

  useEffect(() => () => { version.current++; request.current?.abort(); }, []);
  function chooseRun(id: string) { setChosenId(id); selectRunId(id); setFreshRun(null); setError(null); }
  function change(next: Partial<RunConfig>) { setConfig(previous => ({ ...previous, ...next })); }
  function openLesson(id: LessonTarget) {
    if (run) { selectRunId(run.id); if (id === "handwriting") preferInferenceRun(run.id); }
    if (onOpenLesson) onOpenLesson(id); else window.location.hash = id;
  }
  function openBaseline() {
    if (run) { const { architecture: _architecture, ...baseline } = run.config; storeMnistConfig({ ...baseline, model: "mlp" }); }
    openLesson("mnist-train");
  }
  async function operate(action: "start" | "cancel" | "prepare") {
    if (request.current || (action === "start" && !analysis.valid)) return;
    const controller = new AbortController(); request.current = controller;
    const token = ++version.current;
    setBusy(true); setError(null);
    try {
      if (action === "start") {
        const snapshot: RunConfig = { ...config, model: "custom", architecture: layers.map(layer => ({ ...layer })) };
        const result = await mnistRequest<Run>("/runs", { body: snapshot, signal: controller.signal });
        if (token !== version.current) return;
        setFreshRun(result); setChosenId(result.id); selectRunId(result.id);
        setNotice("自定义训练已提交。后台使用本次提交的结构快照，离开页面后仍会继续运行。");
      } else if (action === "prepare") {
        await mnistRequest<MnistStatus>("/prepare", { body: {}, signal: controller.signal });
        if (token !== version.current) return;
        setNotice("正在准备官方 MNIST；首次使用会下载数据，已有缓存会直接校验加载。");
      } else if (run) {
        const result = await mnistRequest<Run>(`/runs/${run.id}/cancel`, { body: {}, signal: controller.signal });
        if (token !== version.current) return;
        setFreshRun(result); setNotice("取消请求已发送，训练将在批次边界停止，保留已完成记录与最佳权重。");
      }
      if (token === version.current) { void records.refresh(); void status.refresh(); }
    } catch (cause) {
      if (token === version.current && !controller.signal.aborted) setError(cause instanceof Error ? cause.message : "训练操作失败，请重试。");
    } finally { if (token === version.current) { request.current = null; setBusy(false); } }
  }

  return <section className="custom-training" aria-labelledby="custom-training-title">
    <div className="ct-heading"><div><span className="ct-kicker"><FlaskConical size={15} /> BUILD → TRAIN → EVALUATE → USE</span><h2 id="custom-training-title">让你搭建的网络，开始真正学习。</h2><p>使用当前结构在官方 MNIST 上训练，按验证损失保存最佳权重，再进行独立测试与手写识别。</p></div><span className="ct-runtime"><Cpu size={16} />PyTorch · CPU</span></div>
    <div className="ct-design-summary"><span><strong>{layers.length}</strong> 层结构</span><span><strong>{analysis.totalParameters.toLocaleString("zh-CN")}</strong> 可训练参数</span><span className={analysis.valid ? "is-valid" : "is-invalid"}>{analysis.valid ? <CheckCircle2 size={16} /> : <TriangleAlert size={16} />}{analysis.valid ? "输出 [10] · 结构检查通过" : "先修复上方结构"}</span></div>
    {!analysis.valid && <div className="ct-warning">{analysis.errors.join(" ")}</div>}
    <div className="ct-config-grid">
      <label>训练轮数<select aria-label="自定义训练轮数" value={config.epochs} onChange={event => change({ epochs: Number(event.target.value) })}>{[1, 2, 3, 5, 10, 20].map(value => <option key={value} value={value}>{value} 轮</option>)}</select></label>
      <label>训练数据规模<select aria-label="自定义训练数据规模" value={config.train_limit ?? "full"} onChange={event => change({ train_limit: event.target.value === "full" ? null : Number(event.target.value) })}><option value={6000}>快速实验 · 6,000 张</option><option value={12000}>12,000 张</option><option value="full">全部训练样本</option></select></label>
      <label>批次大小<select aria-label="自定义训练批次大小" value={config.batch_size} onChange={event => change({ batch_size: Number(event.target.value) })}>{[32, 64, 128, 256].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <label>学习率<select aria-label="自定义训练学习率" value={config.learning_rate} onChange={event => change({ learning_rate: Number(event.target.value) })}>{[.0001, .0003, .001, .003, .01].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <label>验证集比例<select aria-label="自定义训练验证集比例" value={config.validation_ratio} onChange={event => change({ validation_ratio: Number(event.target.value) })}><option value={.1}>官方训练集的 10%</option><option value={.2}>官方训练集的 20%</option></select></label>
      <label>划分与初始化种子<input aria-label="自定义训练随机种子" type="number" min={0} max={9999} step={1} value={config.seed} onChange={event => { const value = Number(event.target.value); if (Number.isInteger(value) && value >= 0 && value <= 9999) change({ seed: value }); }} /></label>
    </div>
    <div className="ct-actions"><button className="primary-button" disabled={!analysis.valid || busy || anyActive || status.data?.dataset.status !== "ready" || !status.data?.runtime.available} onClick={() => void operate("start")}><Play size={15} />{busy ? "正在提交…" : "开始自定义网络训练"}</button>{status.data?.dataset.status !== "ready" && <button className="secondary-button" disabled={busy || status.data?.dataset.status === "preparing" || !status.data?.runtime.available} onClick={() => void operate("prepare")}><RefreshCw size={15} />{status.data?.dataset.status === "preparing" ? "正在准备数据…" : "准备 MNIST 数据"}</button>}<span>{status.data?.dataset.message ?? "正在连接训练服务…"}</span></div>
    {anyActive && <p className="ct-note">当前已有一个后台训练任务。任务完成或取消后，可以开始下一次实验。</p>}
    <p className="ct-notice" role="status">{notice}</p>
    <ErrorMessage error={error || status.error || records.error || status.data?.dataset.error || null} />

    <div className="ct-run-heading"><h3>我的自定义训练记录</h3><select aria-label="选择自定义训练记录" value={run?.id ?? ""} disabled={!customRuns.length && !freshRun} onChange={event => chooseRun(event.target.value)}>{!customRuns.length && !freshRun && <option value="">尚未训练自定义网络</option>}{freshRun && !customRuns.some(item => item.id === freshRun.id) && <option value={freshRun.id}>{freshRun.id.slice(0, 8)} · {statusLabels[freshRun.status]}</option>}{customRuns.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {statusLabels[item.status]} · {item.config.epochs} 轮 · {item.parameter_count.toLocaleString()} 参数</option>)}</select></div>
    {run ? <div className="ct-run-result">
      <div className="ct-snapshot"><div><strong>运行快照 {run.id.slice(0, 8)}</strong><span>{run.config.architecture?.length ?? 0} 层 · {run.parameter_count.toLocaleString()} 参数 · seed {run.config.seed} · {run.train_samples.toLocaleString()} 张训练图像</span></div><span className={`ct-status ${run.status}`}>{statusLabels[run.status]}</span></div>
      {changedSinceRun && <div className="ct-warning"><TriangleAlert size={16} />当前编辑结构与所选训练记录不同。下方曲线、权重和测试结果属于该记录保存的结构快照。</div>}
      <div className="ct-progress" role="progressbar" aria-label="自定义网络训练进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><span style={{ width: `${progress}%` }} /></div>
      <p className="ct-notice" role="status">{run.message}</p>
      <div className="ct-metrics"><div><span>完整训练轮次</span><strong>{run.history.length} / {run.config.epochs}</strong></div><div><span>最新验证准确率</span><strong>{latest ? `${(latest.validation_accuracy * 100).toFixed(2)}%` : "—"}</strong></div><div><span>验证最佳轮次</span><strong>{run.best_epoch === null ? "—" : `Epoch ${run.best_epoch}`}</strong></div><div><span>独立测试准确率</span><strong>{run.evaluation ? `${(run.evaluation.accuracy * 100).toFixed(2)}%` : "—"}</strong></div></div>
      <TrainingCurves history={run.history} />
      <div className="ct-actions">{activeStatuses.has(run.status) && <button className="secondary-button" disabled={busy || run.status === "cancelling"} onClick={() => void operate("cancel")}><Square size={13} />取消自定义训练</button>}{run.checkpoint_available && <a className="secondary-button" href={`/api/mnist/runs/${run.id}/checkpoint`} download><Download size={15} />下载最佳权重</a>}{run.evaluation && <><button className="secondary-button" onClick={() => openLesson("mnist-evaluation")}>查看混淆矩阵与错例<ArrowRight size={15} /></button><button className="primary-button" onClick={() => openLesson("handwriting")}>用这个模型识别手写数字<ArrowRight size={15} /></button></>}</div>
      <ErrorMessage error={run.error} />
      <div className="ct-comparison-heading"><div><h3>与基线模型对照</h3><p>只列出随机种子、验证比例和训练子集相同的已完成记录；轮数、学习率与批次大小仍需一起比较。</p></div><button className="text-button" onClick={openBaseline}>按相同配置训练基线<ArrowRight size={14} /></button></div>
      {comparison.length ? <div className="ct-table-scroll"><table className="ct-comparison"><thead><tr><th>模型 / 记录</th><th>训练轮数</th><th>学习率 / 批次</th><th>最佳验证损失</th><th>测试准确率</th></tr></thead><tbody>{[run, ...comparison].map(item => <tr key={item.id} className={item.id === run.id ? "is-current" : ""}><th>{modelLabel(item.config.model)}<small>{item.id.slice(0, 8)}</small></th><td>{item.config.epochs}</td><td>{item.config.learning_rate} / {item.config.batch_size}</td><td>{item.best_validation_loss === null ? "—" : format(item.best_validation_loss, 4)}</td><td>{item.evaluation ? `${(item.evaluation.accuracy * 100).toFixed(2)}%` : "—"}</td></tr>)}</tbody></table></div> : <p className="ct-note">尚无相同数据条件的 MLP / CNN 基线。到训练页使用 seed {run.config.seed}、验证比例 {run.config.validation_ratio * 100}%、{run.config.train_limit?.toLocaleString() ?? "全部"} 张训练样本完成一组基线。</p>}
      {excluded > 0 && <p className="ct-note">另有 {excluded} 份基线的数据划分或训练规模不同，未加入本次对照。</p>}
    </div> : <div className="ct-empty"><FlaskConical size={25} /><p>完成第一次真实训练后，这里会展示学习曲线、最佳权重和独立测试结果。</p></div>}
  </section>;
}
