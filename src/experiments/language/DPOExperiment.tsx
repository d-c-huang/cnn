import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Download, Pause, RefreshCw, Send } from "lucide-react";
import { ExperimentPanel, Slider } from "../../components/ui";
import { useLessonTask } from "../../lib/learningProgress";
import { sftApi } from "../../lib/sft";
import type { ChatMessage, SFTRun } from "../../lib/sft";
import { dpoApi, dpoDefaults, parsePreferences, preferenceQuestion, readDPODraft } from "../../lib/dpo";
import type { DPOComparison, DPODraft, DPOPreview, DPORun, DPOStatus } from "../../lib/dpo";
import { LossChart } from "./LossChart";
import "./language.css";
import "./lora.css";
import "./sft.css";

function download(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function storedId() { try { return localStorage.getItem("zhiwei-dpo-run-v1"); } catch { return null; } }
type Turn = { question: string; result: DPOComparison };
function DPOExperiment({ mode, onInteract }: { mode: "data" | "train" | "compare"; onInteract: () => void }) {
  const [draft, setDraft] = useState<DPODraft>(() => readDPODraft() ?? dpoDefaults);
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState<DPOStatus | null>(null);
  const [sources, setSources] = useState<SFTRun[]>([]);
  const [runs, setRuns] = useState<DPORun[]>([]);
  const [run, setRun] = useState<DPORun | null>(null);
  const [preview, setPreview] = useState<DPOPreview | null>(null);
  const [selected, setSelected] = useState(0);
  const [inspected, setInspected] = useState(false);
  const [pending, setPending] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("What is Zhiwei?");
  const [checkpoint, setCheckpoint] = useState<"latest" | "best">("latest");
  const [temperature, setTemperature] = useState(0);
  const [length, setLength] = useState(32);
  const [topK, setTopK] = useState(10);
  const [seed, setSeed] = useState(7);
  const active = useRef<AbortController | null>(null), alive = useRef(true), busy = useRef(false), stop = useRef(false);
  const parsed = useMemo(() => { try { return { records: parsePreferences(draft.dataset_text), error: "" }; } catch (e) { return { records: [], error: e instanceof Error ? e.message : "数据无效" }; } }, [draft.dataset_text]);
  const last = run?.history.at(-1);
  const source = sources.find(row => row.id === draft.source_sft_id);
  useLessonTask("dpo-data", mode === "data" && Boolean(preview?.valid && inspected));
  useLessonTask("dpo-train", mode === "train" && Boolean(run && run.step >= 2 && last!.train_loss < run.history[0].train_loss && run.frozen_verified && run.reference_verified));
  useLessonTask("dpo-compare", mode === "compare" && turns.some(turn => turn.result.step > 0 && Boolean(turn.result.after.reply.trim())));
  function selectRun(next: DPORun) {
    setRun(next); setTurns([]); setQuestion(preferenceQuestion(next.config.dataset[next.train_indices[0]]));
    setRuns(previous => [next, ...previous.filter(row => row.id !== next.id)].slice(0, 30));
    try { localStorage.setItem("zhiwei-dpo-run-v1", next.id); } catch { /* Server checkpoint is retained. */ }
  }
  async function action(work: (signal: AbortSignal) => Promise<void>) {
    if (busy.current) return;
    busy.current = true; setPending(true); setError(""); setNotice("");
    const controller = new AbortController(); active.current = controller;
    try { await work(controller.signal); }
    catch (e) { if (alive.current && !controller.signal.aborted) setError(`${e instanceof Error ? e.message : "计算失败"}。训练中断后请先刷新记录。`); }
    finally { if (active.current === controller) { active.current = null; busy.current = false; if (alive.current) { setPending(false); setTraining(false); } } }
  }
  useEffect(() => {
    alive.current = true;
    void action(async signal => {
      const [info, records, sfts] = await Promise.all([dpoApi<DPOStatus>("/status", undefined, signal), dpoApi<DPORun[]>("/runs", undefined, signal), sftApi<SFTRun[]>("/runs", undefined, signal)]);
      if (signal.aborted || !alive.current) return;
      setStatus(info); setRuns(records); setSources(sfts.filter(row => row.step > 0));
      setDraft(previous => previous.dataset_text ? previous : { ...previous, dataset_text: JSON.stringify(info.dataset, null, 2) });
      setLoaded(true); const next = records.find(row => row.id === storedId()) ?? records[0]; if (next) selectRun(next);
    });
    return () => { alive.current = false; stop.current = true; active.current?.abort(); busy.current = false; };
    // Restore each lesson from the local draft and persisted server snapshot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  useEffect(() => { if (loaded) { try { localStorage.setItem("zhiwei-dpo-draft-v1", JSON.stringify(draft)); } catch { /* Draft remains usable in memory. */ } } }, [draft, loaded]);
  function edit<K extends keyof DPODraft>(key: K, value: DPODraft[K]) { setDraft(previous => ({ ...previous, [key]: value })); setPreview(null); setInspected(false); if (key === "dataset_text") setSelected(0); }
  function config() { const { dataset_text, ...settings } = draft; return { ...settings, dataset: parsePreferences(dataset_text) }; }
  function inspect(index = selected) {
    onInteract(); void action(async signal => {
      const result = await dpoApi<DPOPreview>("/preview", { dataset: parsed.records, system_prompt: draft.system_prompt, max_length: draft.max_length, seed: draft.seed, selected_index: index }, signal);
      if (!signal.aborted && alive.current) { setPreview(result); setSelected(index); setInspected(false); setNotice(result.valid ? "偏好数据校验通过，点击候选回答 token 查看实际标签。" : "存在过长候选，请调整数据或最大长度。"); }
    });
  }
  function refresh() {
    void action(async signal => {
      const [records, sfts, info] = await Promise.all([dpoApi<DPORun[]>("/runs", undefined, signal), sftApi<SFTRun[]>("/runs", undefined, signal), dpoApi<DPOStatus>("/status", undefined, signal)]);
      if (signal.aborted || !alive.current) return;
      setRuns(records); setSources(sfts.filter(row => row.step > 0)); setStatus(info);
      const next = records.find(row => row.id === run?.id) ?? records[0]; if (next) selectRun(next); else { setRun(null); setTurns([]); }
      setNotice("已刷新本机检查点与 SFT 来源。");
    });
  }
  function train(steps: number) {
    if (!run) return; stop.current = false; setTraining(true); onInteract();
    void action(async signal => {
      let latest = run;
      for (let remaining = steps; remaining > 0 && !stop.current; remaining -= 5) {
        latest = await dpoApi<DPORun>(`/runs/${latest.id}/train`, { expected_step: latest.step, steps: Math.min(5, remaining) }, signal);
        if (signal.aborted || !alive.current) return; selectRun(latest);
      }
      setNotice(`${stop.current ? "已暂停" : "DPO 已保存"}：当前 ${latest.step} 步，最佳验证权重为第 ${latest.best_step} 步。`);
    });
  }
  function compare() {
    if (!run || !question.trim()) return;
    const asked = question.trim();
    const messages: ChatMessage[] = turns.slice(-4).flatMap(turn => [{ role: "user" as const, content: turn.question }, { role: "assistant" as const, content: turn.result.after.reply }]);
    messages.push({ role: "user", content: asked }); onInteract();
    void action(async signal => {
      const result = await dpoApi<DPOComparison>(`/runs/${run.id}/compare`, { messages, checkpoint, temperature, length, top_k: topK, seed }, signal);
      if (signal.aborted || !alive.current) return;
      // A blank after-answer cannot be a valid assistant history entry.
      setTurns(previous => [...previous, { question: asked, result }]); setQuestion("");
      setNotice(`已对比固定参考与第 ${result.step} 步权重；相同上下文 ${result.after.context_tokens} tokens，服务裁剪 ${result.after.dropped_turns} 轮。`);
    });
  }
  async function importFile(file: File) {
    try { if (file.size > 200000) throw new Error("文件不能超过 200 KB"); const rows = parsePreferences(await file.text()); edit("dataset_text", JSON.stringify(rows, null, 2)); setNotice(`已导入 ${rows.length} 条偏好记录。`); }
    catch (e) { setError(e instanceof Error ? e.message : "导入失败"); }
  }
  const canCompare = turns.every(turn => turn.result.after.reply.trim());
  return <div className="language-lab">
    <div className="language-intro"><span className="small-label">DIRECT PREFERENCE OPTIMIZATION · FIXED REFERENCE · REAL LORA</span><p>同一个 prompt 配 chosen / rejected 两个回答。DPO 相对于固定参考模型提高偏好回答的相对概率，只更新策略 LoRA，不另外训练奖励模型。</p></div>
    {error && <div className="connection-notice" role="alert">{error}</div>}{notice && <p className="language-status" role="status">{notice}</p>}
    <ExperimentPanel title="预训练基座与固定参考"><button className="secondary-button" disabled={pending} onClick={refresh}><RefreshCw size={14} />刷新记录</button><p>EleutherAI/pythia-14m · 固定版本 cf967c0a9a04 · 对话模板 zhiwei-sft-v1</p><p role="status">{status?.state === "ready" ? "固定版本基座已缓存，可离线训练。" : status?.state === "downloading" ? "基座下载中，请稍后刷新记录。" : "请先准备第七阶段使用的公开预训练模型。"}</p>{status && status.state !== "ready" && <button className="primary-button" disabled={pending || status.state === "downloading"} onClick={() => void action(async signal => { await dpoApi("/prepare", {}, signal); const info = await dpoApi<DPOStatus>("/status", undefined, signal); if (!signal.aborted) setStatus(info); })}>准备固定版本预训练模型</button>}<p>复用现有缓存。推荐从已训练 SFT 适配器开始；小型英文基座和少量偏好样本不保证通用助手能力或回答质量提升。</p></ExperimentPanel>
    {mode === "data" ? <>
      <ExperimentPanel title="自定义偏好数据集"><fieldset className="language-generation-controls" disabled={pending}><legend>JSON 数组 / JSONL 偏好对</legend><p>5–64 条记录，至少 5 个不同首问，总文本 ≤20,000 字符。每条包含 prompt、chosen、rejected；prompt 可为字符串或以 user 结束的交替消息列表。两个回答须不同。</p><label className="language-field">偏好数据 JSON / JSONL<textarea rows={14} maxLength={200000} value={draft.dataset_text} onChange={e => edit("dataset_text", e.target.value)} /></label><div className="language-actions"><label className="secondary-button sft-upload">导入偏好 JSON / JSONL<input type="file" accept=".json,.jsonl,.ndjson,application/json,text/plain" onChange={e => { const file = e.target.files?.[0]; if (file) void importFile(file); e.currentTarget.value = ""; }} /></label><button className="secondary-button" onClick={() => status && edit("dataset_text", JSON.stringify(status.dataset, null, 2))}>恢复示例偏好</button><button className="secondary-button" disabled={Boolean(parsed.error)} onClick={() => download(parsed.records, "zhiwei-dpo-preferences.json")}>下载偏好数据集</button></div><p>{parsed.error || `当前 ${parsed.records.length} 条偏好记录。`}</p><label className="language-field">共享系统提示<textarea rows={2} maxLength={256} value={draft.system_prompt} onChange={e => edit("system_prompt", e.target.value)} /></label><div className="transformer-config-grid"><label className="language-field">样本最大长度<select value={draft.max_length} onChange={e => edit("max_length", Number(e.target.value))}>{[64,128,256].map(v => <option key={v} value={v}>{v} tokens</option>)}</select></label><label className="language-field">划分种子<input type="number" min={0} max={4294967295} value={draft.seed} onChange={e => edit("seed", Number(e.target.value))} /></label></div><button className="primary-button" disabled={Boolean(parsed.error) || status?.state !== "ready"} onClick={() => inspect()}>校验偏好数据与掩码</button></fieldset><p>按首问分组划分 80% 训练 / 20% 验证，整对保持同组；重复 prompt 拒绝，超长候选明确报错而不截断。草稿跨课程、刷新保留。</p></ExperimentPanel>
      {preview && <ExperimentPanel title="两份候选的真实 token 标签"><p>训练 / 验证：{preview.train_indices.length} / {preview.validation_indices.length} 对。只给最终候选回答与 EOS 计分，历史助手消息仅作上下文。</p>{preview.errors.map(e => <p role="alert" key={e}>{e}</p>)}<label className="language-field">预览偏好对<select value={selected} disabled={pending} onChange={e => inspect(Number(e.target.value))}>{parsed.records.map((row, i) => <option key={i} value={i}>#{i + 1} · {preview.train_indices.includes(i) ? "训练" : "验证"} · {preferenceQuestion(row)}</option>)}</select></label><div className="lora-comparison">{(["chosen", "rejected"] as const).map(variant => <section key={variant}><h3>{variant === "chosen" ? "Chosen · 偏好回答" : "Rejected · 非偏好回答"}</h3><p>{preview[variant].total_tokens} 个输入 token；{preview[variant].supervised_tokens} 个计分目标。</p><div className="sft-token-mask language-pairs language-actions">{preview[variant].tokens.map((token, i) => <button key={i} className={`secondary-button ${token.supervised ? "is-supervised" : "is-ignored"}`} title={`ID ${token.id} · label ${token.label}`} onClick={() => { setInspected(token.supervised); setNotice(`已检查 ${variant} token ${token.id}：label=${token.label}${token.supervised ? "，计入该回答的 log 概率之和。" : "，只作上下文。"}`); onInteract(); }}>{token.text.replace(/\n/g, "↵") || `ID ${token.id}`}</button>)}</div></section>)}</div><a className="text-button" href="#dpo-train">用此草稿创建 DPO <ArrowRight size={14} /></a></ExperimentPanel>}
    </> : <>
      <ExperimentPanel title="本机 DPO 实验"><label className="language-field">选择 DPO 实验<select value={run?.id ?? ""} disabled={pending || !runs.length} onChange={e => { const next = runs.find(row => row.id === e.target.value); if (next) selectRun(next); }}>{!runs.length && <option value="">尚无实验，请先创建 DPO</option>}{runs.map(row => <option key={row.id} value={row.id}>{row.id.slice(0,8)} · {row.step} 步 · {row.config.dataset.length} 对</option>)}</select></label><p>数据、初始参考适配器、参考 log 概率、策略适配器和优化器保存在 runs/dpo。SFT 来源后续训练不会改变这个参考快照。</p></ExperimentPanel>
      <ExperimentPanel title="创建偏好优化实验"><fieldset className="language-generation-controls" disabled={pending}><legend>DPO + LoRA 配置</legend><p>当前草稿 {parsed.records.length} 对，最大长度 {draft.max_length}。<a href="#dpo-data">编辑偏好数据和系统提示</a></p><label className="language-field">初始策略与固定参考<select value={draft.source_sft_id ?? ""} onChange={e => edit("source_sft_id", e.target.value || null)}><option value="">固定预训练基座（未做 SFT）</option>{sources.map(row => <option key={row.id} value={row.id}>SFT {row.id.slice(0,8)} · 当前 {row.step} / 最佳 {row.best_step} 步</option>)}</select></label>{draft.source_sft_id && <label className="language-field">SFT 来源权重<select value={draft.source_checkpoint} onChange={e => edit("source_checkpoint", e.target.value as "latest" | "best")}><option value="latest">当前 SFT 适配器</option><option value="best">最佳验证 SFT 适配器</option></select></label>}<div className="transformer-config-grid">{(["rank", "alpha", "batch_size"] as const).map(key => <label className="language-field" key={key}>{({rank:"LoRA rank",alpha:"LoRA alpha",batch_size:"偏好对批大小"})[key]}<select disabled={Boolean(source && key !== "batch_size")} value={source && key !== "batch_size" ? source.config[key] : draft[key]} onChange={e => edit(key, Number(e.target.value))}>{(key === "rank" ? [2,4,8,16] : key === "alpha" ? [4,8,16,32] : [1,2,4]).map(v => <option key={v}>{v}</option>)}</select></label>)}<label className="language-field">学习率<input type="number" min={0.00001} max={0.01} step={0.00001} value={draft.learning_rate} onChange={e => edit("learning_rate", Number(e.target.value))} /></label><label className="language-field">DPO beta<input type="number" min={0.01} max={1} step={0.01} value={draft.beta} onChange={e => edit("beta", Number(e.target.value))} /></label></div><p>从 SFT 继承其 rank / alpha；系统提示和长度使用当前偏好草稿。参考固定为所选来源，策略从同一权重开始；beta 是参考相对 log 概率差的缩放，不是学习率。</p><button className="primary-button" disabled={Boolean(parsed.error) || status?.state !== "ready"} onClick={() => void action(async signal => { const next = await dpoApi<DPORun>("/runs", config(), signal); if (!signal.aborted && alive.current) { selectRun(next); setNotice("DPO 已创建：初始策略等于固定参考，损失约为 ln 2。"); } })}>创建 DPO 实验</button></fieldset></ExperimentPanel>
      {run && <>
        <div className="language-metrics"><div><span>可训练 / 冻结参数</span><strong>{run.trainable_parameters.toLocaleString()} / {run.base_parameters.toLocaleString()}</strong></div><div><span>当前 / 最佳步数</span><strong>{run.step} / {run.best_step}</strong></div><div><span>训练 / 验证偏好对</span><strong>{run.train_indices.length} / {run.validation_indices.length}</strong></div></div><p className="lora-frozen-status">基座冻结与固定参考校验{run.frozen_verified && run.reference_verified ? "通过" : "未通过"}。来源：{run.source.kind === "sft" ? `SFT ${run.source.run_id?.slice(0,8)} 第 ${run.source.step} 步` : "预训练基座"}。</p>
        {mode === "train" && <><ExperimentPanel title="真实 LoRA DPO 训练"><p>beta={run.config.beta}，学习率 {run.config.learning_rate}，每批 {run.config.batch_size} 对。chosen / rejected 分别求最终回答与 EOS 的序列 log 概率之和；固定参考分数已预计算，只抽取训练偏好对更新策略。</p><div className="language-actions">{[1,5,20,100].map(steps => <button key={steps} className="primary-button" disabled={pending || run.step + steps > 500} onClick={() => train(steps)}>DPO {steps} 步</button>)}{training && <button className="secondary-button" onClick={() => { stop.current = true; }}><Pause size={14} />暂停后续 DPO</button>}</div><p>每请求最多 5 步，最多 500 步；暂停或离开阻止后续提交，当前批仍可能保存。中断后先刷新，再续训。</p></ExperimentPanel><ExperimentPanel title="偏好损失与参考相对间隔"><div className="language-metrics"><div><span>训练 DPO 损失</span><strong>{last!.train_loss.toFixed(4)}</strong></div><div><span>验证 DPO 损失</span><strong>{last!.validation_loss.toFixed(4)}</strong></div><div><span>训练 / 验证偏好准确率</span><strong>{(last!.train_accuracy*100).toFixed(0)}% / {(last!.validation_accuracy*100).toFixed(0)}%</strong></div><div><span>训练 / 验证平均间隔</span><strong>{last!.train_margin.toFixed(3)} / {last!.validation_margin.toFixed(3)}</strong></div></div><LossChart history={run.history} label="DPO 损失" /><p>每个偏好对权重相同，不按 token 数平均；间隔为 beta×[(策略 chosen−参考 chosen)−(策略 rejected−参考 rejected)]。准确率统计间隔严格大于 0 的偏好对，初始全平局记为 0%。它不是人类胜率或对话正确率。</p><details><summary>查看真实逐批指标与候选分数</summary><div className="language-table"><table><thead><tr><th>步数</th><th>训练损失</th><th>验证损失</th><th>训练间隔</th><th>验证间隔</th></tr></thead><tbody>{run.history.map(row => <tr key={row.step}><td>{row.step}</td><td>{row.train_loss.toFixed(5)}</td><td>{row.validation_loss.toFixed(5)}</td><td>{row.train_margin.toFixed(4)}</td><td>{row.validation_margin.toFixed(4)}</td></tr>)}</tbody></table></div><pre>{JSON.stringify(run.pair_scores, null, 2)}</pre></details><a className="text-button" href="#dpo-compare">对照优化前后生成 <ArrowRight size={14} /></a></ExperimentPanel></>}
        {mode === "compare" && <ExperimentPanel title="同一上下文，优化前后对照"><fieldset className="language-generation-controls" disabled={pending}><legend>两侧共享的生成设置</legend><label className="language-field">优化后权重<select value={checkpoint} onChange={e => { setCheckpoint(e.target.value as "latest" | "best"); setTurns([]); }}><option value="latest">当前 DPO · 第 {run.step} 步</option><option value="best">最佳验证 DPO · 第 {run.best_step} 步</option></select></label><details><summary>调整共享采样参数</summary><Slider label="温度（0 为贪心）" min={0} max={2} step={0.1} value={temperature} onChange={setTemperature} /><Slider label="最多回答 token 数" min={8} max={64} step={8} value={length} onChange={setLength} /><label className="language-field">Top-k<select value={topK} onChange={e => setTopK(Number(e.target.value))}>{[0,10,50].map(v => <option key={v} value={v}>{v === 0 ? "全部候选" : `前 ${v} 个`}</option>)}</select></label><label className="language-field">共享采样种子<input type="number" min={0} max={4294967295} value={seed} onChange={e => setSeed(Number(e.target.value))} /></label></details></fieldset><p>左侧始终为创建时固定参考；右侧为所选 DPO 权重。每次两侧使用完全相同的系统、历史、问题、种子和采样参数。追问共用优化后回答作为历史（最多最近四轮），以隔离权重差异；不是两条各自分叉的对话。超长时按整轮裁剪。</p><div role="log" aria-label="优化前后对话记录">{turns.map((turn,i) => <section key={i}><h3>你：{turn.question}</h3><div className="lora-comparison">{(["before","after"] as const).map(side => <section key={side}><h4>{side === "before" ? "优化前 · 固定参考" : `优化后 · 第 ${turn.result.step} 步`}</h4><div className="language-output"><pre>{turn.result[side].reply || "（模型未生成可见回答）"}</pre></div><p>{turn.result[side].context_tokens} 上下文 tokens · 生成 {turn.result[side].token_ids.length} tokens · {turn.result[side].finish_reason}</p></section>)}</div></section>)}</div>{!canCompare && <p>优化后模型没有可见回答，请开始新对话再尝试。</p>}<div className="language-actions"><button className="secondary-button" disabled={pending || !turns.length} onClick={() => setTurns([])}>开始新对比</button><button className="secondary-button" disabled={!turns.length} onClick={() => download({run_id:run.id,turns},"zhiwei-dpo-comparison.json")}>导出对比记录</button>{run.train_indices.slice(0,3).map(index => <button key={index} className="text-button" disabled={pending} onClick={() => setQuestion(preferenceQuestion(run.config.dataset[index]))}>{preferenceQuestion(run.config.dataset[index])}</button>)}</div><form onSubmit={e => { e.preventDefault(); compare(); }}><label className="language-field">发送对比问题<textarea rows={3} maxLength={1000} disabled={pending} value={question} onChange={e => setQuestion(e.target.value)} /></label><button className="primary-button" disabled={pending || !canCompare || !question.trim() || !Number.isInteger(seed) || seed < 0 || seed > 4294967295}><Send size={14} />生成优化前后回答</button></form><p>偏好间隔上升不保证自由生成更好；比较训练问题和未见问题，留意重复、回避与提前结束。这里不自动把某一侧判为赢家。</p></ExperimentPanel>}
        <details className="language-reading"><summary>DPO 数据、参考与配置快照</summary><p>模板 {run.template_version}；参考指纹 {run.reference_digest}；数据指纹 {run.dataset_fingerprint}</p><p>训练索引 {run.train_indices.join(", ")}；验证索引 {run.validation_indices.join(", ")}</p><pre>{JSON.stringify(run.config,null,2)}</pre></details><div className="language-actions"><a className="secondary-button" download href={`/api/dpo/runs/${run.id}/adapter?checkpoint=${checkpoint}`}><Download size={14} />下载 DPO 适配器与模板</a><a className="secondary-button" download href={`/api/dpo/runs/${run.id}/checkpoint`}>下载 DPO 续训检查点</a></div>
      </>}
    </>}
    <details className="language-reading"><summary>DPO 原理与官方资料</summary><p>采用标准 sigmoid DPO：−log σ(beta×[(log π chosen−log π rejected)−(log ref chosen−log ref rejected)])。只对最终候选回答与 EOS 求和，prompt 和填充不计分，不附加 SFT 损失。无需单独奖励模型或训练时在线生成。</p><p>参考 <a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noreferrer">DPO 论文</a>与 <a href="https://huggingface.co/docs/trl/dpo_trainer" target="_blank" rel="noreferrer">Hugging Face DPO 文档</a>。本课程用 PyTorch 显式实现，不依赖 TRL Trainer。</p></details>
  </div>;
}
export function DPOData(props: {onInteract:()=>void}) { return <DPOExperiment mode="data" {...props} />; }
export function DPOTraining(props: {onInteract:()=>void}) { return <DPOExperiment mode="train" {...props} />; }
export function DPOCompare(props: {onInteract:()=>void}) { return <DPOExperiment mode="compare" {...props} />; }
