import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Download, Pause, Play, RefreshCw, Send, Upload } from "lucide-react";
import { ExperimentPanel, Slider } from "../../components/ui";
import { useLessonTask } from "../../lib/learningProgress";
import { parseSFTDataset, readSFTDraft, recordQuestion, sftApi, sftDefaults } from "../../lib/sft";
import type { ChatMessage, SFTConfig, SFTDraft, SFTPreview, SFTRecord, SFTReply, SFTRun, SFTStatus } from "../../lib/sft";
import { LossChart } from "./LossChart";
import "./language.css";
import "./sft.css";

const runKey = "zhiwei-sft-run-v1";
const tokenLabel = (text: string) => text.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/^ +$/, "␣");
function storedId() { try { return localStorage.getItem(runKey); } catch { return null; } }
function downloadJson(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function SFTExperiment({ mode, onInteract }: { mode: "data" | "train" | "chat"; onInteract: () => void }) {
  const [draft, setDraft] = useState<SFTDraft>(() => readSFTDraft() ?? sftDefaults);
  const [loaded, setLoaded] = useState(Boolean(draft.dataset_text));
  const [status, setStatus] = useState<SFTStatus | null>(null);
  const [runs, setRuns] = useState<SFTRun[]>([]);
  const [run, setRun] = useState<SFTRun | null>(null);
  const [preview, setPreview] = useState<SFTPreview | null>(null);
  const [selectedSample, setSelectedSample] = useState(0);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [newQuestion, setNewQuestion] = useState("");
  const [newAnswer, setNewAnswer] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("What is Zhiwei?");
  const [reply, setReply] = useState<SFTReply | null>(null);
  const [checkpoint, setCheckpoint] = useState<"latest" | "best">("latest");
  const [temperature, setTemperature] = useState(0);
  const [length, setLength] = useState(32);
  const [topK, setTopK] = useState(10);
  const [seed, setSeed] = useState(7);
  const active = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const busy = useRef(false);
  const stop = useRef(false);
  const parsed = useMemo(() => { try { return { records: parseSFTDataset(draft.dataset_text), error: "" }; } catch (reason) { return { records: [] as SFTRecord[], error: reason instanceof Error ? reason.message : "数据格式无效" }; } }, [draft.dataset_text]);
  const last = run?.history.at(-1);
  useLessonTask("sft-data", mode === "data" && Boolean(preview?.valid && selectedToken !== null && preview.tokens[selectedToken]?.supervised));
  useLessonTask("sft-train", mode === "train" && Boolean(run && run.step >= 2 && run.frozen_verified && last!.train_loss < run.history[0].train_loss));
  useLessonTask("sft-chat", mode === "chat" && Boolean(reply && reply.step > 0 && reply.reply.trim()));

  function selectRun(next: SFTRun) {
    setRun(next); setMessages([]); setReply(null);
    setQuestion(recordQuestion(next.config.dataset[next.train_indices[0]]));
    setRuns(previous => [next, ...previous.filter(item => item.id !== next.id)].slice(0, 30));
    try { localStorage.setItem(runKey, next.id); } catch { /* Training state stays on the server. */ }
  }
  async function action(work: (signal: AbortSignal) => Promise<void>) {
    if (busy.current) return;
    busy.current = true; setPending(true); setError(""); setNotice("");
    const controller = new AbortController(); active.current = controller;
    try { await work(controller.signal); }
    catch (reason) { if (alive.current && !controller.signal.aborted) setError(`${reason instanceof Error ? reason.message : "计算失败"}。训练请求中断时，请先刷新记录。`); }
    finally { if (active.current === controller) { active.current = null; busy.current = false; if (alive.current) { setPending(false); setTraining(false); } } }
  }
  useEffect(() => {
    alive.current = true;
    void action(async signal => {
      const [info, records] = await Promise.all([sftApi<SFTStatus>("/status", undefined, signal), sftApi<SFTRun[]>("/runs", undefined, signal)]);
      if (signal.aborted || !alive.current) return;
      setStatus(info); setRuns(records);
      setDraft(previous => previous.dataset_text ? previous : { ...previous, dataset_text: JSON.stringify(info.dataset, null, 2), system_prompt: info.system_prompt });
      setLoaded(true);
      const selected = records.find(item => item.id === storedId()) ?? records[0]; if (selected) selectRun(selected);
    });
    return () => { alive.current = false; stop.current = true; active.current?.abort(); busy.current = false; };
    // Restore the local dataset draft and the selected server experiment once per lesson.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  useEffect(() => { if (loaded) { try { localStorage.setItem("zhiwei-sft-draft-v1", JSON.stringify(draft)); } catch { /* The draft is still usable in memory. */ } } }, [draft, loaded]);
  useEffect(() => {
    if (status?.state !== "downloading") return;
    const controller = new AbortController(); let timer: number;
    async function poll() {
      try { const info = await sftApi<SFTStatus>("/status", undefined, controller.signal); if (!controller.signal.aborted) setStatus(info); } catch { /* A later refresh may recover. */ }
      if (!controller.signal.aborted) timer = window.setTimeout(() => void poll(), 2000);
    }
    timer = window.setTimeout(() => void poll(), 1000);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [status?.state]);

  function edit(key: keyof SFTDraft, value: string | number) {
    setDraft(previous => ({ ...previous, [key]: value })); setPreview(null); setSelectedToken(null);
    if (key === "dataset_text") setSelectedSample(0);
  }
  function config(): SFTConfig { const { dataset_text, ...settings } = draft; return { ...settings, dataset: parseSFTDataset(dataset_text) }; }
  function inspect(index = selectedSample) {
    onInteract(); void action(async signal => {
      const payload = config();
      const result = await sftApi<SFTPreview>("/preview", { dataset: payload.dataset, system_prompt: payload.system_prompt, max_length: payload.max_length, seed: payload.seed, selected_index: index }, signal);
      if (!signal.aborted && alive.current) { setPreview(result); setSelectedSample(index); setSelectedToken(null); setNotice(result.valid ? "校验通过，点击一个回答 token 查看监督标签。" : "样本过长，请调整数据或最大长度后重新校验。"); }
    });
  }
  function refresh() {
    void action(async signal => {
      const [info, records] = await Promise.all([sftApi<SFTStatus>("/status", undefined, signal), sftApi<SFTRun[]>("/runs", undefined, signal)]);
      if (signal.aborted || !alive.current) return;
      setStatus(info); setRuns(records); const selected = records.find(item => item.id === run?.id) ?? records[0];
      if (selected) selectRun(selected); else { setRun(null); setMessages([]); setReply(null); }
      setNotice("已从本机检查点刷新。");
    });
  }
  function train(steps: number) {
    if (!run) return;
    stop.current = false; setTraining(true); onInteract();
    void action(async signal => {
      let latest = run;
      for (let remaining = steps; remaining > 0 && !stop.current; remaining -= 5) {
        latest = await sftApi<SFTRun>(`/runs/${latest.id}/train`, { expected_step: latest.step, steps: Math.min(5, remaining) }, signal);
        if (signal.aborted || !alive.current) return;
        selectRun(latest);
      }
      setNotice(`${stop.current ? "已暂停" : "SFT 已保存"}：当前 ${latest.step} 步，最佳验证适配器为第 ${latest.best_step} 步。`);
    });
  }
  function send() {
    if (!run || !question.trim()) return;
    onInteract();
    const userMessage: ChatMessage = { role: "user", content: question.trim() };
    const context = [...messages.slice(-8), userMessage];
    const omitted = Math.max(0, (messages.length - 8) / 2);
    void action(async signal => {
      const result = await sftApi<SFTReply>(`/runs/${run.id}/chat`, { messages: context, checkpoint, temperature, length, top_k: topK, seed }, signal);
      if (signal.aborted || !alive.current) return;
      setReply(result);
      if (!result.reply.trim()) { setNotice("模型立即结束或未生成可见回答；可以调整权重或生成设置后重试。"); return; }
      setMessages(previous => [...previous, userMessage, { role: "assistant", content: result.reply }]); setQuestion("");
      setNotice(`已生成 ${result.token_ids.length} 个 token，使用 ${result.context_tokens} 个上下文 token；本次忽略 ${omitted + result.dropped_turns} 轮较早历史。`);
    });
  }
  async function importFile(file: File) {
    try {
      if (file.size > 200000) throw new Error("导入文件不能超过 200 KB。");
      const text = await file.text(); const records = parseSFTDataset(text); edit("dataset_text", JSON.stringify(records, null, 2)); setNotice(`已导入 ${records.length} 条记录，请预览与校验。`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "文件读取失败"); }
  }

  return <div className="language-lab">
    <div className="language-intro"><span className="small-label">SUPERVISED FINETUNING · ASSISTANT-ONLY LOSS · REAL LORA</span><p>把问答组织成固定对话模板，只监督助手回答与 EOS。系统提示、用户问题、角色前缀和填充位置都设为 −100；基座冻结，只更新 LoRA。</p></div>
    {error && <div className="connection-notice" role="alert">{error}</div>}
    <p className="language-status" role="status">{pending ? training ? "正在真实 SFT，完成一批后评估并保存…" : "正在读取或计算…" : notice}</p>
    <ExperimentPanel title="预训练基座与 SFT 模板" action={<button className="secondary-button" disabled={pending} onClick={refresh}><RefreshCw size={14} />刷新记录</button>}>
      <p>{status?.model_id ?? "EleutherAI/pythia-14m"} · 固定版本 {status?.revision.slice(0, 12) ?? "…"} · 模板 {status?.template_version ?? "zhiwei-sft-v1"}</p><p role="status">{status?.message ?? "正在检查本机缓存…"}</p>
      {status && !status.available && <p role="alert">请先安装 requirements.txt 中的 Transformers、PEFT 与 Safetensors 依赖。</p>}
      {status?.state !== "ready" && <button className="primary-button" disabled={pending || !status?.available || status.state === "downloading"} onClick={() => void action(async signal => { const result = await sftApi<SFTStatus>("/prepare", {}, signal); if (!signal.aborted && alive.current) setStatus(previous => ({ ...previous!, ...result })); })}>{status?.state === "downloading" ? "正在准备预训练模型…" : "下载并准备预训练模型"}</button>}
      <p>复用第七阶段缓存，无需重复下载。该基座主要面向英文，默认问答使用英文；小模型学习少量示例不等同于通用聊天能力。</p>
    </ExperimentPanel>
    {mode === "data" && <>
      <ExperimentPanel title="编辑或导入自定义问答集">
        <fieldset className="language-generation-controls" disabled={pending}><legend>JSON 数组 / JSONL</legend><p>5–64 条记录，至少 5 个不同首问，总消息文本不超过 20,000 字符。支持 question/answer 对象，也支持 user、assistant 交替且以回答结束的 messages（每条最多 4 轮）。</p>
          <label className="language-field">问答数据 JSON / JSONL<textarea rows={12} value={draft.dataset_text} maxLength={200000} onChange={event => edit("dataset_text", event.target.value)} spellCheck={false} /></label>
          <div className="language-actions"><label className="secondary-button sft-upload"><Upload size={14} />导入 JSON / JSONL<input type="file" accept=".json,.jsonl,.ndjson,application/json,text/plain" onChange={event => { const file = event.target.files?.[0]; if (file) void importFile(file); event.currentTarget.value = ""; }} /></label><button className="secondary-button" onClick={() => { if (status) edit("dataset_text", JSON.stringify(status.dataset, null, 2)); }}>恢复示例问答</button><button className="secondary-button" disabled={Boolean(parsed.error)} onClick={() => downloadJson(parsed.records, "zhiwei-sft-dataset.json")}><Download size={14} />下载数据集</button></div>
          {parsed.error ? <p role="alert">{parsed.error}</p> : <p>当前 {parsed.records.length} 条记录。校验会拒绝空白回答、重复用户消息序列和过长样本。</p>}
          <details><summary>快速添加一条问答</summary><label className="language-field">新问题<input maxLength={1000} value={newQuestion} onChange={event => setNewQuestion(event.target.value)} /></label><label className="language-field">新回答<textarea rows={2} maxLength={1000} value={newAnswer} onChange={event => setNewAnswer(event.target.value)} /></label><button className="secondary-button" disabled={!newQuestion.trim() || !newAnswer.trim() || Boolean(parsed.error) || parsed.records.length >= 64} onClick={() => { edit("dataset_text", JSON.stringify([...parsed.records, { question: newQuestion.trim(), answer: newAnswer.trim() }], null, 2)); setNewQuestion(""); setNewAnswer(""); }}>添加到数据集</button></details>
          <label className="language-field">共享系统提示<input value={draft.system_prompt} maxLength={256} onChange={event => edit("system_prompt", event.target.value)} /></label>
          <div className="transformer-config-grid"><label className="language-field">样本最大长度<select value={draft.max_length} onChange={event => edit("max_length", Number(event.target.value))}>{[64, 128, 256].map(value => <option value={value} key={value}>{value} tokens</option>)}</select></label><label className="language-field">数据划分种子<input type="number" min={0} max={4294967295} value={draft.seed} onChange={event => edit("seed", Number(event.target.value))} /></label></div>
          <button className="primary-button" disabled={Boolean(parsed.error) || status?.state !== "ready" || parsed.records.length < 5} onClick={() => inspect(0)}>预览与校验数据集</button>
        </fieldset><p>按首问分组，随机抽取 80% 组作训练、20% 组作验证；同一首问的不同对话不会跨集合。草稿保存在当前浏览器，切换课程或刷新后保留。</p>
      </ExperimentPanel>
      {preview && <ExperimentPanel title="真实分词、标签与回答掩码">
        <div className="language-metrics"><div><span>训练 / 验证样本</span><strong>{preview.train_indices.length} / {preview.validation_indices.length}</strong></div><div><span>当前样本 tokens</span><strong>{preview.total_tokens}</strong></div><div><span>参与损失的回答 tokens</span><strong>{preview.records[selectedSample].supervised_tokens}</strong></div></div>
        {preview.errors.map(message => <p role="alert" key={message}>{message}</p>)}
        <label className="language-field">预览样本<select disabled={pending} value={selectedSample} onChange={event => inspect(Number(event.target.value))}>{preview.records.map(record => <option key={record.index} value={record.index}>#{record.index + 1} · {preview.train_indices.includes(record.index) ? "训练" : "验证"} · {record.question.slice(0, 55)}</option>)}</select></label>
        <p>点击 token 查看标签。亮色参与回答损失，灰色为 −100（忽略）。目标标签保存在 token 自身位置，损失计算时只右移一次，由前一个位置的 logits 预测它。</p>
        <div className="sft-token-mask language-pairs language-actions">{preview.tokens.map((token, index) => <button key={index} className={`secondary-button ${token.supervised ? "is-supervised" : "is-ignored"}`} aria-pressed={selectedToken === index} title={`位置 ${index}，token ID ${token.id}，label ${token.label}`} onClick={() => { setSelectedToken(index); onInteract(); }}>{tokenLabel(token.text) || `ID ${token.id}`}</button>)}</div>
        {selectedToken !== null && <p role="status">位置 {selectedToken}：ID {preview.tokens[selectedToken].id}，label={preview.tokens[selectedToken].label}，{preview.tokens[selectedToken].supervised ? "参与回答损失" : "忽略此目标，不计算损失"}。</p>}
        {preview.total_tokens > preview.tokens.length && <p>仅展示前 {preview.tokens.length} 个 token；该过长样本不会进入训练。</p>}
        <details><summary>查看实际模板文本与划分</summary><pre>{preview.rendered_text}</pre><p>训练索引（从 0 开始）：{preview.train_indices.join(", ")}</p><p>验证索引：{preview.validation_indices.join(", ")}</p><p>数据指纹：{preview.dataset_fingerprint}</p></details><a className="text-button" href="#sft-train">用此草稿创建 SFT 实验 <ArrowRight size={14} /></a>
      </ExperimentPanel>}
    </>}
    {mode !== "data" && <>
      <ExperimentPanel title="本机 SFT 实验"><label className="language-field">选择 SFT 实验<select disabled={pending || !runs.length} value={run?.id ?? ""} onChange={event => { const item = runs.find(row => row.id === event.target.value); if (item) selectRun(item); }}>
        {!runs.length && <option value="">尚无实验，请先创建 SFT</option>}{runs.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · r={item.config.rank} · {item.step} 步</option>)}
      </select></label><p>数据快照、模板、当前/最佳适配器、优化器和采样状态保存在 runs/sft。与第七阶段的领域文本 LoRA 实验独立。</p>{!run && mode === "chat" && <a className="text-button" href="#sft-train">前往真实 SFT <ArrowRight size={14} /></a>}</ExperimentPanel>
      {mode === "train" && <ExperimentPanel title="从自定义问答草稿创建实验"><fieldset className="language-generation-controls" disabled={pending}><legend>SFT + LoRA 配置</legend>
        <p>当前草稿 {parsed.records.length} 条样本，最大长度 {draft.max_length}，划分 seed {draft.seed}。</p>{parsed.error && <p role="alert">{parsed.error}</p>}
        <a className="text-button" href="#sft-data">编辑问答数据、系统提示与回答掩码 <ArrowRight size={14} /></a>
        <div className="transformer-config-grid">{[["rank", "LoRA rank", [2, 4, 8, 16]], ["alpha", "LoRA alpha", [4, 8, 16, 32]], ["batch_size", "批大小", [1, 2, 4]]].map(([key, label, values]) => <label className="language-field" key={key as string}>{label as string}<select value={draft[key as keyof SFTDraft]} onChange={event => edit(key as keyof SFTDraft, Number(event.target.value))}>{(values as number[]).map(value => <option key={value} value={value}>{value}</option>)}</select></label>)}<label className="language-field">学习率<input type="number" min={0.0001} max={0.01} step={0.0001} value={draft.learning_rate} onChange={event => edit("learning_rate", Number(event.target.value))} /></label></div>
        <button className="primary-button" disabled={Boolean(parsed.error) || parsed.records.length < 5 || status?.state !== "ready" || draft.learning_rate < 0.0001 || draft.learning_rate > 0.01} onClick={() => { onInteract(); void action(async signal => { const next = await sftApi<SFTRun>("/runs", config(), signal); if (!signal.aborted && alive.current) { selectRun(next); setNotice("SFT 适配器已创建，训练仅监督回答与 EOS。"); } }); }}>创建 SFT 实验</button>
      </fieldset><p>现有实验继续使用已保存的数据和配置，修改浏览器草稿不会改变它。</p></ExperimentPanel>}
      {run && <>
        <div className="language-metrics"><div><span>可训练 / 冻结参数</span><strong>{run.trainable_parameters.toLocaleString()} / {run.base_parameters.toLocaleString()}</strong></div><div><span>训练 / 验证样本</span><strong>{run.train_indices.length} / {run.validation_indices.length}</strong></div><div><span>当前 / 最佳步数</span><strong>{run.step} / {run.best_step}</strong></div></div>
        <p className="lora-frozen-status">{run.frozen_verified ? "冻结校验通过：原始基座参数保持不变。" : "冻结校验未通过。"} 损失只统计回答 token 与 EOS。</p>
        {mode === "train" && <>
          <ExperimentPanel title="真实回答监督微调"><p>AdamW 学习率 {run.config.learning_rate}，batch {run.config.batch_size}，rank {run.config.rank}，alpha {run.config.alpha}。右侧填充位置 attention_mask=0、label=−100；每步只更新 LoRA，梯度范数裁剪为 1。</p><div className="language-actions">{[1, 5, 20, 100].map(steps => <button key={steps} className={steps === 20 ? "primary-button" : "secondary-button"} disabled={pending || run.step + steps > 500} onClick={() => train(steps)}><Play size={14} />SFT {steps} 步</button>)}{training && <button className="secondary-button" onClick={() => { stop.current = true; }}><Pause size={14} />暂停后续 SFT</button>}</div><p>每批提交最多 5 步，保存后评估；最多 500 步。暂停或离开只阻止后续提交，当前批次仍可能完成，恢复前请先刷新记录。</p></ExperimentPanel>
          <ExperimentPanel title="只看回答的真实损失曲线"><div className="language-metrics"><div><span>训练回答交叉熵</span><strong>{last!.train_loss.toFixed(4)}</strong></div><div><span>验证回答交叉熵</span><strong>{last!.validation_loss.toFixed(4)}</strong></div><div><span>验证回答困惑度</span><strong>{last!.perplexity.toFixed(2)}</strong></div></div><LossChart history={run.history} /><p>评估覆盖两份完整问答集，按有效回答目标数加权平均；用户问题、系统提示和填充不进入分母。训练 / 验证有效目标共 {run.train_supervised_tokens} / {run.validation_supervised_tokens} 个。没有额外独立测试集，损失低不保证回答可靠。</p><details><summary>查看逐批保存指标</summary><div className="language-table"><table><thead><tr><th>步数</th><th>训练回答损失</th><th>验证回答损失</th></tr></thead><tbody>{run.history.map(row => <tr key={row.step}><td>{row.step}</td><td>{row.train_loss.toFixed(5)}</td><td>{row.validation_loss.toFixed(5)}</td></tr>)}</tbody></table></div></details><a className="text-button" href="#sft-chat">使用 SFT 适配器开始对话 <ArrowRight size={14} /></a></ExperimentPanel>
        </>}
        <details className="language-reading"><summary>实验数据与模板快照</summary><p>模板 {run.template_version}；系统提示：{run.config.system_prompt || "（无）"}</p><p>数据指纹：{run.dataset_fingerprint}</p><p>训练索引：{run.train_indices.join(", ")}；验证索引：{run.validation_indices.join(", ")}</p><pre>{JSON.stringify(run.config.dataset, null, 2)}</pre></details>
        {mode === "chat" && <>
          <ExperimentPanel title="SFT 多轮对话"><fieldset className="language-generation-controls" disabled={pending}><legend>固定模型与生成设置</legend><label className="language-field">SFT 适配器权重<select value={checkpoint} onChange={event => { setCheckpoint(event.target.value as "latest" | "best"); setMessages([]); setReply(null); }}><option value="latest">当前适配器 · 第 {run.step} 步</option><option value="best">最佳验证适配器 · 第 {run.best_step} 步</option></select></label><details><summary>调整生成参数</summary><Slider label="温度（0 为贪心）" value={temperature} min={0} max={2} step={0.1} onChange={setTemperature} /><Slider label="最多回答 token 数" value={length} min={8} max={64} step={8} onChange={setLength} /><label className="language-field">Top-k<select value={topK} onChange={event => setTopK(Number(event.target.value))}><option value={0}>全部候选</option><option value={10}>前 10 个候选</option><option value={50}>前 50 个候选</option></select></label><label className="language-field">生成种子<input type="number" min={0} max={4294967295} value={seed} onChange={event => setSeed(Number(event.target.value))} /></label><p>温度 0 使用贪心解码，此时 Top-k 和随机种子不影响选择。</p></details></fieldset>
            <p>训练与对话共用 {run.template_version}。当前对话只保存在本页；更换实验或权重会清空历史。请求最多携带最近 4 轮，服务按完整旧轮次进一步裁剪，并保留系统提示和最新问题。</p>
            {(checkpoint === "best" ? run.best_step : run.step) === 0 && <p>所选适配器还未训练，对话行为尚未学到。</p>}
            <div className="sft-chat-log language-lab" role="log" aria-label="对话记录" aria-live="polite">{messages.length ? messages.map((message, index) => <article key={index} className={`language-output sft-message is-${message.role}`}><strong>{message.role === "user" ? "你" : "SFT 助手"}</strong><p>{message.content}</p></article>) : <p>从训练问答中的问题开始，观察模型是否学会回答格式。</p>}</div>
            <div className="language-actions"><button className="secondary-button" disabled={pending || !messages.length} onClick={() => { setMessages([]); setReply(null); setNotice("当前对话已清空。"); }}>开始新对话</button><button className="secondary-button" disabled={!messages.length} onClick={() => downloadJson({ run_id: run.id, checkpoint, messages }, "zhiwei-sft-conversation.json")}><Download size={14} />导出对话</button></div>
            <div className="language-actions">{run.train_indices.slice(0, 3).map(index => { const text = recordQuestion(run.config.dataset[index]); return <button key={index} className="text-button" disabled={pending} onClick={() => setQuestion(text)}>{text}</button>; })}</div>
            <form onSubmit={event => { event.preventDefault(); send(); }}><label className="language-field">发送问题<textarea rows={3} value={question} maxLength={1000} disabled={pending} onChange={event => setQuestion(event.target.value)} /></label><button className="primary-button" disabled={pending || !question.trim() || !Number.isInteger(seed) || seed < 0 || seed > 4294967295}><Send size={14} />发送并生成回答</button></form>
            {reply && <p className="sft-reply-meta" role="status">实际使用第 {reply.step} 步{reply.checkpoint === "best" ? "最佳验证" : "当前"}适配器；上下文 {reply.context_tokens} 个 token，生成 {reply.token_ids.length} 个，停止原因：{({ eos: "EOS", length: "回答长度上限", context_limit: "上下文长度上限", role_boundary: "角色边界" } as Record<string, string>)[reply.finish_reason] ?? reply.finish_reason}。</p>}
          </ExperimentPanel>
        </>}
        <div className="language-actions"><a className="secondary-button" href={`/api/sft/runs/${run.id}/adapter?checkpoint=${checkpoint}`} download><Download size={14} />下载 SFT 适配器与对话模板</a><a className="secondary-button" href={`/api/sft/runs/${run.id}/checkpoint`} download><Download size={14} />下载 SFT 续训检查点</a></div>
      </>}
    </>}
    <details className="language-reading"><summary>SFT 原理与官方资料</summary><p>SFT 的区别在训练数据与监督目标；LoRA 是参数更新方法。这里把二者组合，沿用标准因果预测，只让助手回答和结束 token 的标签参与损失。</p><p>参考 <a href="https://huggingface.co/docs/trl/sft_trainer" target="_blank" rel="noreferrer">SFT 数据与回答监督</a>和<a href="https://huggingface.co/docs/transformers/chat_templating" target="_blank" rel="noreferrer">对话模板说明</a>。本项目用 PyTorch 实现显式回答掩码，不依赖 TRL Trainer。</p></details>
  </div>;
}

export function SFTData(props: { onInteract: () => void }) { return <SFTExperiment mode="data" {...props} />; }
export function SFTTraining(props: { onInteract: () => void }) { return <SFTExperiment mode="train" {...props} />; }
export function SFTChat(props: { onInteract: () => void }) { return <SFTExperiment mode="chat" {...props} />; }
