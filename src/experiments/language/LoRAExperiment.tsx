import { useEffect, useRef, useState } from "react";
import { ArrowRight, Download, Pause, Play, RefreshCw } from "lucide-react";
import { ExperimentPanel, Slider } from "../../components/ui";
import { useLessonTask } from "../../lib/learningProgress";
import { loraApi, loraDefaults } from "../../lib/lora";
import type { LoRAConfig, LoRAStatus, LoRARun, LoRAGeneration } from "../../lib/lora";
import { LossChart } from "./LossChart";
import "./language.css";
import "./transformer.css";
import "./lora.css";

const storageKey = "zhiwei-lora-run-v1";
function storedId() { try { return localStorage.getItem(storageKey); } catch { return null; } }

function LoRAExperiment({ mode, onInteract }: { mode: "basics" | "train" | "generate"; onInteract: () => void }) {
  const [status, setStatus] = useState<LoRAStatus | null>(null);
  const [config, setConfig] = useState<LoRAConfig>(loraDefaults);
  const [runs, setRuns] = useState<LoRARun[]>([]);
  const [run, setRun] = useState<LoRARun | null>(null);
  const [pending, setPending] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [inspected, setInspected] = useState(false);
  const [rank, setRank] = useState(4);
  const [prompt, setPrompt] = useState("Zhiwei is");
  const [temperature, setTemperature] = useState(0.8);
  const [length, setLength] = useState(32);
  const [seed, setSeed] = useState(7);
  const [topK, setTopK] = useState(10);
  const [checkpoint, setCheckpoint] = useState<"latest" | "best">("latest");
  const [generation, setGeneration] = useState<LoRAGeneration | null>(null);
  const current = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const busy = useRef(false);
  const stop = useRef(false);
  const last = run?.history.at(-1);
  useLessonTask("lora-basics", mode === "basics" && status?.state === "ready" && inspected);
  useLessonTask("lora-train", mode === "train" && Boolean(run && run.step >= 2 && run.frozen_verified && last!.train_loss < run.history[0].train_loss));
  useLessonTask("lora-generate", mode === "generate" && Boolean(generation && generation.step > 0 && generation.adapted.text.trim()));

  function selectRun(next: LoRARun) {
    setRun(next); setGeneration(null); setRuns(previous => [next, ...previous.filter(item => item.id !== next.id)].slice(0, 30));
    try { localStorage.setItem(storageKey, next.id); } catch { /* Adapter checkpoints remain on the server. */ }
  }
  async function action(work: (signal: AbortSignal) => Promise<void>) {
    if (busy.current) return;
    busy.current = true; setPending(true); setError(""); setNotice("");
    const controller = new AbortController(); current.current = controller;
    try { await work(controller.signal); }
    catch (reason) { if (alive.current && !controller.signal.aborted) setError(`${reason instanceof Error ? reason.message : "计算失败"}。若训练请求中断，请先刷新记录再继续。`); }
    finally {
      if (current.current === controller) {
        current.current = null; busy.current = false;
        if (alive.current) { setPending(false); setTraining(false); }
      }
    }
  }
  useEffect(() => {
    alive.current = true;
    void action(async signal => {
      const [info, records] = await Promise.all([loraApi<LoRAStatus>("/status", undefined, signal), loraApi<LoRARun[]>("/runs", undefined, signal)]);
      if (signal.aborted || !alive.current) return;
      setStatus(info); setConfig(previous => ({ ...previous, corpus: info.corpus ?? "" })); setRuns(records);
      const selected = records.find(item => item.id === storedId()) ?? records[0];
      if (selected) selectRun(selected);
    });
    return () => { alive.current = false; stop.current = true; current.current?.abort(); busy.current = false; };
    // Restore the saved server-side experiment once per mounted lesson.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  useEffect(() => {
    if (status?.state !== "downloading") return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await loraApi<LoRAStatus>("/status", undefined, controller.signal);
        if (!controller.signal.aborted) setStatus(result);
      } catch { /* A later status refresh can recover a temporary network error. */ }
      if (!controller.signal.aborted) timer = window.setTimeout(() => void poll(), 2000);
    }
    timer = window.setTimeout(() => void poll(), 1000);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [status?.state]);

  function refresh() {
    void action(async signal => {
      const [info, records] = await Promise.all([loraApi<LoRAStatus>("/status", undefined, signal), loraApi<LoRARun[]>("/runs", undefined, signal)]);
      if (signal.aborted || !alive.current) return;
      setStatus(info); setRuns(records);
      const selected = records.find(item => item.id === run?.id) ?? records[0];
      if (selected) selectRun(selected); else { setRun(null); setGeneration(null); }
      if (!config.corpus && info.corpus) setConfig(previous => ({ ...previous, corpus: info.corpus! }));
      setNotice("已刷新预训练模型状态与微调记录。");
    });
  }
  function train(steps: number) {
    if (!run) return;
    stop.current = false; setTraining(true); onInteract();
    void action(async signal => {
      let latest = run;
      for (let remaining = steps; remaining > 0 && !stop.current; remaining -= 5) {
        latest = await loraApi<LoRARun>(`/runs/${latest.id}/train`, { expected_step: latest.step, steps: Math.min(5, remaining) }, signal);
        if (signal.aborted || !alive.current) return;
        selectRun(latest);
      }
      setNotice(`${stop.current ? "已暂停" : "微调已保存"}：当前 ${latest.step} 步；最佳验证适配器为第 ${latest.best_step} 步。`);
    });
  }
  const edit = (key: keyof LoRAConfig, value: string | number) => setConfig(previous => ({ ...previous, [key]: value }));
  const change = (work: () => void) => { setGeneration(null); work(); };

  return <div className="language-lab">
    <div className="language-intro"><span className="small-label">PRETRAINED PYTHIA-14M · PEFT LORA · CPU</span><p>加载公开预训练基座，冻结原有参数，只学习低秩更新 ΔW=(α/r)BA。本课程使用英文领域文本做因果语言建模微调，保留原模型的子词分词器。</p></div>
    {error && <div className="connection-notice" role="alert">{error}</div>}
    <p className="language-status" role="status">{pending ? training ? "正在真实微调，仅更新 LoRA 参数，完成一批就保存…" : "正在读取或计算…" : notice}</p>
    <ExperimentPanel title="公开预训练基座与缓存" action={<button className="secondary-button" disabled={pending} onClick={refresh}><RefreshCw size={14} />刷新记录</button>}>
      <p><strong>{status?.model_id ?? "EleutherAI/pythia-14m"}</strong> · 约 14M 参数 · 英文基础语言模型 · CPU FP32</p>
      <p>固定版本 <code>{status?.revision.slice(0, 12) ?? "…"}</code>，采用原始预训练 Safetensors 权重。首次需从 Hugging Face 下载约 28 MB 权重及分词器，准备完成后可离线使用。</p>
      <p role="status">{status?.message ?? "正在检查依赖与缓存…"}</p>
      {status && !status.available && <p role="alert">缺少依赖：{Object.entries(status.dependencies).filter(([, ready]) => !ready).map(([name]) => name).join("、")}。请安装 requirements.txt 后重启服务。</p>}
      <button className="primary-button" disabled={pending || !status?.available || status.state === "downloading"} onClick={() => { onInteract(); void action(async signal => { const info = await loraApi<LoRAStatus>("/prepare", {}, signal); if (!signal.aborted && alive.current) setStatus(info); }); }}>{status?.state === "ready" ? "检查预训练缓存" : status?.state === "downloading" ? "正在下载预训练模型…" : "下载并准备预训练模型"}</button>
      <a className="text-button" href="https://huggingface.co/EleutherAI/pythia-14m" target="_blank" rel="noreferrer">查看模型来源与说明 <ArrowRight size={14} /></a>
    </ExperimentPanel>
    {mode === "basics" && <ExperimentPanel title="冻结 W，只训练 A 与 B">
      <p>Pythia-14M 每层的 query_key_value 是 384×128 的线性矩阵。LoRA 旁路先用 A 把 128 维投影到 r 维，再用 B 投影到 384 维；原始 W 不更新。</p>
      <div className="transformer-token-row">{[2, 4, 8, 16].map(value => <button key={value} className="secondary-button" aria-pressed={rank === value} onClick={() => { setRank(value); setInspected(true); onInteract(); }}>rank = {value}</button>)}</div>
      <div className="lora-path"><div><strong>基座路径 · 冻结</strong><span>x → W[384,128] → Wx</span></div><div><strong>LoRA 旁路 · 可训练</strong><span>x → A[{rank},128] → B[384,{rank}] → (α/r)BAx</span></div><div><strong>相加得到输出</strong><span>y = Wx + (α/r)BAx</span></div></div>
      <div className="language-metrics"><div><span>单个 W 参数量</span><strong>{(384 * 128).toLocaleString()}</strong></div><div><span>单层 A + B</span><strong>{(rank * (128 + 384)).toLocaleString()}</strong></div><div><span>六层可训练参数</span><strong>{(6 * rank * (128 + 384)).toLocaleString()}</strong></div></div>
      <p>固定 α=8 时，当前缩放 α/r = {(8 / rank).toFixed(2)}。A 使用局部随机初始化，B 初始为 0，因此开始时 LoRA 更新为 0，预测与原基座一致。</p>
      <p>低秩减少的是可训练参数与优化器状态，仍然需要加载整个基座。rank 更高不保证验证损失或生成质量更好。</p><a className="text-button" href="#lora-train">创建真实微调实验 <ArrowRight size={14} /></a>
    </ExperimentPanel>}
    {mode !== "basics" && <>
      <ExperimentPanel title="本机 LoRA 微调实验"><label className="language-field">选择微调实验<select disabled={pending || !runs.length} value={run?.id ?? ""} onChange={event => { const item = runs.find(row => row.id === event.target.value); if (item) selectRun(item); }}>
        {!runs.length && <option value="">尚无实验，请先创建适配器</option>}{runs.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · r={item.config.rank} · {item.step} 步</option>)}
      </select></label><p>适配器、最佳权重、AdamW 状态与随机窗口采样状态保存到本机 runs/lora；基座按固定版本引用，不重复保存。</p>{!run && mode === "generate" && <a className="text-button" href="#lora-train">前往真实微调 <ArrowRight size={14} /></a>}</ExperimentPanel>
      {mode === "train" && <ExperimentPanel title="创建 LoRA 适配器与语料快照"><fieldset className="language-generation-controls" disabled={pending}><legend>新实验配置</legend>
        <label className="language-field">微调文本<textarea rows={6} value={config.corpus} maxLength={6000} onChange={event => edit("corpus", event.target.value)} /></label>
        <p>100–6,000 个字符，先按原文前 80% / 后 20% 划分，再分别用预训练分词器编码。默认文本是英文领域续写示例，不是对话指令模板。</p>
        <div className="transformer-config-grid">{[["rank", "LoRA rank", [2, 4, 8, 16]], ["alpha", "LoRA alpha", [4, 8, 16, 32]], ["context_length", "上下文长度", [16, 32, 64]], ["batch_size", "批大小", [1, 2, 4]]].map(([key, label, values]) => <label className="language-field" key={key as string}>{label as string}<select value={config[key as keyof LoRAConfig]} onChange={event => edit(key as keyof LoRAConfig, Number(event.target.value))}>{(values as number[]).map(value => <option key={value} value={value}>{value}</option>)}</select></label>)}
          <label className="language-field">学习率<input type="number" min={0.0001} max={0.01} step={0.0001} value={config.learning_rate} onChange={event => edit("learning_rate", Number(event.target.value))} /></label>
          <label className="language-field">初始化种子<input type="number" min={0} max={4294967295} value={config.seed} onChange={event => edit("seed", Number(event.target.value))} /></label></div>
        <button className="primary-button" disabled={status?.state !== "ready" || Array.from(config.corpus).length < 100 || !Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295 || config.learning_rate < 0.0001 || config.learning_rate > 0.01} onClick={() => { onInteract(); void action(async signal => { const next = await loraApi<LoRARun>("/runs", config, signal); if (!signal.aborted && alive.current) { selectRun(next); setNotice("预训练基座与 LoRA 适配器已载入，可开始微调。"); } }); }}>创建微调实验</button>
      </fieldset><p>目标模块固定为六层 query_key_value；LoRA dropout=0、bias=none。修改配置只影响新实验。</p></ExperimentPanel>}
      {run && <>
        <div className="language-metrics"><div><span>冻结基座参数</span><strong>{run.base_parameters.toLocaleString()}</strong></div><div><span>可训练 LoRA 参数</span><strong>{run.trainable_parameters.toLocaleString()}</strong></div><div><span>可训练占比</span><strong>{(run.trainable_ratio * 100).toFixed(3)}%</strong></div><div><span>当前 / 最佳步数</span><strong>{run.step} / {run.best_step}</strong></div></div>
        <p className="lora-frozen-status">{run.frozen_verified ? "冻结校验通过：每批保存前均比较基座参数的 SHA-256，原参数保持不变。" : "冻结校验未通过。"}</p>
        {mode === "train" && <>
          <ExperimentPanel title="真实 LoRA 微调控制台"><p>实验 {run.id.slice(0, 8)} · r={run.config.rank}、α={run.config.alpha}，AdamW 学习率 {run.config.learning_rate}；只优化 LoRA 参数，梯度范数裁剪为 1。训练 / 验证 token 数 {run.train_tokens} / {run.validation_tokens}，实际训练窗口 {run.training_context}。</p><div className="language-actions">{[1, 5, 20].map(steps => <button className={steps === 20 ? "primary-button" : "secondary-button"} disabled={pending || run.step + steps > 500} key={steps} onClick={() => train(steps)}><Play size={14} />微调 {steps} 步</button>)}{training && <button className="secondary-button" onClick={() => { stop.current = true; }}><Pause size={14} />暂停后续微调</button>}</div><p>每次请求最多 5 步，完成后评估并原子保存。暂停或离开会停止后续提交，当前请求仍可能保存；恢复前先刷新记录。单实验最多 500 步。</p></ExperimentPanel>
          <ExperimentPanel title="真实微调曲线与验证"><div className="language-metrics"><div><span>训练交叉熵</span><strong>{last!.train_loss.toFixed(4)}</strong></div><div><span>验证交叉熵</span><strong>{last!.validation_loss.toFixed(4)}</strong></div><div><span>验证困惑度</span><strong>{last!.perplexity.toFixed(2)}</strong></div></div><LossChart history={run.history} /><p>第 0 步对应未微调的真实预训练基座。每批结束后，对训练和验证文本分别分块评估，每个下一 token 目标计一次；块间重置上下文。验证不反传，没有额外独立测试集。</p><details><summary>查看微调指标</summary><div className="language-table"><table><thead><tr><th>步数</th><th>训练损失</th><th>验证损失</th><th>验证困惑度</th></tr></thead><tbody>{run.history.map(row => <tr key={row.step}><td>{row.step}</td><td>{row.train_loss.toFixed(5)}</td><td>{row.validation_loss.toFixed(5)}</td><td>{row.perplexity.toFixed(3)}</td></tr>)}</tbody></table></div></details><a className="text-button" href="#lora-generate">对比基座与微调后输出 <ArrowRight size={14} /></a></ExperimentPanel>
        </>}
        <details className="language-reading"><summary>查看语料快照与模型来源</summary><p>{run.model_id} · revision {run.revision}</p><p className="lora-digest">原始权重 SHA-256：{run.weight_sha256}</p><h3>训练文本</h3><pre>{Array.from(run.config.corpus).slice(0, Math.floor(Array.from(run.config.corpus).length * 0.8)).join("")}</pre><h3>验证文本</h3><pre>{Array.from(run.config.corpus).slice(Math.floor(Array.from(run.config.corpus).length * 0.8)).join("")}</pre></details>
        {mode === "generate" && <>
          <ExperimentPanel title="同一提示词，对比微调前后"><fieldset className="language-generation-controls" disabled={pending}><legend>生成设置</legend>
            <label className="language-field">适配器权重<select value={checkpoint} onChange={event => change(() => setCheckpoint(event.target.value as "latest" | "best"))}><option value="latest">当前适配器 · 第 {run.step} 步</option><option value="best">最佳验证适配器 · 第 {run.best_step} 步</option></select></label>
            {(checkpoint === "best" ? run.best_step : run.step) === 0 && <p>此适配器尚未微调，初始 B=0，输出应与基座一致。</p>}
            <label className="language-field">提示词<input value={prompt} maxLength={256} onChange={event => change(() => setPrompt(event.target.value))} /></label>
            <Slider label="采样温度" value={temperature} min={0.1} max={2} step={0.1} onChange={value => change(() => setTemperature(value))} />
            <Slider label="最多生成 token 数" value={length} min={8} max={64} step={8} onChange={value => change(() => setLength(value))} />
            <label className="language-field">Top-k<select value={topK} onChange={event => change(() => setTopK(Number(event.target.value)))}><option value={0}>全部候选</option><option value={10}>前 10 个候选</option><option value={50}>前 50 个候选</option></select></label>
            <label className="language-field">采样种子<input type="number" min={0} max={4294967295} value={seed} onChange={event => change(() => setSeed(Number(event.target.value)))} /></label>
            <button className="primary-button" disabled={!prompt.trim() || !Number.isInteger(seed) || seed < 0 || seed > 4294967295} onClick={() => { onInteract(); void action(async signal => { const result = await loraApi<LoRAGeneration>(`/runs/${run.id}/generate`, { prompt, length, temperature, top_k: topK, seed, checkpoint }, signal); if (!signal.aborted && alive.current) setGeneration(result); }); }}><Play size={14} />生成并对比</button>
          </fieldset><p>两列使用相同提示词、采样设置与独立的同种子 RNG；左侧关闭适配器，右侧启用所选适配器。每步使用最近 {run.config.context_length} 个 token，遇 EOS 提前结束。小模型输出可能重复或不通顺。</p></ExperimentPanel>
          {generation && <ExperimentPanel title="真实基座与 LoRA 输出"><p>基座 {generation.model_id} · 适配器第 {generation.step} 步 · {generation.checkpoint === "best" ? "最佳验证" : "当前"}权重 · 首次实际上下文 {generation.context_tokens} 个 token</p><div className="lora-comparison">{[["预训练基座 · 适配器关闭", generation.baseline], ["LoRA 微调 · 适配器启用", generation.adapted]].map(([title, output]) => <section key={title as string}><h3>{title as string}</h3><pre className="language-output"><span>{generation.prompt}</span>{(output as LoRAGeneration["adapted"]).text}</pre><small>生成 {(output as LoRAGeneration["adapted"]).token_ids.length} 个 token · {(output as LoRAGeneration["adapted"]).finish_reason === "eos" ? "遇到 EOS" : "达到长度上限"}</small></section>)}</div></ExperimentPanel>}
        </>}
        <div className="language-actions"><a className="secondary-button" href={`/api/lora/runs/${run.id}/adapter?checkpoint=${checkpoint}`} download><Download size={14} />下载{checkpoint === "best" ? "最佳" : "当前"} PEFT 适配器 ZIP</a><a className="secondary-button" href={`/api/lora/runs/${run.id}/checkpoint`} download><Download size={14} />下载续训检查点</a></div><p>适配器包只包含低秩权重与配置；重新加载时仍需要固定版本的 Pythia-14M 基座。</p>
      </>}
    </>}
    <details className="language-reading"><summary>LoRA 公式与官方资料</summary><p>标准 LoRA 在冻结线性映射上增加 (α/r)BA。A 的形状为 [r, d_in]，B 为 [d_out, r]；本实现只训练六层融合 QKV 的 A、B，不训练偏置或词嵌入，不使用量化。</p><p>参考 <a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">LoRA 原论文</a>、<a href="https://huggingface.co/docs/peft/developer_guides/lora" target="_blank" rel="noreferrer">PEFT LoRA 文档</a>与<a href="https://huggingface.co/docs/peft/developer_guides/checkpoint" target="_blank" rel="noreferrer">适配器保存格式</a>。</p></details>
  </div>;
}

export function LoRABasics(props: { onInteract: () => void }) { return <LoRAExperiment mode="basics" {...props} />; }
export function LoRATraining(props: { onInteract: () => void }) { return <LoRAExperiment mode="train" {...props} />; }
export function LoRAGenerationLesson(props: { onInteract: () => void }) { return <LoRAExperiment mode="generate" {...props} />; }
