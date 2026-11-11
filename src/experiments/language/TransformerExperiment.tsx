import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { ArrowRight, Download, Pause, Play, RefreshCw } from "lucide-react";
import { ExperimentPanel, Slider } from "../../components/ui";
import { useLessonTask } from "../../lib/learningProgress";
import { displayLanguageToken as tokenLabel, transformerApi, transformerDefaults } from "../../lib/transformer";
import type { TransformerConfig, TransformerRun, TransformerGeneration } from "../../lib/transformer";
import { LossChart } from "./LossChart";
import "./language.css";
import "./transformer.css";

const storedRunKey = "zhiwei-transformer-run-v1";
const maskTokens = Array.from("小猫在花园里看花");
function storedRun() { try { return localStorage.getItem(storedRunKey); } catch { return null; } }

function CausalArchitecture({ onInteract }: { onInteract: () => void }) {
  const [position, setPosition] = useState(3);
  const [inspected, setInspected] = useState(false);
  useLessonTask("transformer-architecture", inspected);
  return <>
    <ExperimentPanel title="从 Token 到下一 Token 的分数">
      <ol className="transformer-pipeline">{[
        ["字符 ID", "[B, T]，输入是整数索引"], ["Token + 位置嵌入", "[B, T, D]，同时表达字符与顺序"],
        ["因果自注意力", "分成 H 个头；当前位置只能读取自身与过去"], ["残差 + 前馈网络", "Pre-LN，每个位置经过 D → 4D → D 的 GELU 网络"],
        ["LayerNorm + 输出层", "[B, T, V] logits，每个位置预测下一个字符"],
      ].map(([title, description]) => <li key={title}><strong>{title}</strong><span>{description}</span></li>)}</ol>
      <p>本课程实现 decoder-only Transformer：1–2 个块，32 或 64 维嵌入，2 或 4 个注意力头，上下文 8 / 16 / 32 个字符。使用学习式位置嵌入，不含跨注意力和 Dropout。</p>
    </ExperimentPanel>
    <ExperimentPanel title="亲手检查因果掩码">
      <p>点击一个输入位置，查看它能读取哪些字符。这里展示结构规则，生成课会展示模型实际计算的注意力权重。</p>
      <div className="transformer-token-row">{maskTokens.map((token, index) => <button className="secondary-button" key={index} aria-pressed={position === index} onClick={() => { setPosition(index); setInspected(true); onInteract(); }}>{index + 1} · {token}</button>)}</div>
      <div className="transformer-mask" role="img" aria-label={`位置 ${position + 1} 只能读取前 ${position + 1} 个字符，后面的字符全部屏蔽`}>
        {maskTokens.map((_, row) => <div key={row}>{maskTokens.map((token, column) => <span key={column} className={`${column <= row ? "is-visible" : "is-masked"} ${row === position ? "is-inspected" : ""}`} title={`位置 ${row + 1} → ${token}：${column <= row ? "可见" : "屏蔽"}`}>{column <= row ? "可见" : "×"}</span>)}</div>)}
      </div>
      <p role="status">当前位置 {position + 1}「{maskTokens[position]}」可读取「{maskTokens.slice(0, position + 1).join("")}」，用来预测下一个字符。未来位置在 Softmax 前设为 −∞，概率为 0。</p>
      <a className="text-button" href="#transformer-train">创建并训练 Transformer <ArrowRight size={14} /></a>
    </ExperimentPanel>
  </>;
}

function AttentionInspector({ generation }: { generation: TransformerGeneration }) {
  const [head, setHead] = useState(0);
  const [position, setPosition] = useState(generation.context_tokens.length - 1);
  const weights = generation.attention[head][position];
  return <div className="transformer-attention">
    <h3>真实因果注意力 · 最后一个 Transformer 块</h3>
    <p>这是生成第一个新 token 时，第 {generation.attention_layer} 层的实际注意力。选择注意力头与查询位置，查看它如何分配权重；注意力不是因果解释。</p>
    <label className="language-field">注意力头<select value={head} onChange={event => setHead(Number(event.target.value))}>{generation.attention.map((_, index) => <option key={index} value={index}>Head {index + 1}</option>)}</select></label>
    <div className="transformer-token-row">{generation.context_tokens.map((token, index) => <button className="secondary-button" key={index} aria-pressed={position === index} onClick={() => setPosition(index)}>{index + 1} · {tokenLabel(token)}</button>)}</div>
    <div className="transformer-attention-row">{generation.context_tokens.map((token, index) => <div key={index} style={{ "--attention-strength": `${Math.round(weights[index] * 100)}%` } as CSSProperties} className={index > position ? "is-masked" : ""}><strong>{tokenLabel(token)}</strong><span>{(weights[index] * 100).toFixed(2)}%</span>{index > position && <small>未来 · 屏蔽</small>}</div>)}</div>
    <p>当前行权重总和：{weights.reduce((sum, value) => sum + value, 0).toFixed(6)}。因果掩码保证右侧未来位置权重为 0。</p>
  </div>;
}

function TransformerExperiment({ mode, onInteract }: { mode: "architecture" | "train" | "generate"; onInteract: () => void }) {
  const [config, setConfig] = useState<TransformerConfig>(transformerDefaults);
  const [runs, setRuns] = useState<TransformerRun[]>([]);
  const [run, setRun] = useState<TransformerRun | null>(null);
  const [pending, setPending] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [prompt, setPrompt] = useState("小猫");
  const [temperature, setTemperature] = useState(0.8);
  const [length, setLength] = useState(60);
  const [seed, setSeed] = useState(7);
  const [topK, setTopK] = useState(0);
  const [checkpoint, setCheckpoint] = useState<"latest" | "best">("latest");
  const [generation, setGeneration] = useState<TransformerGeneration | null>(null);
  const currentController = useRef<AbortController | null>(null);
  const stop = useRef(false);
  const alive = useRef(true);
  const busy = useRef(false);
  const last = run?.history.at(-1);
  useLessonTask("transformer-train", mode === "train" && Boolean(run && run.step >= 3 && last!.train_loss < run.history[0].train_loss));
  useLessonTask("transformer-generate", mode === "generate" && Boolean(generation && generation.step > 0));

  function selectRun(next: TransformerRun) {
    setRun(next); setGeneration(null);
    setRuns(previous => [next, ...previous.filter(item => item.id !== next.id)].slice(0, 30));
    try { localStorage.setItem(storedRunKey, next.id); } catch { /* Checkpoints stay on the server. */ }
  }
  async function action(work: (signal: AbortSignal) => Promise<void>) {
    if (busy.current) return;
    busy.current = true; setPending(true); setError(""); setNotice("");
    const controller = new AbortController(); currentController.current = controller;
    try { await work(controller.signal); }
    catch (reason) { if (alive.current && !controller.signal.aborted) setError(`${reason instanceof Error ? reason.message : "计算失败"}。若请求中断，请先刷新记录，当前批次可能已保存。`); }
    finally {
      if (currentController.current === controller) {
        currentController.current = null; busy.current = false;
        if (alive.current) { setPending(false); setTraining(false); }
      }
    }
  }
  useEffect(() => {
    alive.current = true;
    if (mode !== "architecture") {
      void action(async signal => {
        const [status, records] = await Promise.all([
          transformerApi<{ corpus: string; available: boolean }>("/status", undefined, signal),
          transformerApi<TransformerRun[]>("/runs", undefined, signal),
        ]);
        if (signal.aborted || !alive.current) return;
        setConfig(previous => ({ ...previous, corpus: status.corpus })); setRuns(records);
        const selected = records.find(item => item.id === storedRun()) ?? records[0];
        if (selected) selectRun(selected);
        if (!status.available) setError("PyTorch 未安装，请安装 requirements.txt 中的依赖。");
      });
    }
    return () => { alive.current = false; stop.current = true; currentController.current?.abort(); busy.current = false; };
    // Each mode mounts independently. A saved run is restored without starting training.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  function train(steps: number) {
    if (!run) return;
    stop.current = false; setTraining(true); onInteract();
    void action(async signal => {
      let latest = run;
      for (let remaining = steps; remaining > 0 && !stop.current; remaining -= 25) {
        latest = await transformerApi<TransformerRun>(`/runs/${latest.id}/train`, { expected_step: latest.step, steps: Math.min(25, remaining) }, signal);
        if (signal.aborted || !alive.current) return;
        selectRun(latest);
      }
      setNotice(`${stop.current ? "已暂停" : "训练已保存"}：当前 ${latest.step} 步，最佳验证权重为第 ${latest.best_step} 步。`);
    });
  }
  function refresh() {
    void action(async signal => {
      const records = await transformerApi<TransformerRun[]>("/runs", undefined, signal);
      if (signal.aborted || !alive.current) return;
      setRuns(records);
      const selected = records.find(item => item.id === run?.id) ?? records[0];
      if (selected) selectRun(selected);
      else { setRun(null); setGeneration(null); }
      setNotice("已从本机检查点刷新。");
    });
  }
  const invalidPrompt = Boolean(run && Array.from(prompt).some(char => !run.vocab.includes(char)));
  const updateConfig = (field: keyof TransformerConfig, value: string | number) => setConfig(previous => ({ ...previous, [field]: value }));
  const changeGeneration = (change: () => void) => { setGeneration(null); change(); };
  return <div className="language-lab">
    <div className="language-intro"><span className="small-label">DECODER-ONLY TRANSFORMER · PYTORCH / CPU</span><p>从随机参数开始真实训练：Token 与位置嵌入 → 因果多头注意力 → 前馈网络 → 下一字符 logits。小语料用于理解计算，不保证生成内容通顺。</p></div>
    {error && <div className="connection-notice" role="alert">{error}</div>}
    <p className="language-status" role="status">{pending ? training ? "正在真实训练，完成一批就保存检查点…" : "正在读取或计算…" : notice}</p>
    {mode === "architecture" ? <CausalArchitecture onInteract={onInteract} /> : <>
      <ExperimentPanel title="本机 Transformer 实验记录" action={<button className="secondary-button" disabled={pending} onClick={refresh}><RefreshCw size={14} />刷新记录</button>}>
        <label className="language-field">选择实验<select disabled={pending || runs.length === 0} value={run?.id ?? ""} onChange={event => { const next = runs.find(item => item.id === event.target.value); if (next) selectRun(next); }}>
          {!runs.length && <option value="">尚无实验，请先创建模型</option>}{runs.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.config.layers} 层 / {item.config.d_model} 维 · {item.step} 步</option>)}
        </select></label>
        <p>语料、当前权重、最低验证损失权重、AdamW 状态和随机采样状态保存在本机 runs/transformer。刷新或重启服务后可以继续训练。</p>
        {!run && mode === "generate" && <a className="text-button" href="#transformer-train">前往创建并训练模型 <ArrowRight size={14} /></a>}
      </ExperimentPanel>
      {mode === "train" && <ExperimentPanel title="创建新模型与语料快照">
        <fieldset className="language-generation-controls" disabled={pending}><legend>训练配置</legend>
          <label className="language-field">训练语料<textarea rows={6} value={config.corpus} maxLength={2000} onChange={event => updateConfig("corpus", event.target.value)} /></label>
          <p>40–2,000 个 Unicode 字符，训练部分含 2–95 种字符。前 80% 训练、后 20% 验证，词表仅用训练部分建立。</p>
          <div className="transformer-config-grid">{[
            ["context_length", "上下文长度", [8, 16, 32]], ["d_model", "嵌入维度", [32, 64]], ["heads", "注意力头数", [2, 4]],
            ["layers", "Transformer 层数", [1, 2]], ["batch_size", "批大小", [8, 16, 32]],
          ].map(([field, label, values]) => <label className="language-field" key={field as string}>{label as string}<select value={config[field as keyof TransformerConfig]} onChange={event => updateConfig(field as keyof TransformerConfig, Number(event.target.value))}>{(values as number[]).map(value => <option key={value} value={value}>{value}</option>)}</select></label>)}
          <label className="language-field">学习率<input type="number" min={0.0001} max={0.01} step={0.0001} value={config.learning_rate} onChange={event => updateConfig("learning_rate", Number(event.target.value))} /></label>
          <label className="language-field">初始化种子<input type="number" min={0} max={4294967295} value={config.seed} onChange={event => updateConfig("seed", Number(event.target.value))} /></label></div>
          <button className="primary-button" disabled={Array.from(config.corpus).length < 40 || !Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295 || config.learning_rate < 0.0001 || config.learning_rate > 0.01} onClick={() => { onInteract(); void action(async signal => {
            const next = await transformerApi<TransformerRun>("/runs", config, signal);
            if (!signal.aborted && alive.current) { selectRun(next); setNotice("模型已初始化，可单步或连续训练。"); }
          }); }}>创建新实验</button>
        </fieldset><p>修改这些配置只影响新实验。选中实验按创建时的配置与语料快照继续训练。</p>
      </ExperimentPanel>}
      {run && <>
        <div className="language-metrics"><div><span>可训练参数</span><strong>{run.parameter_count.toLocaleString()}</strong></div><div><span>上下文 / 层 / 头</span><strong>{run.config.context_length} / {run.config.layers} / {run.config.heads}</strong></div><div><span>训练 / 验证字符</span><strong>{run.train_tokens} / {run.validation_tokens}</strong></div><div><span>当前 / 最佳步数</span><strong>{run.step} / {run.best_step}</strong></div></div>
        {mode === "train" && <>
          <ExperimentPanel title="真实训练控制台">
            <p>当前实验 {run.id.slice(0, 8)}：AdamW，学习率 {run.config.learning_rate}，批大小 {run.config.batch_size}，seed {run.config.seed}。每批随机抽取长度 {run.training_context} 的训练窗口，梯度裁剪上限 1；无 Dropout。</p>
            <div className="language-actions">{[1, 25, 100].map(steps => <button key={steps} className={steps === 100 ? "primary-button" : "secondary-button"} disabled={pending || run.step + steps > 2000} onClick={() => train(steps)}><Play size={14} />训练 {steps} 步</button>)}{training && <button className="secondary-button" onClick={() => { stop.current = true; setNotice("将在当前批次保存后暂停。"); }}><Pause size={14} />暂停后续训练</button>}</div>
            <p>连续训练每次提交最多 25 步，完成一批后评估并原子保存。暂停或离开页面会停止后续提交；已提交批次仍可能完成。单实验最多 2,000 步。</p>
          </ExperimentPanel>
          <ExperimentPanel title="真实损失、验证与最佳权重"><div className="language-metrics"><div><span>训练交叉熵</span><strong>{last!.train_loss.toFixed(4)}</strong></div><div><span>验证交叉熵</span><strong>{last!.validation_loss.toFixed(4)}</strong></div><div><span>验证困惑度</span><strong>{last!.perplexity.toFixed(2)}</strong></div></div><LossChart history={run.history} />
            <p>曲线点是每个已保存批次结束后，对两份完整文本分块计算的损失，不是训练批次的瞬时损失。分块内保持因果性，每个下一字符目标计一次；块间重新开始上下文。验证不反传，也没有独立测试集。</p>
            <details><summary>查看逐次保存的指标</summary><div className="language-table"><table><thead><tr><th>步数</th><th>训练损失</th><th>验证损失</th><th>验证困惑度</th></tr></thead><tbody>{run.history.map(row => <tr key={row.step}><td>{row.step}</td><td>{row.train_loss.toFixed(5)}</td><td>{row.validation_loss.toFixed(5)}</td><td>{row.perplexity.toFixed(3)}</td></tr>)}</tbody></table></div></details>
            <a className="text-button" href="#transformer-generate">使用训练权重生成文本 <ArrowRight size={14} /></a>
          </ExperimentPanel>
        </>}
        <details className="language-reading"><summary>查看本实验语料与词表</summary><p>词表外验证字符：{run.unknown_validation} 个，映射到 UNK。实际训练窗口长度 {run.training_context}。</p><h3>训练文本</h3><pre>{Array.from(run.config.corpus).slice(0, run.train_tokens).join("")}</pre><h3>验证文本</h3><pre>{Array.from(run.config.corpus).slice(run.train_tokens).join("")}</pre><div className="language-vocab">{run.vocab.map((token, index) => <span key={index}><b>{tokenLabel(token)}</b><small>{index}</small></span>)}</div></details>
        {mode === "generate" && <>
          <ExperimentPanel title="用真实 Transformer 权重续写"><fieldset className="language-generation-controls" disabled={pending}><legend>生成设置</legend>
            <label className="language-field">生成权重<select value={checkpoint} onChange={event => changeGeneration(() => setCheckpoint(event.target.value as "latest" | "best"))}><option value="latest">当前权重 · 第 {run.step} 步</option><option value="best">最低验证损失权重 · 第 {run.best_step} 步</option></select></label>
            {(checkpoint === "best" ? run.best_step : run.step) === 0 && <p>选中的权重尚未训练。可以观察随机输出；完成生成任务需使用训练过的权重。</p>}
            <label className="language-field">提示词<input value={prompt} maxLength={128} onChange={event => changeGeneration(() => setPrompt(event.target.value))} /></label>{invalidPrompt && <p role="alert">提示词包含词表外字符，请从当前实验语料中取字。</p>}
            <Slider label="采样温度" value={temperature} min={0.1} max={2} step={0.1} onChange={value => changeGeneration(() => setTemperature(value))} />
            <Slider label="生成 token 数" value={length} min={10} max={120} step={10} onChange={value => changeGeneration(() => setLength(value))} />
            <label className="language-field">Top-k<select value={topK} onChange={event => changeGeneration(() => setTopK(Number(event.target.value)))}><option value={0}>全部候选</option><option value={5}>前 5 个候选</option><option value={10}>前 10 个候选</option></select></label>
            <label className="language-field">采样种子<input type="number" min={0} max={4294967295} value={seed} onChange={event => changeGeneration(() => setSeed(Number(event.target.value)))} /></label>
            <button className="primary-button" disabled={!prompt || invalidPrompt || !Number.isInteger(seed) || seed < 0 || seed > 4294967295} onClick={() => { onInteract(); void action(async signal => {
              const result = await transformerApi<TransformerGeneration>(`/runs/${run.id}/generate`, { prompt, temperature, length, top_k: topK, seed, checkpoint }, signal);
              if (!signal.aborted && alive.current) setGeneration(result);
            }); }}><Play size={14} />生成文本</button>
          </fieldset><p>模型每步读取最后 {run.config.context_length} 个 token，超长提示词会从左侧截断作为上下文；输出仍保留完整提示词。滑窗后位置从 0 重新编号。采样不更新权重。</p></ExperimentPanel>
          {generation && <ExperimentPanel title="Transformer 实际生成结果"><p>实验 {generation.run_id.slice(0, 8)} · {generation.checkpoint === "best" ? "最佳验证" : "当前"}权重 · 第 {generation.step} 步</p><pre className="language-output"><span>{generation.prompt}</span>{generation.continuation}</pre><h3>第一个生成位置的采样概率 · 前 8 项</h3><div className="language-distribution">{generation.distribution.map(item => <div key={item.token}><span>{tokenLabel(item.token)}</span><meter min={0} max={1} value={item.probability} aria-label={`${tokenLabel(item.token)} 采样概率`} /><strong>{(item.probability * 100).toFixed(2)}%</strong></div>)}</div><AttentionInspector key={JSON.stringify([generation.run_id, generation.step, generation.prompt, temperature, seed])} generation={generation} /></ExperimentPanel>}
        </>}
        <a className="secondary-button" href={`/api/transformer/runs/${run.id}/checkpoint`} download><Download size={14} />下载完整检查点（当前 / 最佳权重与优化器）</a>
      </>}
    </>}
    <details className="language-reading"><summary>原理与官方资料</summary><p>训练标签是输入向右错开一个位置的序列，模型对全部位置输出 logits。上三角因果掩码避免目标信息从未来泄漏；残差与 LayerNorm 帮助深层网络优化。</p><p>参考 <a href="https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html" target="_blank" rel="noreferrer">PyTorch Transformer 组成模块</a>与<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html" target="_blank" rel="noreferrer">交叉熵定义</a>。本实验实现小型字符级模型，不包含预训练语料、指令微调或对话模板。</p></details>
  </div>;
}

export function TransformerArchitecture(props: { onInteract: () => void }) { return <TransformerExperiment mode="architecture" {...props} />; }
export function TransformerTraining(props: { onInteract: () => void }) { return <TransformerExperiment mode="train" {...props} />; }
export function TransformerGenerationLesson(props: { onInteract: () => void }) { return <TransformerExperiment mode="generate" {...props} />; }
