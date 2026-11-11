import { useEffect, useRef, useState } from "react";
import { ArrowRight, Download, Play, RotateCcw } from "lucide-react";
import { calculate } from "../../lib/api";
import { useLessonTask } from "../../lib/learningProgress";
import { ExperimentPanel, Slider } from "../../components/ui";
import "./language.css";
import { LossChart } from "./LossChart";

type Metric = { step: number; train_loss: number; validation_loss: number; perplexity: number };
type Model = {
  corpus: string; vocab: string[]; weights: number[][]; step: number; history: Metric[];
  train_text: string; validation_text: string; train_tokens: number; validation_tokens: number;
  unknown_validation: number; pairs: { input: number; target: number }[];
};
type Generation = { continuation: string; distribution: { token: string; probability: number }[] };
// Keep a single model while moving through lessons. Refresh starts a new experiment.
let sessionModel: Model | null = null;
const displayToken = (token: string) => token === "\n" ? "↵" : token === " " ? "空格" : token;


function LanguageExperiment({ mode, onInteract }: { mode: "data" | "train" | "generate"; onInteract: () => void }) {
  const [model, setModel] = useState<Model | null>(sessionModel);
  const [draft, setDraft] = useState(sessionModel?.corpus ?? "");
  const [selected, setSelected] = useState<number | null>(null);
  const [learningRate, setLearningRate] = useState(5);
  const [temperature, setTemperature] = useState(1);
  const [prompt, setPrompt] = useState("小猫");
  const [seed, setSeed] = useState(7);
  const [length, setLength] = useState(60);
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const controller = useRef<AbortController | null>(null);
  const last = model?.history.at(-1);
  useLessonTask("language-data", mode === "data" && selected !== null && Boolean(model));
  useLessonTask("language-train", mode === "train" && Boolean(model && model.step >= 3 && last!.train_loss < model.history[0].train_loss));
  useLessonTask("language-generate", mode === "generate" && Boolean(generation && model && model.step > 0));

  async function request<T>(path: string, payload: unknown, commit: (result: T) => void) {
    if (controller.current) return;
    const current = new AbortController(); controller.current = current;
    setPending(true); setError(""); setNotice("");
    try {
      const result = await calculate<T>(`/language/${path}`, payload, current.signal);
      if (!current.signal.aborted) commit(result);
    } catch (reason) {
      if (!current.signal.aborted) setError(reason instanceof Error ? reason.message : "请求失败，请检查计算服务。");
    } finally {
      if (controller.current === current) { controller.current = null; setPending(false); }
    }
  }
  function save(next: Model) { sessionModel = next; setModel(next); }
  function prepare(useDraft: boolean) {
    void request<Model>("prepare", useDraft ? { corpus: draft } : {}, result => {
      save(result); setDraft(result.corpus); setSelected(null); setGeneration(null);
      setNotice("语料已准备，模型已归零。点击样本查看输入与标签。");
    });
  }
  useEffect(() => {
    if (!sessionModel) prepare(false);
    return () => { controller.current?.abort(); controller.current = null; };
    // Preparation runs once per lesson mount, using the shared in-memory model.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  function train(steps: number) {
    if (!model) return;
    onInteract();
    void request<Pick<Model, "weights" | "step" | "history">>("train", { corpus: model.corpus, weights: model.weights, step: model.step, steps, learning_rate: learningRate }, result => {
      save({ ...model, ...result, history: [...model.history, ...result.history] }); setGeneration(null);
      setNotice(`已完成 ${result.step} 次真实 SGD 更新。`);
    });
  }
  const invalidPrompt = Boolean(model && Array.from(prompt).some(char => !model.vocab.includes(char)));
  const changes = model && draft !== model.corpus;

  return <div className="language-lab" aria-busy={pending}>
    <div className="language-intro"><span className="small-label">字符级 BIGRAM · CPU / NUMPY 真实计算</span><p>模型把当前字符映射为下一字符的概率，只有一张 V × V 参数表，上下文长度为 1。它适合学习训练流程；连贯写作需要能利用更长上下文的模型，例如 Transformer。</p><p>实验权重在本次页面会话内共享，刷新后重新开始；可下载 JSON 保存数值，当前不支持导入。学习完成记录另行保留。</p></div>
    {error && <div className="connection-notice" role="alert">{error} <button className="text-button" disabled={pending} onClick={() => prepare(Boolean(draft))}>重新准备语料</button></div>}
    <p className="language-status" role="status">{pending ? "正在计算，请稍候…" : notice}</p>
    {mode === "data" && <ExperimentPanel title="01 / 准备你的训练文本">
      <label className="language-field">教学语料<textarea value={draft} maxLength={2000} rows={7} disabled={pending} onChange={event => setDraft(event.target.value)} /></label>
      <p>40–2,000 个 Unicode 字符；训练部分含 2–95 种字符。按顺序先分为 80% 训练、20% 验证，再分别生成相邻字符对，边界不跨集合。</p>
      <div className="language-actions"><button className="primary-button" disabled={pending || Array.from(draft).length < 40} onClick={() => { onInteract(); prepare(true); }}><RotateCcw size={15} />准备语料并重置模型</button><button className="secondary-button" disabled={pending} onClick={() => prepare(false)}>恢复教学语料</button></div>
      {changes && <p role="status">文本已编辑，点击「准备语料」后生效，并清除已有权重。</p>}
    </ExperimentPanel>}
    {model && <>
      <div className="language-metrics"><div><span>词表（含 UNK）</span><strong>{model.vocab.length}</strong></div><div><span>训练 / 验证字符</span><strong>{model.train_tokens} / {model.validation_tokens}</strong></div><div><span>可训练参数</span><strong>{model.vocab.length ** 2}</strong></div><div><span>已更新</span><strong>{model.step} 步</strong></div></div>
      {mode === "data" && <>
        <ExperimentPanel title="02 / 字符 → Token ID"><div className="language-vocab">{model.vocab.map((token, id) => <span key={id}><b>{displayToken(token)}</b><small>{id}</small></span>)}</div><p>Token ID 只是索引，没有大小或语义远近之分。验证中 {model.unknown_validation} 个词表外字符映射到 ID 0。</p></ExperimentPanel>
        <ExperimentPanel title="03 / 标签就在下一位"><div className="language-pairs">{model.pairs.map((pair, index) => <button className="secondary-button" aria-pressed={selected === index} key={index} onClick={() => { setSelected(index); onInteract(); }}><span>{displayToken(model.vocab[pair.input])}<small> {pair.input}</small></span><ArrowRight size={14} /><span>{displayToken(model.vocab[pair.target])}<small> {pair.target}</small></span></button>)}</div>{selected !== null && <p role="status">第 {selected + 1} 对：输入「{displayToken(model.vocab[model.pairs[selected].input])}」，目标「{displayToken(model.vocab[model.pairs[selected].target])}」。标签来自原文，无需人工标注。</p>}<details><summary>查看实际划分的原文</summary><h3>训练文本</h3><pre>{model.train_text}</pre><h3>验证文本（不更新参数）</h3><pre>{model.validation_text}</pre></details><a className="text-button" href="#language-train">开始训练 <ArrowRight size={14} /></a></ExperimentPanel>
      </>}
      {mode === "train" && <>
        <ExperimentPanel title="真实训练控制台"><p>每一步使用全部 {model.train_tokens - 1} 对训练样本计算平均交叉熵与精确梯度。零初始化时概率均匀，初始损失为 ln(V) = {model.history[0].train_loss.toFixed(4)}。最多 5,000 步。</p><Slider label="学习率" value={learningRate} min={0.5} max={20} step={0.5} onChange={setLearningRate} /><div className="language-actions">{[1, 100, 200].map(steps => <button className={steps === 100 ? "primary-button" : "secondary-button"} key={steps} disabled={pending || model.step + steps > 5000} onClick={() => train(steps)}><Play size={14} />训练 {steps} 步</button>)}<button className="secondary-button" disabled={pending} onClick={() => { setDraft(model.corpus); void request<Model>("prepare", { corpus: model.corpus }, save); }}>重置权重</button></div></ExperimentPanel>
        <ExperimentPanel title="训练与验证：看见模型在学习"><div className="language-metrics"><div><span>训练交叉熵</span><strong>{last!.train_loss.toFixed(4)}</strong></div><div><span>验证交叉熵</span><strong>{last!.validation_loss.toFixed(4)}</strong></div><div><span>验证困惑度 exp(L)</span><strong>{last!.perplexity.toFixed(2)}</strong></div></div><LossChart history={model.history} /><p>损失使用自然对数，越低表示模型越倾向于真实的下一字符。验证用于观察未参与更新的文本，本教学没有额外独立测试集，也不宣称语言理解能力。</p><details><summary>最近 10 次更新的精确指标</summary><div className="language-table"><table><thead><tr><th>步数</th><th>训练损失</th><th>验证损失</th><th>验证困惑度</th></tr></thead><tbody>{model.history.slice(-10).map(row => <tr key={row.step}><td>{row.step}</td><td>{row.train_loss.toFixed(5)}</td><td>{row.validation_loss.toFixed(5)}</td><td>{row.perplexity.toFixed(3)}</td></tr>)}</tbody></table></div></details><a className="text-button" href="#language-generate">用当前权重生成文本 <ArrowRight size={14} /></a></ExperimentPanel>
      </>}
      {mode === "generate" && <>
        <ExperimentPanel title="逐字符采样"><fieldset className="language-generation-controls" disabled={pending}><legend className="sr-only">采样设置</legend><p>固定权重、提示词、温度和随机种子，可以复现相同输出。本模型只使用提示词最后一个字符。</p>{model.step === 0 && <p>当前是未训练模型，输出接近均匀随机。请先<a href="#language-train">完成训练</a>再记录生成任务。</p>}<label className="language-field">提示词（从训练文本取字）<input value={prompt} maxLength={80} onChange={event => { setPrompt(event.target.value); setGeneration(null); }} /></label>{invalidPrompt && <p role="alert">提示词含词表外字符，请从上节训练原文中选择。</p>}<Slider label="采样温度" value={temperature} min={0.1} max={2} step={0.1} onChange={value => { setTemperature(value); setGeneration(null); }} /><Slider label="生成 token 数" value={length} min={10} max={120} step={10} onChange={value => { setLength(value); setGeneration(null); }} /><label className="language-field">随机种子<input type="number" min={0} max={4294967295} value={seed} onChange={event => { setSeed(Number(event.target.value)); setGeneration(null); }} /></label><button className="primary-button" disabled={pending || !prompt || invalidPrompt || !Number.isInteger(seed) || seed < 0 || seed > 4294967295} onClick={() => { onInteract(); void request<Generation>("generate", { corpus: model.corpus, weights: model.weights, prompt, temperature, length, seed }, setGeneration); }}><Play size={15} />生成文本</button></fieldset>
        </ExperimentPanel>
        {generation && <ExperimentPanel title="模型实际输出"><pre className="language-output"><span>{prompt}</span>{generation.continuation}</pre><h3>第一个生成位置的候选概率（前 8 项）</h3><div className="language-distribution">{generation.distribution.map(item => <div key={item.token}><span>{displayToken(item.token)}</span><meter min={0} max={1} value={item.probability} aria-label={`${displayToken(item.token)} 概率`} /><strong>{(item.probability * 100).toFixed(2)}%</strong></div>)}</div><p>温度改变采样分布，不更新权重。较低温度更偏向高分字符，较高温度更分散；随机样本不保证通顺。</p></ExperimentPanel>}
      </>}
      {mode !== "data" && <button className="secondary-button" disabled={pending} onClick={() => {
        const url = URL.createObjectURL(new Blob([JSON.stringify({ version: 1, kind: "character-bigram", ...model }, null, 2)], { type: "application/json" }));
        const link = document.createElement("a"); link.href = url; link.download = "zhiwei-bigram.json"; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}><Download size={15} />下载权重、词表与训练记录</button>}
    </>}
    <details className="language-reading"><summary>从 Bigram 走向 Transformer</summary><p>Bigram 的 P(下一字符 | 当前字符) 只记住一阶转移。Transformer 则用嵌入、位置信息、因果注意力与前馈网络综合多个历史 token；训练时屏蔽未来位置，避免把答案泄漏给输入。</p><p>两者都可学习预测下一 token，但本实验没有注意力、多层网络、指令微调或对话能力。可继续阅读 <a href="https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html" target="_blank" rel="noreferrer">PyTorch 的 N-gram 语言建模教程</a>和<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html" target="_blank" rel="noreferrer">交叉熵文档</a>。</p></details>
  </div>;
}

export function LanguageData(props: { onInteract: () => void }) { return <LanguageExperiment mode="data" {...props} />; }
export function LanguageTraining(props: { onInteract: () => void }) { return <LanguageExperiment mode="train" {...props} />; }
export function LanguageGeneration(props: { onInteract: () => void }) { return <LanguageExperiment mode="generate" {...props} />; }
