import { MathFormula } from "../../components/MathFormula";
import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent, PointerEvent } from "react";
import { ArrowDown, ArrowUp, ArrowDownRight, Box, Braces, CheckCircle2, ChevronDown, Code2, Copy, Cpu, Download, Eye, GripVertical, Layers3, Maximize2, Plus, RotateCcw, Trash2, TriangleAlert, Waves } from "lucide-react";
import { analyzeNetwork, createExample, createLayer, MAX_NETWORK_LAYERS, NETWORK_INPUT, NETWORK_STORAGE_KEY, readSavedNetwork, toPytorchCode } from "../../lib/networkBuilder";
import type { LayerType, NetworkLayer } from "../../lib/networkBuilder";
import "./network-builder.css";
import ArchitectureScene from "./ArchitectureScene";
import CustomTrainingPanel from "./CustomTrainingPanel";

const palette = [
  { type: "conv", title: "Conv2D", name: "提取局部特征", icon: Box, description: "卷积核扫描空间，生成新的特征通道。" },
  { type: "relu", title: "ReLU", name: "引入非线性", icon: Waves, description: "保留正值，将负值置零；形状不变。" },
  { type: "pool", title: "MaxPool2D", name: "压缩空间尺寸", icon: Maximize2, description: "2×2 窗口、步幅 2，保留区域最大值。" },
  { type: "flatten", title: "Flatten", name: "展平特征", icon: Layers3, description: "保留批次维度，将其余维度展平成向量。" },
  { type: "linear", title: "Linear", name: "连接分类输出", icon: Cpu, description: "全连接映射；MNIST 最后需要 10 个输出。" },
] as const;
type DragPayload = { source: "palette"; type: LayerType } | { source: "stack"; id: string };
type PointerDrag = { payload: DragPayload; pointerId: number; startX: number; startY: number; active: boolean; element: HTMLElement; targetIndex: number | null };
const shapeText = (shape: number[] | null) => shape ? shape.join(" × ") : "等待修复";
const numberText = (value: number) => value.toLocaleString("zh-CN");

function downloadText(name: string, text: string, type: string) {
  const href = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = href;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export default function NetworkBuilder({ onInteract, onOpenLesson }: { onInteract?: () => void; onOpenLesson?: (id: "mnist-train" | "mnist-evaluation" | "handwriting") => void }) {
  const [layers, setLayers] = useState<NetworkLayer[]>(() => readSavedNetwork() ?? createExample("cnn"));
  const [dragging, setDragging] = useState(false);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [message, setMessage] = useState("");
  const [storageWarning, setStorageWarning] = useState(false);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const pointerDrag = useRef<PointerDrag | null>(null);
  const suppressClick = useRef(false);
  const stack = useRef<HTMLDivElement>(null);
  const trainingSection = useRef<HTMLDivElement>(null);
  const analysis = useMemo(() => analyzeNetwork(layers), [layers]);
  const code = useMemo(() => analysis.valid ? toPytorchCode(layers) : "", [layers, analysis.valid]);

  useEffect(() => {
    try {
      localStorage.setItem(NETWORK_STORAGE_KEY, JSON.stringify({ version: 1, layers }));
      setStorageWarning(false);
      window.dispatchEvent(new Event("zhiwei:network-change"));
    } catch { setStorageWarning(true); }
  }, [layers]);

  function commit(next: NetworkLayer[], description = "结构已更新，尺寸与参数已重新计算。") {
    setLayers(next);
    setMessage(description);
    onInteract?.();
  }
  function addLayer(type: LayerType, index = layers.length) {
    if (layers.length >= MAX_NETWORK_LAYERS) { setMessage("已达到 16 层上限，请先删除一层。"); return; }
    const next = [...layers];
    next.splice(index, 0, createLayer(type));
    commit(next, `已添加 ${palette.find(item => item.type === type)?.title} 层。`);
  }
  function moveLayer(index: number, target: number) {
    if (target < 0 || target >= layers.length) return;
    const next = [...layers];
    const [layer] = next.splice(index, 1);
    next.splice(target, 0, layer);
    commit(next, `已将第 ${index + 1} 层移动到第 ${target + 1} 层。`);
  }
  function patchLayer(id: string, field: string, value: number) {
    const next = layers.map(layer => {
      if (layer.id !== id) return layer;
      if (layer.type === "conv" && ["outChannels", "kernel", "stride", "padding"].includes(field)) return { ...layer, [field]: value };
      if (layer.type === "linear" && field === "units") return { ...layer, units: value };
      return layer;
    });
    commit(next);
  }
  function targetAt(clientX: number, clientY: number): number | null {
    const hit = document.elementFromPoint(clientX, clientY)?.closest<HTMLElement>("[data-builder-drop-index]");
    if (!hit || !stack.current?.contains(hit)) return null;
    const index = Number(hit.dataset.builderDropIndex);
    return Number.isInteger(index) && index >= 0 && index <= layers.length ? index : null;
  }
  function startPointer(event: PointerEvent<HTMLElement>, payload: DragPayload) {
    if (!event.isPrimary || event.button !== 0) return;
    suppressClick.current = false;
    pointerDrag.current = { payload, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, active: false, element: event.currentTarget, targetIndex: null };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function movePointer(event: PointerEvent<HTMLElement>) {
    const drag = pointerDrag.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (!drag.active && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) <= 8) return;
    if (!drag.active) { drag.active = true; setDragging(true); }
    event.preventDefault();
    const target = targetAt(event.clientX, event.clientY);
    if (target !== drag.targetIndex) { drag.targetIndex = target; setDropIndex(target); }
  }
  function clearPointer() {
    const drag = pointerDrag.current;
    pointerDrag.current = null;
    setDragging(false);
    setDropIndex(null);
    if (drag?.element.hasPointerCapture(drag.pointerId)) drag.element.releasePointerCapture(drag.pointerId);
  }
  function finishPointer(event: PointerEvent<HTMLElement>) {
    const drag = pointerDrag.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const index = drag.active ? targetAt(event.clientX, event.clientY) : null;
    if (drag.active) { event.preventDefault(); suppressClick.current = true; }
    clearPointer();
    if (index === null) return;
    if (drag.payload.source === "palette") addLayer(drag.payload.type, index);
    else {
      const id = drag.payload.id;
      const from = layers.findIndex(layer => layer.id === id);
      if (from >= 0) moveLayer(from, Math.min(layers.length - 1, index > from ? index - 1 : index));
    }
  }
  function cancelPointer(event: PointerEvent<HTMLElement>) {
    if (pointerDrag.current?.pointerId === event.pointerId) {
      if (pointerDrag.current.active) suppressClick.current = true;
      clearPointer();
    }
  }
  function clickPalette(event: MouseEvent<HTMLButtonElement>, type: LayerType) {
    if (suppressClick.current && event.detail !== 0) { event.preventDefault(); suppressClick.current = false; return; }
    addLayer(type);
  }
  function exportJson() {
    downloadText("zhiwei-network-design.json", JSON.stringify({ format: "zhiwei-network-design-v1", input: [...NETWORK_INPUT], batchDimensionExcluded: true, layers, analysis, trained: false }, null, 2), "application/json");
    setMessage("结构 JSON 已导出；文件包含结构与尺寸，不包含训练权重。");
  }

  return (
    <section className={`builder-lab ${dragging ? "builder-is-dragging" : ""}`} aria-label="神经网络结构搭建器">
      <ArchitectureScene layers={layers} selectedLayerId={selectedLayerId} onSelectLayer={setSelectedLayerId} />
      <div className="builder-training-shortcut"><div><span>01 设计结构</span><ArrowDownRight size={15} /><span>02 真实训练</span><ArrowDownRight size={15} /><span>03 评估与识别</span></div><button type="button" className="primary-button" disabled={!analysis.valid} onClick={() => trainingSection.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches || trainingSection.current?.closest(".effects-off") ? "instant" : "smooth", block: "start" })}>前往训练控制台<ArrowDown size={15} /></button></div>
      <div className="builder-layout">
        <aside className="builder-palette" aria-label="网络层库">
          <div className="builder-section-heading"><h3>层库</h3><span>拖入 / 点击添加</span></div>
          {palette.map(item => {
            const Icon = item.icon;
            return <button type="button" key={item.type} className={`builder-palette-item builder-type-${item.type}`} onPointerDown={event => startPointer(event, { source: "palette", type: item.type })} onPointerMove={movePointer} onPointerUp={finishPointer} onPointerCancel={cancelPointer} onLostPointerCapture={cancelPointer} onClick={event => clickPalette(event, item.type)} disabled={layers.length >= MAX_NETWORK_LAYERS} aria-label={`添加 ${item.title} 层`}>
              <span className="builder-palette-title"><Icon size={17} /><strong>{item.title}</strong><Plus size={14} /></span><span className="builder-palette-name">{item.name}</span><span className="builder-palette-description">{item.description}</span>
            </button>;
          })}
          <p className="builder-palette-tip"><GripVertical size={15} />电脑可拖拽排序；手机和键盘可用层卡片上的上下按钮。</p>
          <div className="builder-formula-note"><span>CONV / POOL SIZE</span><MathFormula tex={String.raw`\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1`} block /><p>n：输入尺寸 · p：填充<br />k：窗口大小 · s：步幅</p></div>
        </aside>
        <div className="builder-workspace">
          <div className="builder-workspace-bar"><div><span className="builder-section-label">YOUR NETWORK</span><strong>{layers.length} / {MAX_NETWORK_LAYERS} 层</strong></div><div className="builder-presets"><button type="button" onClick={() => commit(createExample("cnn"), "已载入 CNN 示例：卷积 → 激活 → 池化 → 展平 → 分类。")}>CNN 示例</button><button type="button" onClick={() => commit(createExample("mlp"), "已载入 MLP 示例：展平 → 全连接 → 激活 → 分类。")}>MLP 示例</button><button type="button" onClick={() => commit([], "网络已清空，可以从层库重新搭建。")}><RotateCcw size={13} />清空</button></div></div>
          <div className="builder-metrics" aria-live="polite"><div><span>输入形状 C × H × W</span><strong>1 × 28 × 28</strong><small>所有尺寸均不包含 batch</small></div><div><span>总可训练参数</span><strong>{numberText(analysis.totalParameters)}</strong><small>{analysis.shapeValid ? "包含权重与偏置" : "仅统计首个错误之前的有效层"}</small></div><div><span>当前输出</span><strong>{shapeText(analysis.output)}</strong><small>{analysis.classifierReady ? "10 类 logits · 当前结构定义" : "目标：10 个分类输出"}</small></div></div>
          <div ref={stack} className={`builder-stack ${dragging ? "is-dragging" : ""}`} aria-label="已搭建的网络层" data-builder-drop-index={layers.length}>
            <div className="builder-input-node" data-builder-drop-index={0}><span className="builder-input-icon"><Box size={20} /></span><div><strong>输入图像</strong><span>MNIST · 灰度 · [batch, 1, 28, 28]</span></div><span className="builder-input-shape">1 × 28 × 28</span></div>
            {layers.map((layer, index) => {
              const meta = palette.find(item => item.type === layer.type)!;
              const Icon = meta.icon;
              const row = analysis.rows[index];
              return <div className="builder-layer-slot" key={layer.id}>
                <div className={`builder-drop-zone ${dropIndex === index ? "is-over" : ""}`} data-builder-drop-index={index}><ArrowDown size={14} /><span>放到这里</span></div>
                <article className={`builder-layer-card builder-type-${layer.type} ${row?.error ? "has-error" : ""} ${dropIndex === index ? "is-drop-target" : ""} ${selectedLayerId === layer.id ? "is-scene-selected" : ""}`} data-builder-drop-index={index} aria-label={`第 ${index + 1} 层 ${meta.title}`}>
                  <div className="builder-layer-header"><span className="builder-drag-handle" title="拖动排序；也可使用右侧上下按钮" onPointerDown={event => startPointer(event, { source: "stack", id: layer.id })} onPointerMove={movePointer} onPointerUp={finishPointer} onPointerCancel={cancelPointer} onLostPointerCapture={cancelPointer}><GripVertical size={16} aria-hidden="true" /></span><span className="builder-layer-icon"><Icon size={18} /></span><div className="builder-layer-name"><strong>{meta.title}</strong><span>{meta.name}</span></div><span className="builder-layer-index">{String(index + 1).padStart(2, "0")}</span><div className="builder-layer-actions"><button type="button" aria-label={`查看第 ${index + 1} 层结构`} aria-pressed={selectedLayerId === layer.id} onClick={() => setSelectedLayerId(layer.id)}><Eye size={14} /></button><button type="button" aria-label={`上移第 ${index + 1} 层`} disabled={index === 0} onClick={() => moveLayer(index, index - 1)}><ArrowUp size={14} /></button><button type="button" aria-label={`下移第 ${index + 1} 层`} disabled={index === layers.length - 1} onClick={() => moveLayer(index, index + 1)}><ArrowDown size={14} /></button><button type="button" className="builder-delete" aria-label={`删除第 ${index + 1} 层`} onClick={() => commit(layers.filter(item => item.id !== layer.id), `已删除第 ${index + 1} 层 ${meta.title}。`)}><Trash2 size={14} /></button></div></div>
                  {layer.type === "conv" && <div className="builder-layer-config"><label>输出通道<select aria-label={`第 ${index + 1} 层输出通道`} value={layer.outChannels} onChange={event => patchLayer(layer.id, "outChannels", Number(event.target.value))}>{[8, 16, 32, 64].map(value => <option key={value} value={value}>{value}</option>)}</select></label><label>卷积核<select aria-label={`第 ${index + 1} 层卷积核`} value={layer.kernel} onChange={event => patchLayer(layer.id, "kernel", Number(event.target.value))}>{[3, 5].map(value => <option key={value} value={value}>{value} × {value}</option>)}</select></label><label>步幅<select aria-label={`第 ${index + 1} 层步幅`} value={layer.stride} onChange={event => patchLayer(layer.id, "stride", Number(event.target.value))}>{[1, 2].map(value => <option key={value} value={value}>{value}</option>)}</select></label><label>填充<select aria-label={`第 ${index + 1} 层填充`} value={layer.padding} onChange={event => patchLayer(layer.id, "padding", Number(event.target.value))}>{[0, 1, 2].map(value => <option key={value} value={value}>{value}</option>)}</select></label></div>}
                  {layer.type === "linear" && <div className="builder-layer-config builder-linear-config"><label>输出单元<select aria-label={`第 ${index + 1} 层输出单元`} value={layer.units} onChange={event => patchLayer(layer.id, "units", Number(event.target.value))}>{[10, 32, 64, 128].map(value => <option key={value} value={value}>{value}{value === 10 ? " · 十类分类" : ""}</option>)}</select></label><span>参数 = (输入特征数 + 1) × 输出单元</span></div>}
                  <div className="builder-layer-shapes"><span><small>IN</small>{shapeText(row?.input ?? null)}</span><ArrowDownRight size={15} /><span><small>OUT</small>{shapeText(row?.output ?? null)}</span><span className="builder-layer-params">{row?.error ? "—" : numberText(row?.parameters ?? 0)}<small>参数</small></span></div>
                  {row?.error && <p className="builder-layer-error"><TriangleAlert size={14} />{row.error}</p>}
                </article>
              </div>;
            })}
            <div className={`builder-drop-zone builder-final-drop ${dropIndex === layers.length ? "is-over" : ""}`} data-builder-drop-index={layers.length}>{layers.length ? <><ArrowDown size={14} /><span>拖入下一层</span></> : <><Plus size={24} /><strong>从层库添加第一层</strong><span>也可以选择上方 CNN / MLP 示例。</span></>}</div>
            <div className={`builder-validation ${analysis.valid ? "is-valid" : "is-invalid"}`} role="status">{analysis.valid ? <CheckCircle2 size={20} /> : <TriangleAlert size={20} />}<div><strong>{analysis.valid ? "结构有效，已连接十类输出。" : "继续调整，连接完整的分类网络。"}</strong>{analysis.valid ? <p>尺寸与参数检查已通过。可在下方启动真实训练，完成独立测试后使用最佳权重识别手写数字。</p> : analysis.errors.map(error => <p key={error}>{error}</p>)}</div></div>
          </div>
          <div className="builder-export-bar"><button type="button" className="secondary-button" onClick={exportJson}><Download size={15} />导出结构 JSON</button><button type="button" className="primary-button" disabled={!analysis.valid} aria-expanded={showCode && analysis.valid} aria-controls="builder-pytorch-code" onClick={() => setShowCode(!showCode)}><Code2 size={15} />{showCode && analysis.valid ? "收起 PyTorch 代码" : "查看 PyTorch 代码"}<ChevronDown size={14} /></button></div>
          {showCode && analysis.valid && <section className="builder-code" id="builder-pytorch-code" aria-label="生成的 PyTorch 网络结构代码"><div><span><Braces size={16} />MODEL DEFINITION / 未训练结构</span><div><button type="button" onClick={async () => { try { await navigator.clipboard.writeText(code); setMessage("PyTorch 代码已复制。"); } catch { setMessage("当前浏览器无法复制，请手动选择下方代码。"); } }}><Copy size={14} />复制</button><button type="button" onClick={() => { downloadText("zhiwei-network.py", code, "text/x-python;charset=utf-8"); setMessage("PyTorch 结构代码已导出。"); }}><Download size={14} />下载 .py</button></div></div><pre tabIndex={0}><code>{code}</code></pre></section>}
          <div className="builder-save-status"><span className={storageWarning ? "is-warning" : ""}>{storageWarning ? "当前浏览器无法保存结构；仍可继续设计并导出文件。" : "结构自动保存在当前浏览器，可在挑战关卡中检查。"}</span><span role="status">{message}</span></div>
        </div>
      </div>
      <div ref={trainingSection} className="builder-training-anchor"><CustomTrainingPanel layers={layers} onOpenLesson={onOpenLesson} /></div>
    </section>
  );
}
