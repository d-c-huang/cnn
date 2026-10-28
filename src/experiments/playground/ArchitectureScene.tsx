import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import { ArrowLeft, ArrowRight, Box, CircuitBoard, Layers3, Pause, Play, TriangleAlert } from "lucide-react";
import { analyzeNetwork, NETWORK_INPUT } from "../../lib/networkBuilder";
import type { LayerType, NetworkLayer } from "../../lib/networkBuilder";
import "./architecture-scene.css";

type Props = {
  layers: NetworkLayer[];
  selectedLayerId?: string | null;
  onSelectLayer?: (id: string) => void;
};
type Stage = { id: string; type: LayerType | "input"; title: string; shape: number[] | null; parameters: number; error: string | null; index: number };
const accents: Record<Stage["type"], string> = { input: "#a9ddc3", conv: "#62dfce", relu: "#c4e98a", pool: "#79bede", flatten: "#e2bc7b", linear: "#a9d78a" };
const names: Record<LayerType, string> = { conv: "Conv2D", relu: "ReLU", pool: "MaxPool2D", flatten: "Flatten", linear: "Linear" };
const shapeText = (shape: number[] | null) => shape ? shape.join(" × ") : "无法推导";
const countText = (value: number) => value.toLocaleString("zh-CN");

function explanation(layer: NetworkLayer) {
  if (layer.type === "conv") return { title: "在局部窗口里提取特征", formula: `K ${layer.kernel}×${layer.kernel} · S ${layer.stride} · P ${layer.padding} · ${layer.outChannels} channels`, note: "每个输出通道对应一组可学习的卷积核；权重与偏置均计入参数量。" };
  if (layer.type === "pool") return { title: "保留响应，缩小空间", formula: "MaxPool 2×2 · stride 2", note: "每个窗口保留最大值，通道数保持不变；池化没有可训练参数。" };
  if (layer.type === "relu") return { title: "给网络引入非线性", formula: "f(x) = max(0, x)", note: "逐元素把负值置零，保留正值；特征形状保持不变，也不增加参数。" };
  if (layer.type === "flatten") return { title: "让空间特征连接到向量", formula: "C × H × W → features", note: "展平保留所有元素，只改变排列形状；批次维度不参与展平。" };
  if (layer.type === "linear") return { title: "把特征连接到新的输出空间", formula: `out = ${layer.units} · parameters = (in + 1) × out`, note: layer.units === 10 ? "十个输出对应数字 0–9 的 logits；需要训练后才能用于识别。" : "每个输出单元连接全部输入特征。图中的节点为抽象表示，实际单元数以尺寸标注为准。" };
  return { title: "检查当前网络层", formula: "", note: "请从层库选择支持的网络层。" };
}

function SpatialVolume({ stage, selected }: { stage: Stage; selected: boolean }) {
  const shape = stage.shape ?? [1, 16, 16];
  const size = Math.max(57, Math.min(110, 47 + Math.sqrt(shape[1] ?? 16) * 12));
  const channels = shape[0] ?? 1;
  const planes = Math.min(5, channels);
  const color = stage.error ? "#e79279" : accents[stage.type];
  return <g className="architecture-volume" transform="translate(0 4) skewY(-11)">
    {Array.from({ length: planes }, (_, index) => {
      const depth = planes - index - 1;
      return <rect key={index} x={-size / 2 + depth * 6} y={-size / 2 - depth * 6} width={size} height={size} rx="2" fill={index === planes - 1 ? "#102c25" : "#102622"} fillOpacity={index === planes - 1 ? 0.97 : .72} stroke={color} strokeWidth={index === planes - 1 && selected ? 1.6 : .85} strokeOpacity={.3 + index / planes * .5} />;
    })}
    <g stroke={color} strokeOpacity={selected ? ".3" : ".16"} strokeWidth=".6">
      {[1, 2, 3, 4, 5].map(i => <g key={i}><path d={`M${-size / 2 + size * i / 6} ${-size / 2}v${size}`} /><path d={`M${-size / 2} ${-size / 2 + size * i / 6}h${size}`} /></g>)}
    </g>
    <path d={`M${-size / 2} ${-size / 2 + 15}v-15h15 M${size / 2 - 15} ${size / 2}h15v-15`} stroke={color} strokeWidth="1.8" opacity={selected ? 1 : .6} />
    {stage.type === "relu" && <path d="M-19 6h19V-17 M0 6L21-15" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity=".85" />}
    {stage.type === "pool" && <g stroke={color} strokeWidth="1.5" fill="none"><rect x="-16" y="-16" width="32" height="32" rx="1" /><path d="M0-16V16M-16 0H16" /></g>}
  </g>;
}

function VectorVolume({ stage, selected }: { stage: Stage; selected: boolean }) {
  const color = stage.error ? "#e79279" : accents[stage.type];
  if (stage.type === "flatten") return <g transform="translate(0 -5)">
    <path d="M-28-66L-13-73L34-64L19-57Z M19-57L34-64V73L19 80Z" fill={color} fillOpacity=".07" stroke={color} strokeOpacity=".35" />
    <rect x="-28" y="-66" width="47" height="146" rx="3" fill="#192c23" stroke={color} strokeWidth={selected ? 1.6 : .9} strokeOpacity=".85" />
    {Array.from({ length: 12 }, (_, i) => <rect key={i} x="-20" y={-56 + i * 10} width={i % 3 === 0 ? 30 : 23} height="4" rx="1" fill={color} fillOpacity={i % 3 === 0 ? .72 : .3} />)}
  </g>;
  const units = stage.shape?.[0] ?? 0;
  const visible = units === 10 ? 10 : 8;
  return <g>
    <path d="M-30-77L35-66V79L-30 68Z" stroke={color} strokeOpacity=".18" fill={color} fillOpacity=".025" />
    {Array.from({ length: visible }, (_, i) => {
      const y = -63 + i * 126 / (visible - 1);
      return <g key={i}><path d={`M-43 ${y * .67}L-10 ${y}L30 ${y * .78}`} stroke={color} strokeWidth=".65" opacity=".25" /><circle cx="-10" cy={y} r={selected ? 5.8 : 5} fill="#102b20" stroke={color} strokeWidth="1.2" /><circle cx="-10" cy={y} r="1.7" fill={color} opacity=".85" />{units === 10 && <text x="4" y={y + 3.5} className="architecture-digit-label" fill={color}>{i}</text>}</g>;
    })}
  </g>;
}

export default function ArchitectureScene({ layers, selectedLayerId, onSelectLayer }: Props) {
  const id = useId().replace(/:/g, "");
  const [localSelection, setLocalSelection] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [viewportWidth, setViewportWidth] = useState(960);
  const viewport = useRef<HTMLDivElement>(null);
  const analysis = useMemo(() => analyzeNetwork(layers), [layers]);
  const activeId = layers.some(layer => layer.id === selectedLayerId) ? selectedLayerId : layers.some(layer => layer.id === localSelection) ? localSelection : layers[0]?.id;
  const activeIndex = layers.findIndex(layer => layer.id === activeId);
  const activeLayer = layers[activeIndex];
  const activeAnalysis = analysis.rows[activeIndex];
  const detail = activeLayer ? explanation(activeLayer) : null;
  const stages: Stage[] = [{ id: "architecture-input", type: "input", title: "INPUT", shape: [...NETWORK_INPUT], parameters: 0, error: null, index: -1 }, ...layers.map((layer, index) => ({ id: layer.id, type: layer.type, title: names[layer.type], shape: analysis.rows[index]?.output ?? null, parameters: analysis.rows[index]?.parameters ?? 0, error: analysis.rows[index] ? analysis.rows[index].error : "结构配置无效，无法推导此层。", index }))];
  const canvasWidth = Math.max(viewportWidth, stages.length * 155 + 45);
  const spacing = stages.length > 1 ? (canvasWidth - 155) / (stages.length - 1) : 0;
  const position = (index: number) => 72 + index * spacing;

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(entries => setViewportWidth(Math.max(280, entries[0].contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  function selectLayer(layerId: string) {
    setLocalSelection(layerId);
    onSelectLayer?.(layerId);
    const index = layers.findIndex(layer => layer.id === layerId);
    const element = viewport.current;
    if (index >= 0 && element) {
      const x = position(index + 1);
      if (x < element.scrollLeft + 100 || x > element.scrollLeft + element.clientWidth - 100) {
        element.scrollTo({ left: Math.max(0, x - element.clientWidth / 2), behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
      }
    }
  }
  function activate(event: KeyboardEvent<SVGGElement>, layerId: string) {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectLayer(layerId); }
  }

  return <section className={`architecture-scene ${playing ? "is-playing" : "is-paused"}`} aria-label="交互式神经网络结构全景">
    <div className="architecture-scene-heading"><div><span className="architecture-scene-kicker"><CircuitBoard size={15} /> NETWORK ARCHITECTURE / 结构全景</span><h2>让网络的每一层，都有迹可循。</h2></div><div className="architecture-scene-actions"><button type="button" className="architecture-flow-toggle" aria-pressed={playing} onClick={() => setPlaying(!playing)}>{playing ? <Pause size={14} /> : <Play size={14} />}{playing ? "暂停数据流" : "播放数据流"}</button><span>{layers.length} 层 · {countText(analysis.totalParameters)} 参数</span></div></div>
    <div className="architecture-canvas-viewport" ref={viewport} tabIndex={0} aria-label="网络结构画布，可横向滚动查看所有层">
      <svg className="architecture-canvas" width={canvasWidth} height="330" viewBox={`0 0 ${canvasWidth} 330`} role="group" aria-label="从输入图像到分类输出的逐层结构图">
        <defs>
          <pattern id={`${id}-grid`} width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="#6bab8e" strokeOpacity=".075" strokeWidth=".6" /></pattern>
          <linearGradient id={`${id}-flow`} x1="0" x2="1"><stop stopColor="#4abda8" stopOpacity=".2" /><stop offset=".55" stopColor="#a7edb1" stopOpacity=".85" /><stop offset="1" stopColor="#d5ec94" stopOpacity=".15" /></linearGradient>
          <radialGradient id={`${id}-aura`}><stop stopColor="#86d5a7" stopOpacity=".15" /><stop offset="1" stopColor="#86d5a7" stopOpacity="0" /></radialGradient>
          <filter id={`${id}-glow`} x="-70%" y="-70%" width="240%" height="240%"><feGaussianBlur stdDeviation="3" /></filter>
        </defs>
        <rect width={canvasWidth} height="330" fill={`url(#${id}-grid)`} />
        <path d={`M25 256L${canvasWidth - 45} 256L${canvasWidth - 19} 236H51Z`} fill="#70a888" fillOpacity=".025" stroke="#5d987b" strokeOpacity=".16" />
        <path d={`M40 265H${canvasWidth - 35}`} stroke="#5d987b" strokeWidth=".7" strokeOpacity=".2" />
        {stages.slice(0, -1).map((stage, index) => {
          const next = stages[index + 1];
          const x1 = position(index) + 57;
          const x2 = position(index + 1) - 49;
          const bend = Math.max(13, (x2 - x1) * .45);
          const broken = Boolean(stage.error || next.error);
          return <g key={`connection-${stage.id}`} className={`architecture-connection ${broken ? "is-broken" : ""}`}>
            {[-24, 0, 24].map((offset, pathIndex) => {
              const d = `M${x1} ${167 + offset}C${x1 + bend} ${167 + offset},${x2 - bend} ${167 + offset * .65},${x2} ${167 + offset * .65}`;
              return <g key={offset}><path d={d} stroke={broken ? "#b87863" : "#69ba9c"} strokeOpacity={broken ? .24 : .2} strokeWidth=".8" fill="none" />{!broken && <path className="architecture-flow" d={d} stroke={`url(#${id}-flow)`} pathLength="100" strokeWidth={pathIndex === 1 ? 1.8 : 1} fill="none" style={{ animationDelay: `${-index * .8 - pathIndex}s` }} />}</g>;
            })}
          </g>;
        })}
        {stages.map((stage, index) => {
          const selected = stage.id === activeId;
          const color = stage.error ? "#e79279" : accents[stage.type];
          const spatial = stage.shape?.length === 3 || Boolean(stage.error && (stage.type === "conv" || stage.type === "pool"));
          return <g key={stage.id} className={`architecture-stage ${selected ? "is-selected" : ""} ${stage.error ? "has-error" : ""}`} transform={`translate(${position(index)} 167)`} role={stage.index >= 0 ? "button" : undefined} tabIndex={stage.index >= 0 ? 0 : undefined} aria-pressed={stage.index >= 0 ? selected : undefined} aria-label={stage.index >= 0 ? `查看第 ${stage.index + 1} 层 ${stage.title}，输出 ${shapeText(stage.shape)}，参数 ${stage.parameters}${stage.error ? "，尺寸错误" : ""}` : "输入图像，1 通道，28 乘 28 像素"} onClick={stage.index >= 0 ? () => selectLayer(stage.id) : undefined} onKeyDown={stage.index >= 0 ? event => activate(event, stage.id) : undefined} style={{ "--architecture-accent": color } as CSSProperties}>
            <rect className="architecture-stage-hit" x="-70" y="-142" width="145" height="298" rx="9" fill="transparent" />
            {selected && <><ellipse cx="0" cy="4" rx="100" ry="108" fill={`url(#${id}-aura)`} /><path d="M-63-123v-8h18 M49-131h18v8 M-63 101v8h18 M49 109h18v-8" fill="none" stroke={color} strokeOpacity=".65" strokeWidth="1.1" /></>}
            <text className="architecture-stage-number" x="0" y="-135" textAnchor="middle" fill={selected ? color : "#668b78"}>{stage.type === "input" ? "SOURCE" : `LAYER ${String(stage.index + 1).padStart(2, "0")}`}</text>
            {stage.type === "input" || spatial ? <SpatialVolume stage={stage} selected={selected} /> : <VectorVolume stage={stage} selected={selected} />}
            <ellipse cx="0" cy="91" rx={spatial || stage.type === "input" ? 51 : 30} ry="8" fill={color} fillOpacity=".04" stroke={color} strokeOpacity={selected ? .38 : .14} strokeWidth=".6" />
            <path d="M0 85v20" stroke={color} strokeOpacity=".25" strokeDasharray="2 3" />
            <circle cx="0" cy="111" r="2.3" fill={color} opacity={selected ? 1 : .45} />
            <text className="architecture-stage-title" x="0" y="132" textAnchor="middle" fill={selected ? "#e0f0d6" : "#b4cfbd"}>{stage.title}</text>
            <text className="architecture-stage-shape" x="0" y="151" textAnchor="middle" fill={stage.error ? color : "#82a990"}>{shapeText(stage.shape)}</text>
          </g>;
        })}
      </svg>
    </div>
    <div className="architecture-canvas-caption"><span><Layers3 size={13} />特征图堆叠与节点为结构示意，尺寸和参数来自当前配置；不代表真实激活值。</span><span>可点选层 · ← 横向滚动 →</span></div>
    {activeLayer && detail ? <div className={`architecture-detail ${activeAnalysis?.error ? "has-error" : ""}`} aria-live="polite">
      <div className="architecture-detail-title"><span>{String(activeIndex + 1).padStart(2, "0")}</span><div><strong>{names[activeLayer.type]}</strong><p>{detail.title}</p></div><div className="architecture-step-controls"><button type="button" aria-label="查看上一网络层" disabled={activeIndex <= 0} onClick={() => selectLayer(layers[activeIndex - 1].id)}><ArrowLeft size={15} /></button><span>{activeIndex + 1} / {layers.length}</span><button type="button" aria-label="查看下一网络层" disabled={activeIndex >= layers.length - 1} onClick={() => selectLayer(layers[activeIndex + 1].id)}><ArrowRight size={15} /></button></div></div>
      <div className="architecture-detail-metrics"><div><span>输入 / IN</span><strong>{shapeText(activeAnalysis?.input ?? null)}</strong></div><ArrowRight size={17} /><div><span>输出 / OUT</span><strong>{shapeText(activeAnalysis?.output ?? null)}</strong></div><div><span>可训练参数</span><strong>{countText(activeAnalysis?.parameters ?? 0)}</strong></div></div>
      <div className="architecture-detail-note">{activeAnalysis?.error ? <><TriangleAlert size={16} /><p>{activeAnalysis.error}</p></> : <><code>{detail.formula}</code><p>{detail.note}</p></>}</div>
    </div> : <div className="architecture-empty"><Box size={20} /><p>从下方层库添加第一层，结构图会实时展开。</p></div>}
  </section>;
}
