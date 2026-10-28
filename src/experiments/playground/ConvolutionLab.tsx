import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, ChevronLeft, ChevronRight, Grid3X3, Pause, Play, RotateCcw, ScanLine } from "lucide-react";
import { convolutionWindow, crossCorrelate, maxPool2d, outputSize, relu } from "../../lib/convolution";
import type { Matrix } from "../../lib/convolution";
import "./convolution-lab.css";

const imagePatterns = {
  edge: ["00001111", "00001111", "00001111", "00001111", "00001111", "00001111", "00001111", "00001111"],
  digit: ["00000000", "00111100", "00000110", "00011100", "00000110", "00000110", "00111100", "00000000"],
  cross: ["00011000", "00011000", "00011000", "11111111", "11111111", "00011000", "00011000", "00011000"],
};
const kernelPresets: Record<string, { name: string; matrix: Matrix; note: string }> = {
  vertical: { name: "垂直边缘", matrix: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], note: "比较左右像素的差异；对垂直边界产生响应。" },
  horizontal: { name: "水平边缘", matrix: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], note: "比较上下像素的差异；对水平边界产生响应。" },
  blur: { name: "均值模糊", matrix: Array.from({ length: 3 }, () => Array(3).fill(1 / 9) as number[]), note: "九个像素等权平均，让局部变化更平滑。" },
  sharpen: { name: "锐化", matrix: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], note: "强调中心像素与邻域之间的差异，增强局部细节。" },
};

function makeImage(name: keyof typeof imagePatterns): Matrix {
  return imagePatterns[name].map(row => Array.from(row, Number));
}

function numberText(value: number) {
  return Number.isInteger(value) ? String(value) : Number(value.toFixed(3)).toString();
}

function cellText(value: number) {
  return Number(value.toFixed(2)).toString().replace(/^(-?)0\./, "$1.");
}

function heatColor(value: number, scale: number) {
  if (Math.abs(value) < 1e-10) return "#0b1d20";
  const ratio = Math.min(1, Math.abs(value) / Math.max(scale, 1e-10));
  return value > 0 ? `rgba(84, 214, 183, ${0.12 + ratio * 0.51})` : `rgba(225, 158, 100, ${0.12 + ratio * 0.5})`;
}

export default function ConvolutionLab({ onInteract }: { onInteract?: () => void }) {
  const [image, setImage] = useState<Matrix>(() => makeImage("edge"));
  const [imageName, setImageName] = useState<string>("edge");
  const [kernel, setKernel] = useState<Matrix>(() => kernelPresets.vertical.matrix.map(row => [...row]));
  const [kernelName, setKernelName] = useState("vertical");
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [useRelu, setUseRelu] = useState(false);
  const [usePool, setUsePool] = useState(true);
  const [selected, setSelected] = useState(0);
  const [running, setRunning] = useState(false);
  const pixelRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const feature = useMemo(() => crossCorrelate(image, kernel, stride, padding), [image, kernel, stride, padding]);
  const activated = useMemo(() => useRelu ? relu(feature) : feature, [feature, useRelu]);
  const pooled = useMemo(() => maxPool2d(activated), [activated]);
  const side = feature.length;
  const count = side * side;
  const position = Math.min(selected, count - 1);
  const row = Math.floor(position / side);
  const column = position % side;
  const window = convolutionWindow(image, kernel, row, column, stride, padding);
  const featureScale = Math.max(...feature.flat().map(Math.abs));
  const poolScale = Math.max(...pooled.flat().map(Math.abs));
  const sum = feature[row][column];

  useEffect(() => {
    if (!running) return;
    const timer = globalThis.setInterval(() => setSelected(current => (current + 1) % count), 700);
    return () => globalThis.clearInterval(timer);
  }, [running, count]);

  function change(action: () => void) {
    setRunning(false);
    setSelected(0);
    action();
    onInteract?.();
  }

  function move(next: number) {
    setRunning(false);
    setSelected(Math.max(0, Math.min(count - 1, next)));
    onInteract?.();
  }

  return (
    <div className="conv-lab">
      <div className="conv-intro"><ScanLine size={18} /><p>用一个 3 × 3 小窗口扫描图像，亲眼看见特征如何产生。所有数值在浏览器中实时计算，使用手工教学图像与卷积核。</p><span>LOCAL COMPUTE</span></div>
      <div className="conv-layout">
        <section className="conv-workbench" aria-labelledby="conv-scan-title">
          <div className="conv-panel-heading"><div><span>01 / CONVOLUTION SCANNER</span><h2 id="conv-scan-title">扫描一小块，理解一整层。</h2></div><span className="conv-coordinate">Y[{row}, {column}]</span></div>
          <div className="conv-flow">
            <div className="conv-input-block"><div className="conv-grid-label"><span>输入图像</span><code>8 × 8</code></div>
              <div className="conv-image-grid" role="group" aria-label="8 行 8 列教学图像，点击切换 0 与 1，方向键移动">
                {image.flatMap((values, ir) => values.map((value, ic) => {
                  const index = ir * 8 + ic;
                  const active = window.some(item => !item.padded && item.row === ir && item.column === ic);
                  return <button key={index} ref={element => { pixelRefs.current[index] = element; }} className={`conv-pixel ${value ? "is-on" : ""} ${active ? "is-window" : ""}`} aria-label={`输入行 ${ir + 1} 列 ${ic + 1}，像素 ${value}`} aria-pressed={value === 1} onClick={() => change(() => { setImage(current => current.map((r, ri) => r.map((v, ci) => ri === ir && ci === ic ? 1 - v : v))); setImageName("custom"); })} onKeyDown={event => {
                    const deltas: Record<string, [number, number]> = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] };
                    const delta = deltas[event.key];
                    if (delta) { event.preventDefault(); pixelRefs.current[Math.max(0, Math.min(7, ir + delta[0])) * 8 + Math.max(0, Math.min(7, ic + delta[1]))]?.focus(); }
                  }}>{value}</button>;
                }))}
              </div><p className="conv-grid-note">点击编辑像素；绿色框为当前感受野。</p>
            </div>
            <div className="conv-flow-operation" aria-hidden="true">∗<span>3 × 3</span></div>
            <div className="conv-feature-block"><div className="conv-grid-label"><span>卷积特征图</span><code>{side} × {side}</code></div>
              <div className="conv-feature-grid" style={{ gridTemplateColumns: `repeat(${side}, minmax(0, 1fr))` }} role="group" aria-label="卷积输出特征图，选择单元格查看计算">
                {feature.flatMap((values, ir) => values.map((value, ic) => <button key={`${ir}-${ic}`} className={`conv-feature-cell ${ir * side + ic === position ? "is-selected" : ""}`} style={{ background: heatColor(value, featureScale) }} aria-label={`输出行 ${ir + 1} 列 ${ic + 1}，值 ${numberText(value)}`} aria-pressed={ir * side + ic === position} onClick={() => move(ir * side + ic)}>{cellText(value)}</button>))}
              </div><div className="conv-heat-legend"><span>− 负响应</span><i /><span>+ 正响应</span></div>
            </div>
          </div>
          <div className="conv-playback"><div><button className="conv-icon-button" aria-label="上一个卷积位置" disabled={position === 0} onClick={() => move(position - 1)}><ChevronLeft size={17} /></button><button className="primary-button" onClick={() => { setRunning(!running); onInteract?.(); }}>{running ? <Pause size={13} /> : <Play size={13} />}{running ? "暂停扫描" : "开始扫描"}</button><button className="conv-icon-button" aria-label="下一个卷积位置" disabled={position === count - 1} onClick={() => move(position + 1)}><ChevronRight size={17} /></button></div><span>{position + 1}<em> / {count} 个位置</em></span></div>
          <div className="conv-size-equation"><code>⌊(N + 2P − K) / S⌋ + 1</code><span>= ⌊(8 + {2 * padding} − 3) / {stride}⌋ + 1 = <strong>{outputSize(8, 3, stride, padding)}</strong></span></div>
        </section>

        <aside className="conv-controls" aria-label="卷积参数控制">
          <div className="conv-controls-title"><h2><Grid3X3 size={16} />配置扫描器</h2><button className="conv-icon-button" aria-label="重置卷积实验" onClick={() => change(() => { setImage(makeImage("edge")); setImageName("edge"); setKernel(kernelPresets.vertical.matrix.map(r => [...r])); setKernelName("vertical"); setStride(1); setPadding(0); setUseRelu(false); setUsePool(true); })}><RotateCcw size={15} /></button></div>
          <label className="conv-field">教学图像<select value={imageName} onChange={event => change(() => { const name = event.target.value as keyof typeof imagePatterns; setImage(makeImage(name)); setImageName(name); })}><option value="edge">垂直边界</option><option value="digit">手工数字 3</option><option value="cross">十字图案</option>{imageName === "custom" && <option value="custom" disabled>自定义图像</option>}</select></label>
          <label className="conv-field">卷积核<select value={kernelName} onChange={event => change(() => { const name = event.target.value; setKernelName(name); setKernel(kernelPresets[name].matrix.map(r => [...r])); })}>{Object.entries(kernelPresets).map(([key, item]) => <option key={key} value={key}>{item.name}</option>)}{kernelName === "custom" && <option value="custom" disabled>自定义卷积核</option>}</select></label>
          <div className="conv-kernel-grid" role="group" aria-label="可编辑 3 行 3 列卷积核">{kernel.flatMap((values, kr) => values.map((value, kc) => <input key={`${kr}-${kc}`} type="number" min={-9} max={9} step="any" value={value} aria-label={`卷积核行 ${kr + 1} 列 ${kc + 1}`} onChange={event => { const next = Number(event.target.value); if (Number.isFinite(next)) change(() => { setKernel(current => current.map((r, ri) => r.map((v, ci) => ri === kr && ci === kc ? Math.max(-9, Math.min(9, next)) : v))); setKernelName("custom"); }); }} />))}</div>
          <p className="conv-control-note">{kernelPresets[kernelName]?.note ?? "修改权重，观察特征图对局部像素的响应。"}</p>
          <div className="conv-geometry"><label className="conv-field">步幅 S<select value={stride} onChange={event => change(() => setStride(Number(event.target.value)))}><option value="1">1 · 逐格扫描</option><option value="2">2 · 隔格扫描</option></select></label><label className="conv-field">零填充 P<select value={padding} onChange={event => change(() => setPadding(Number(event.target.value)))}><option value="0">0 · 不填充</option><option value="1">1 · 四周补零</option></select></label></div>
          <label className="conv-checkbox"><input type="checkbox" checked={useRelu} onChange={event => change(() => setUseRelu(event.target.checked))} /><span>应用 ReLU：负数变为 0</span></label>
          <label className="conv-checkbox"><input type="checkbox" checked={usePool} onChange={event => change(() => setUsePool(event.target.checked))} /><span>显示 2 × 2 最大池化</span></label>
          <div className="conv-method-note">这里演示深度学习中的“卷积”：卷积核不翻转，数学上称为互相关。偏置固定为 0；显示值经过舍入，计算保留完整精度。</div>
        </aside>
      </div>

      <section className="conv-calculation" aria-labelledby="conv-calc-title">
        <div className="conv-panel-heading"><div><span>02 / EVERY MULTIPLICATION MATTERS</span><h2 id="conv-calc-title">把当前输出，拆成九次乘法。</h2></div><code>Y[{row}, {column}] = {numberText(sum)}</code></div>
        <div className="conv-products">{window.map((item, index) => <div key={index} className={`conv-product ${item.padded ? "is-padded" : ""}`}><span>{item.padded ? "零填充" : `X[${item.row}, ${item.column}]`}</span><code>{numberText(item.input)} × {numberText(item.kernel)}</code><strong>{numberText(item.product)}</strong></div>)}</div>
        <div className="conv-sum-line"><span>逐项相乘，然后求和</span><code>Σ(X × K) = <strong>{numberText(sum)}</strong></code>{useRelu && <><ArrowRight size={15} /><code>ReLU = <strong>{numberText(Math.max(0, sum))}</strong></code></>}</div>
      </section>

      <section className="conv-pipeline" aria-labelledby="conv-pipeline-title"><div className="conv-panel-heading"><div><span>03 / FEATURE EXTRACTION PIPELINE</span><h2 id="conv-pipeline-title">提取响应，再压缩空间。</h2></div><span className="conv-pipeline-shape">{side} × {side}{usePool ? ` → ${pooled.length} × ${pooled[0]?.length ?? 0}` : " · 原始尺寸"}</span></div>
        <div className="conv-pipeline-body"><div className="conv-activation-copy"><span className="conv-step-tag">{useRelu ? "ReLU 已开启" : "线性输出"}</span><h3>{useRelu ? "保留正响应，滤去负响应。" : "同时观察正向与负向响应。"}</h3><p>{useRelu ? "ReLU(x) = max(0, x)，逐个元素变换，不改变特征图尺寸。" : "当前没有应用激活函数，所有正、负卷积结果都保留下来。"}</p><div className="conv-stage-chain"><span>卷积</span><ArrowRight size={13} /><span>{useRelu ? "ReLU" : "恒等映射"}</span><ArrowRight size={13} /><span>{usePool ? "MaxPool 2 × 2" : "不池化"}</span></div><p>此处展示一个固定核对单通道图像的计算；训练中的 CNN 会从数据中学习多个核。</p></div>
          <div className="conv-pool-block"><div className="conv-grid-label"><span>{usePool ? "最大池化输出" : "激活后的特征图"}</span><code>{usePool ? `${pooled.length} × ${pooled[0]?.length ?? 0}` : `${side} × ${side}`}</code></div><div className="conv-pool-grid" style={{ gridTemplateColumns: `repeat(${usePool ? pooled[0]?.length ?? 1 : side}, minmax(0, 1fr))` }} role="group" aria-label={usePool ? "2 乘 2 步幅 2 最大池化结果" : "激活后的特征图"}>{(usePool ? pooled : activated).flatMap((values, ir) => values.map((value, ic) => <span key={`${ir}-${ic}`} style={{ background: heatColor(value, usePool ? poolScale : featureScale) }} title={`行 ${ir + 1} 列 ${ic + 1}：${numberText(value)}`}>{cellText(value)}</span>))}</div><p className="conv-grid-note">{usePool ? "每个 2 × 2 区域取最大值，步幅为 2；边缘不足一窗时舍弃。" : "未执行池化，空间分辨率保持不变。"}</p></div>
        </div>
      </section>
    </div>
  );
}
