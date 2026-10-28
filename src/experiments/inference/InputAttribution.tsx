import { useId, useMemo, useState } from "react";
import { Eye, ScanEye } from "lucide-react";
import type { InputAttributionData } from "../../lib/inference";
import "./input-attribution.css";

function heatColor(value: number) {
  // Relative gradient magnitude controls colour and opacity; zero has no overlay.
  const channel = (from: number, to: number) => Math.round(from + (to - from) * value);
  return `rgb(${channel(85, 255)}, ${channel(205, 115)}, ${channel(190, 65)})`;
}

export default function InputAttribution({ attribution, pixels }: { attribution: InputAttributionData; pixels: number[][] }) {
  const [opacity, setOpacity] = useState(.72);
  const [overlay, setOverlay] = useState(true);
  const id = useId();
  const valid = attribution.method === "input_gradient" && attribution.heatmap.length === 28 && attribution.heatmap.every(row => row.length === 28 && row.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) && pixels.length === 28 && pixels.every(row => row.length === 28 && row.every(Number.isFinite)) && Number.isFinite(attribution.maximum_gradient) && attribution.maximum_gradient >= 0;
  const basePixels = useMemo(() => pixels.flatMap((row, y) => row.map((value, x) => <rect key={`${y}-${x}`} x={x} y={y} width={1} height={1} fill={`rgb(${Math.round(value)}, ${Math.round(value)}, ${Math.round(value)})`} />)), [pixels]);
  const heatPixels = useMemo(() => attribution.heatmap.flatMap((row, y) => row.map((value, x) => <rect key={`${y}-${x}`} x={x} y={y} width={1} height={1} fill={heatColor(value)} opacity={value} />)), [attribution.heatmap]);
  if (!valid) return <p className="ia-invalid" role="alert">输入敏感度数据格式异常，请重新识别。</p>;
  const zero = attribution.maximum_gradient === 0;
  return <section className="input-attribution" aria-labelledby={`${id}-title`}>
    <div className="ia-heading"><h3 id={`${id}-title`}><ScanEye size={17} />输入敏感度热力图</h3><span>INPUT GRADIENT</span></div>
    <p className="ia-description">针对本次预测类别 <strong>{attribution.target_label}</strong>，观察哪些输入位置的微小变化更容易影响该类别的原始输出（logit）。</p>
    <div className="ia-images">
      <figure><svg viewBox="0 0 28 28" role="img" aria-label="本次推理使用的 28 乘 28 预处理灰度原图" shapeRendering="crispEdges">{basePixels}</svg><figcaption>同一次请求的预处理原图</figcaption></figure>
      <figure><svg className="ia-heatmap" viewBox="0 0 28 28" role="img" aria-label={overlay ? `预测类别 ${attribution.target_label} 的相对输入敏感度叠加图${zero ? "，梯度为零，没有响应" : ""}` : "热力叠加已关闭，当前显示灰度原图"} shapeRendering="crispEdges">{basePixels}{overlay && <g opacity={opacity}>{heatPixels}</g>}</svg><figcaption>{overlay ? `类别 ${attribution.target_label} · 敏感度叠加` : "仅显示原图"}</figcaption></figure>
    </div>
    <div className="ia-controls"><label htmlFor={`${id}-opacity`}>叠加透明度 <output htmlFor={`${id}-opacity`}>{Math.round(opacity * 100)}%</output><input id={`${id}-opacity`} aria-label="热力图叠加透明度" type="range" min={0} max={1} step={.01} value={opacity} disabled={!overlay} onChange={event => setOpacity(Number(event.target.value))} /></label><button className="ia-toggle" aria-pressed={overlay} onClick={() => setOverlay(!overlay)}><Eye size={15} />{overlay ? "查看原图" : "显示热力叠加"}</button></div>
    <div className="ia-legend"><span>较低</span><i aria-hidden="true" /><span>较高</span><span>相对敏感度 0–1</span></div>
    <p className="ia-method"><code>|∂ logit[{attribution.target_label}] / ∂ x标准|</code><span>逐像素取绝对值，再按本图最大梯度归一化。最大梯度 {attribution.maximum_gradient === 0 ? "0" : attribution.maximum_gradient.toExponential(3)}。</span></p>
    {zero && <p className="ia-zero" role="status">本次输入位置的梯度全部为 0，热力图没有显示响应；这不等于模型没有使用输入。</p>}
    <p className="ia-caution">亮度代表输入敏感度，不是因果证明或正确率保证。不同模型的热力图分别归一化，颜色不能直接比较梯度绝对大小。</p>
  </section>;
}
