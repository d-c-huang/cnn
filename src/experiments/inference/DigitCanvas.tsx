import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { RotateCcw, Undo2 } from "lucide-react";
import { Slider } from "../../components/ui";
import type { CanvasInput, DigitExample } from "../../lib/inference";
import "./inference-visuals.css";

const side = 280;
const maxUndo = 20;
type Point = { x: number; y: number };

export default function DigitCanvas({
  example,
  onChange,
  onDirty,
}: {
  example: DigitExample | null;
  onChange: (input: CanvasInput) => void;
  onDirty: () => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const callbacks = useRef({ onChange, onDirty });
  callbacks.current = { onChange, onDirty };
  const pointer = useRef<number | null>(null);
  const lastPoint = useRef<Point | null>(null);
  const snapshots = useRef<ImageData[]>([]);
  const [brush, setBrush] = useState(22);
  const [empty, setEmpty] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [undoCount, setUndoCount] = useState(0);
  const [mode, setMode] = useState<CanvasInput["mode"]>("canvas");

  function context() {
    return (
      canvas.current?.getContext("2d", { willReadFrequently: true }) ?? null
    );
  }

  function emit(input: CanvasInput) {
    setEmpty(input.empty);
    setMode(input.mode);
    callbacks.current.onChange(input);
  }

  function readCanvas(): CanvasInput {
    const pixels = context()?.getImageData(0, 0, side, side).data;
    let hasInk = false;
    const matrix = Array.from({ length: side }, (_, row) =>
      Array.from({ length: side }, (_, column) => {
        const offset = (row * side + column) * 4;
        const value = pixels
          ? Math.round(
              pixels[offset] * 0.299 +
                pixels[offset + 1] * 0.587 +
                pixels[offset + 2] * 0.114,
            )
          : 0;
        if (value > 16) hasInk = true;
        return value;
      }),
    );
    return { pixels: matrix, mode: "canvas", empty: !hasInk };
  }

  function remember() {
    const image = context()?.getImageData(0, 0, side, side);
    if (!image) return;
    snapshots.current.push(image);
    if (snapshots.current.length > maxUndo) snapshots.current.shift();
    setUndoCount(snapshots.current.length);
  }

  function releasePointer() {
    const id = pointer.current;
    pointer.current = null;
    lastPoint.current = null;
    setDrawing(false);
    if (id !== null && canvas.current?.hasPointerCapture(id)) {
      canvas.current.releasePointerCapture(id);
    }
  }

  useEffect(() => {
    const ctx = context();
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, side, side);
    snapshots.current = [];
    setUndoCount(0);
    emit({
      pixels: Array.from({ length: side }, () => Array<number>(side).fill(0)),
      mode: "canvas",
      empty: true,
    });
    return () => {
      const id = pointer.current;
      pointer.current = null;
      lastPoint.current = null;
      if (id !== null && canvas.current?.hasPointerCapture(id)) {
        canvas.current.releasePointerCapture(id);
      }
    };
    // The canvas backing store is initialized once, never when its CSS size changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (
      !example ||
      example.pixels.length !== 28 ||
      example.pixels.some((row) => row.length !== 28)
    )
      return;
    const ctx = context();
    if (!ctx) return;
    releasePointer();
    callbacks.current.onDirty();
    remember();
    const source = document.createElement("canvas");
    source.width = 28;
    source.height = 28;
    const sourceContext = source.getContext("2d");
    if (!sourceContext) return;
    const image = sourceContext.createImageData(28, 28);
    let hasInk = false;
    for (let row = 0; row < 28; row++) {
      for (let column = 0; column < 28; column++) {
        const value = Math.max(
          0,
          Math.min(255, Math.round(example.pixels[row][column])),
        );
        const offset = (row * 28 + column) * 4;
        image.data[offset] =
          image.data[offset + 1] =
          image.data[offset + 2] =
            value;
        image.data[offset + 3] = 255;
        if (example.pixels[row][column] > 0) hasInk = true;
      }
    }
    sourceContext.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(source, 0, 0, side, side);
    emit({
      pixels: example.pixels.map((row) => [...row]),
      mode: "mnist",
      empty: !hasInk,
    });
    // Callback refs prevent a parent render from redrawing an already loaded example.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [example]);

  function position(event: { clientX: number; clientY: number }): Point {
    const bounds = canvas.current!.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(side, ((event.clientX - bounds.left) / bounds.width) * side),
      ),
      y: Math.max(
        0,
        Math.min(side, ((event.clientY - bounds.top) / bounds.height) * side),
      ),
    };
  }

  function start(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (pointer.current !== null || event.button !== 0) return;
    const ctx = context();
    if (!ctx) return;
    event.preventDefault();
    remember();
    callbacks.current.onDirty();
    pointer.current = event.pointerId;
    lastPoint.current = position(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawing(true);
    setEmpty(false);
    setMode("canvas");
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.arc(
      lastPoint.current.x,
      lastPoint.current.y,
      brush / 2,
      0,
      2 * Math.PI,
    );
    ctx.fill();
  }

  function move(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (pointer.current !== event.pointerId || !lastPoint.current) return;
    const ctx = context();
    if (!ctx) return;
    event.preventDefault();
    const coalesced = event.nativeEvent.getCoalescedEvents?.() || [];
    const points = coalesced.length ? coalesced : [event.nativeEvent];
    ctx.beginPath();
    ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
    for (const item of points) {
      const next = position(item);
      ctx.lineTo(next.x, next.y);
      lastPoint.current = next;
    }
    ctx.stroke();
  }

  function finish(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (pointer.current !== event.pointerId) return;
    releasePointer();
    emit(readCanvas());
  }

  function clear() {
    if (empty && !drawing) return;
    releasePointer();
    callbacks.current.onDirty();
    remember();
    const ctx = context();
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, side, side);
    emit(readCanvas());
  }

  function undo() {
    const previous = snapshots.current.pop();
    if (!previous) return;
    releasePointer();
    callbacks.current.onDirty();
    context()?.putImageData(previous, 0, 0);
    setUndoCount(snapshots.current.length);
    // Restored image data is a canvas edit, not a new authoritative MNIST sample.
    emit(readCanvas());
  }

  return (
    <div className="iv-drawing">
      <div className="iv-canvas-surface">
        <canvas
          ref={canvas}
          width={side}
          height={side}
          role="img"
          aria-label="手写数字画布。可用鼠标、触控或笔书写，也可载入真实 MNIST 示例。"
          onPointerDown={start}
          onPointerMove={move}
          onPointerUp={finish}
          onPointerCancel={finish}
          onLostPointerCapture={finish}
        >
          请使用支持 Canvas 的浏览器，或载入真实 MNIST 示例进行识别。
        </canvas>
        {empty && (
          <div className="iv-canvas-hint" aria-hidden="true">
            <strong>0–9</strong>
            <span>在这里写一个数字</span>
          </div>
        )}
      </div>
      <div className="iv-canvas-status" role="status">
        {drawing
          ? "正在书写…"
          : empty
            ? "画布为空，写下数字或载入示例。"
            : mode === "mnist"
              ? "已载入原始 MNIST 样本，可直接识别或继续书写。"
              : "笔迹已就绪，点击识别查看模型输出。"}
      </div>
      <div className="iv-canvas-controls">
        <div className="iv-brush-control">
          <Slider
            label="画笔粗细"
            value={brush}
            min={8}
            max={36}
            step={1}
            onChange={setBrush}
          />
        </div>
        <div className="iv-canvas-actions">
          <button
            className="secondary-button"
            disabled={!undoCount || drawing}
            onClick={undo}
          >
            <Undo2 size={14} />
            撤销
          </button>
          <button
            className="secondary-button"
            disabled={empty && !drawing}
            onClick={clear}
          >
            <RotateCcw size={14} />
            清空
          </button>
        </div>
      </div>
      <p className="iv-canvas-note">
        黑底白笔迹。每次书写、清空或撤销都会使旧预测失效；编辑后将按手绘图像重新预处理。
      </p>
    </div>
  );
}
