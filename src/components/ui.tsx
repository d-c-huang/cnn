import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import { AlertCircle, SlidersHorizontal } from "lucide-react";
import { format } from "../lib/api";

export function Slider({
  label,
  symbol,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  symbol?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  return (
    <div className="slider-field">
      <div className="slider-heading">
        <label htmlFor={id}>
          {label} {symbol && <span className="math-symbol">{symbol}</span>}
        </label>
        <output htmlFor={id}>{format(value)}</output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="slider-limits" aria-hidden="true">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export function ExperimentPanel({
  title,
  eyebrow,
  children,
  action,
  className = "",
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`experiment-panel ${className}`}>
      <div className="panel-heading">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Result({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="result">
      <span className="small-label">{label}</span>
      <strong>{value}</strong>
      {hint && <span className="result-hint">{hint}</span>}
    </div>
  );
}

export function ErrorMessage({ error }: { error: string | null }) {
  return error ? (
    <div className="error-message" role="alert">
      <AlertCircle size={17} />
      <span>
        {error === "Failed to fetch"
          ? "无法连接计算服务，请使用 npm run dev 启动前后端。"
          : error}
      </span>
    </div>
  ) : null;
}

export function ControlsHeading({
  children = "调整参数",
  action,
}: {
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="controls-heading">
      <h3>
        <SlidersHorizontal size={16} />
        {children}
      </h3>
      {action}
    </div>
  );
}

export type Point = { x: number; y: number };
type Series = { points: Point[]; color?: string; dashed?: boolean };
type Marker = Point & { label?: string; color?: string };

export function Chart({
  series,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  markers = [],
  ariaLabel,
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  markers?: Marker[];
  ariaLabel: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(500);
  const [hover, setHover] = useState<Point | null>(null);
  const id = useId().replace(/:/g, "");
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(entries[0].contentRect.width),
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const all = [...series.flatMap((s) => s.points), ...markers].filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y),
  );
  function extent(
    values: number[],
    specified?: [number, number],
  ): [number, number] {
    if (specified) return specified;
    if (!values.length) return [-1, 1];
    const low = Math.min(...values),
      high = Math.max(...values);
    const padding = (high - low || Math.abs(low) || 1) * 0.12;
    return [low - padding, high + padding];
  }
  const dx = extent(
    all.map((p) => p.x),
    xDomain,
  );
  const dy = extent(
    all.map((p) => p.y),
    yDomain,
  );
  const h = 286,
    left = 58,
    right = 22,
    top = 30,
    bottom = 48;
  const plotWidth = Math.max(width - left - right, 10),
    plotHeight = h - top - bottom;
  const sx = (x: number) => left + ((x - dx[0]) / (dx[1] - dx[0])) * plotWidth;
  const sy = (y: number) => top + ((dy[1] - y) / (dy[1] - dy[0])) * plotHeight;
  const ticks = width < 420 ? 3 : 4;
  const values = (domain: [number, number]) =>
    Array.from(
      { length: ticks + 1 },
      (_, i) => domain[0] + ((domain[1] - domain[0]) * i) / ticks,
    );
  const path = (points: Point[]) =>
    points
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
      .map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`)
      .join(" ");
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${width} ${h}`} role="img" aria-label={ariaLabel}>
        <defs>
          <clipPath id={id}>
            <rect x={left} y={top} width={plotWidth} height={plotHeight} />
          </clipPath>
        </defs>
        {values(dy).map((v, i) => (
          <g key={`y${i}`}>
            <line
              className="grid-line"
              x1={left}
              x2={left + plotWidth}
              y1={sy(v)}
              y2={sy(v)}
            />
            <text
              className="chart-tick"
              x={left - 10}
              y={sy(v) + 4}
              textAnchor="end"
            >
              {format(v, 1)}
            </text>
          </g>
        ))}
        {values(dx).map((v, i) => (
          <g key={`x${i}`}>
            <line
              className="grid-line vertical"
              y1={top}
              y2={top + plotHeight}
              x1={sx(v)}
              x2={sx(v)}
            />
            <text
              className="chart-tick"
              x={sx(v)}
              y={h - 27}
              textAnchor="middle"
            >
              {format(v, 1)}
            </text>
          </g>
        ))}
        <text className="chart-axis-label" x={left} y={15}>
          {yLabel}
        </text>
        <text
          className="chart-axis-label"
          x={left + plotWidth / 2}
          y={h - 4}
          textAnchor="middle"
        >
          {xLabel}
        </text>
        <g clipPath={`url(#${id})`}>
          {dx[0] < 0 && dx[1] > 0 && (
            <line
              className="zero-line"
              x1={sx(0)}
              x2={sx(0)}
              y1={top}
              y2={top + plotHeight}
            />
          )}
          {dy[0] < 0 && dy[1] > 0 && (
            <line
              className="zero-line"
              x1={left}
              x2={left + plotWidth}
              y1={sy(0)}
              y2={sy(0)}
            />
          )}
          {series.map((s, i) => (
            <path
              key={i}
              d={path(s.points)}
              fill="none"
              stroke={s.color || "var(--ink)"}
              strokeWidth="2.5"
              strokeDasharray={s.dashed ? "5 5" : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {markers.map((p, i) => (
            <g key={i}>
              <circle
                cx={sx(p.x)}
                cy={sy(p.y)}
                r={7}
                fill={p.color || "var(--lime)"}
                stroke="var(--ink)"
                strokeWidth={1.5}
              >
                <title>
                  {p.label || ""} ({format(p.x)}, {format(p.y)})
                </title>
              </circle>
            </g>
          ))}
          {hover && (
            <>
              <line
                className="hover-guide"
                x1={sx(hover.x)}
                x2={sx(hover.x)}
                y1={top}
                y2={top + plotHeight}
              />
              <circle
                cx={sx(hover.x)}
                cy={sy(hover.y)}
                r={4}
                fill="var(--ink)"
              />
            </>
          )}
          <rect
            x={left}
            y={top}
            width={plotWidth}
            height={plotHeight}
            fill="transparent"
            onPointerLeave={() => setHover(null)}
            onPointerMove={(e) => {
              const bounds =
                e.currentTarget.ownerSVGElement!.getBoundingClientRect();
              const cursorX =
                dx[0] +
                ((((e.clientX - bounds.left) * width) / bounds.width - left) /
                  plotWidth) *
                  (dx[1] - dx[0]);
              const points = series[0]?.points || [];
              if (points.length)
                setHover(
                  points.reduce((nearest, p) =>
                    Math.abs(p.x - cursorX) < Math.abs(nearest.x - cursorX)
                      ? p
                      : nearest,
                  ),
                );
            }}
          />
        </g>
      </svg>
      {hover && (
        <div className="chart-tooltip">
          {xLabel}: {format(hover.x)} <span>·</span> {yLabel}: {format(hover.y)}
        </div>
      )}
    </div>
  );
}
