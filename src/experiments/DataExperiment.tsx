import { useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useCalculation } from "../lib/api";
import {
  Slider,
  ExperimentPanel,
  Result,
  ErrorMessage,
} from "../components/ui";

interface DataResponse {
  matrix: number[][];
  flat: number[];
  shape: [number, number];
  count: number;
  min: number;
  max: number;
  mean: number;
}

interface ExperimentProps {
  onInteract: () => void;
}

type Sample = "3" | "7" | "blank";

const SAMPLE_ROWS: Record<Sample, string[]> = {
  "3": [
    "00000000",
    "00111100",
    "00000110",
    "00011100",
    "00000110",
    "00000110",
    "00111100",
    "00000000",
  ],
  "7": [
    "00000000",
    "01111110",
    "00000110",
    "00001100",
    "00011000",
    "00011000",
    "00011000",
    "00000000",
  ],
  blank: [
    "00000000",
    "00000000",
    "00000000",
    "00000000",
    "00000000",
    "00000000",
    "00000000",
    "00000000",
  ],
};

function createSample(sample: Sample): number[][] {
  return SAMPLE_ROWS[sample].map((row) =>
    Array.from(row, (value) => (value === "1" ? 255 : 0)),
  );
}

function display(value: number | undefined, digits = 3): string {
  if (value === undefined) return "—";
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

export default function DataExperiment({ onInteract }: ExperimentProps) {
  const [pixels, setPixels] = useState<number[][]>(() => createSample("3"));
  const [sample, setSample] = useState<Sample | null>("3");
  const [selected, setSelected] = useState(19);
  const [normalize, setNormalize] = useState(true);
  const pixelRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const payload = useMemo(() => ({ pixels, normalize }), [pixels, normalize]);
  const { data, error, pending } = useCalculation<DataResponse>(
    "/data",
    payload,
  );
  const row = Math.floor(selected / 8);
  const column = selected % 8;
  const rawValue = pixels[row]?.[column] ?? 0;
  const convertedValue = data?.matrix[row]?.[column];

  function updatePixel(index: number, value: number) {
    const targetRow = Math.floor(index / 8);
    const targetColumn = index % 8;
    setPixels((current) =>
      current.map((values, i) =>
        i === targetRow
          ? values.map((pixel, j) => (j === targetColumn ? value : pixel))
          : values,
      ),
    );
    setSample(null);
    onInteract();
  }

  function chooseSample(next: Sample) {
    setSample(next);
    setPixels(createSample(next));
    onInteract();
  }

  function movePixel(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const targetRow = Math.floor(index / 8);
    const targetColumn = index % 8;
    let next = index;
    if (event.key === "ArrowRight")
      next = targetRow * 8 + Math.min(7, targetColumn + 1);
    else if (event.key === "ArrowLeft")
      next = targetRow * 8 + Math.max(0, targetColumn - 1);
    else if (event.key === "ArrowUp")
      next = Math.max(0, targetRow - 1) * 8 + targetColumn;
    else if (event.key === "ArrowDown")
      next = Math.min(7, targetRow + 1) * 8 + targetColumn;
    else return;
    event.preventDefault();
    setSelected(next);
    pixelRefs.current[next]?.focus();
  }

  return (
    <div className="experiment-layout">
      <div className="experiment-main">
        <ExperimentPanel
          title="一张图片，也是一张数字表"
          eyebrow="01 / DATA → TENSOR"
        >
          <div className="data-workbench">
            <div className="data-image-area">
              <div className="small-label">8 × 8 灰度像素 / 手工构造示例</div>
              <div
                className="pixel-grid"
                role="group"
                aria-label="8 行 8 列像素编辑器，空格切换像素，方向键移动"
              >
                {pixels.flatMap((values, i) =>
                  values.map((value, j) => {
                    const index = i * 8 + j;
                    return (
                      <button
                        key={index}
                        ref={(element) => {
                          pixelRefs.current[index] = element;
                        }}
                        type="button"
                        className={`pixel-cell${selected === index ? " is-selected" : ""}`}
                        style={{
                          backgroundColor: `rgb(${value}, ${value}, ${value})`,
                        }}
                        aria-label={`第 ${i + 1} 行，第 ${j + 1} 列，灰度 ${value}；按下切换黑白`}
                        aria-pressed={selected === index}
                        onFocus={() => setSelected(index)}
                        onKeyDown={(event) => movePixel(event, index)}
                        onClick={() => {
                          setSelected(index);
                          updatePixel(index, value > 127 ? 0 : 255);
                        }}
                      >
                        <span className="pixel-selection" aria-hidden="true">
                          {selected === index ? "+" : ""}
                        </span>
                      </button>
                    );
                  }),
                )}
              </div>
              <p className="panel-note">
                点击切换黑白；选中后可调灰度。此处是 8 × 8 教学样本，不是 MNIST
                数据。
              </p>
            </div>
            <div className="data-matrix-area" aria-busy={pending}>
              <div className="small-label">
                送入模型的矩阵 {pending ? "· 正在计算…" : ""}
              </div>
              <div
                className="tensor-matrix"
                role="group"
                aria-label="后端转换后的 8 行 8 列矩阵"
              >
                {Array.from({ length: 64 }, (_, index) => (
                  <span
                    key={index}
                    className={`tensor-cell${selected === index ? " is-selected" : ""}`}
                  >
                    {display(
                      data?.matrix[Math.floor(index / 8)]?.[index % 8],
                      2,
                    )}
                  </span>
                ))}
              </div>
              <p className="axis-note">行优先读取：从左到右，再从上到下。</p>
            </div>
          </div>
          <div className="result-grid" aria-live="polite">
            <Result
              label="矩阵形状"
              value={data ? `[${data.shape.join(", ")}]` : "—"}
              hint="高 × 宽；灰度图只有一个通道"
            />
            <Result
              label="元素数量"
              value={display(data?.count)}
              hint="每个元素对应一个像素"
            />
            <Result
              label="转换后均值"
              value={display(data?.mean)}
              hint={
                data
                  ? `范围 ${display(data.min)} → ${display(data.max)}`
                  : "等待后端计算"
              }
            />
          </div>
          <ErrorMessage error={error} />
        </ExperimentPanel>

        <ExperimentPanel
          title="展平：从矩阵到向量"
          eyebrow="SAME VALUES / NEW SHAPE"
        >
          <p className="panel-note">
            展平只改变排列形状，保留所有像素值；高亮位置与上面的选中像素对应。
          </p>
          <div
            className="flat-vector"
            role="group"
            aria-label="展平后的 64 个数值"
          >
            {Array.from({ length: 64 }, (_, index) => (
              <button
                key={index}
                type="button"
                className={`vector-cell${selected === index ? " is-selected" : ""}`}
                aria-label={`索引 ${index}，值 ${display(data?.flat[index])}`}
                aria-pressed={selected === index}
                onClick={() => {
                  setSelected(index);
                  onInteract();
                }}
              >
                <span className="vector-index">{index}</span>
                <span>{display(data?.flat[index], 2)}</span>
              </button>
            ))}
          </div>
          <div className="formula-row">
            <code className="formula">[8, 8] → [64]</code>
            <span className="axis-note">
              当前索引：{row} × 8 + {column} = {selected}（从 0 开始）
            </span>
          </div>
        </ExperimentPanel>
      </div>

      <aside className="control-panel" aria-label="数据实验控制">
        <div className="control-stack">
          <div>
            <div className="small-label">选择教学样本</div>
            <div className="segmented" role="group" aria-label="教学样本">
              {(["3", "7", "blank"] as const).map((value) => (
                <button
                  type="button"
                  key={value}
                  aria-pressed={sample === value}
                  onClick={() => chooseSample(value)}
                >
                  {value === "blank" ? "空白" : `数字 ${value}`}
                </button>
              ))}
            </div>
          </div>
          <Slider
            label={`选中像素 · 行 ${row + 1} / 列 ${column + 1}`}
            symbol="x"
            value={rawValue}
            min={0}
            max={255}
            step={1}
            onChange={(value) => updatePixel(selected, value)}
          />
          <label className="normalize-toggle">
            <input
              type="checkbox"
              checked={normalize}
              onChange={(event) => {
                setNormalize(event.target.checked);
                onInteract();
              }}
            />
            <span>归一化到 0–1</span>
          </label>
          <div>
            <div className="small-label">当前转换公式</div>
            <div className="formula">
              {normalize ? "x′ = x / 255" : "x′ = x"}
            </div>
            <p className="panel-note">
              {normalize
                ? `${rawValue} / 255 = ${display(convertedValue)}`
                : `${rawValue} → ${display(convertedValue)}`}
            </p>
          </div>
          <div className="insight">
            像素值是输入特征。改变一个像素，就改变了模型接收到的一个数字。
          </div>
          <p className="panel-note">
            本节只观察数据，不训练模型。矩阵、展平和统计量均由后端计算。
          </p>
        </div>
      </aside>
    </div>
  );
}
