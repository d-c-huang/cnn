import { useEffect, useRef } from "react";
import "./mnist-visuals.css";

export default function DigitImage({
  pixels,
  label,
  size = 84,
}: {
  pixels: number[][];
  label: string;
  size?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rows = pixels.length || 28;
  const columns = pixels[0]?.length || 28;
  useEffect(() => {
    const context = ref.current?.getContext("2d");
    if (!context) return;
    const image = context.createImageData(columns, rows);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const value = Math.round(
          Math.max(0, Math.min(255, pixels[y]?.[x] ?? 0)),
        );
        const offset = (y * columns + x) * 4;
        image.data[offset] = value;
        image.data[offset + 1] = value;
        image.data[offset + 2] = value;
        image.data[offset + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [pixels, rows, columns]);
  return (
    <canvas
      ref={ref}
      className="mv-digit"
      width={columns}
      height={rows}
      style={{ width: size }}
      role="img"
      aria-label={label}
    >
      {label}
    </canvas>
  );
}
