/** Choose a label color without altering the quantitative heatmap underneath it. */
export function overlayLabelColor(rgb: readonly number[], alpha: number, background: readonly number[] = [11, 16, 23]) {
  const linear = rgb.map((value, index) => {
    const channel = (value * alpha + background[index] * (1 - alpha)) / 255;
    return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  const luminance = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
  return luminance > 0.179 ? "var(--shadow-ink)" : "var(--data-white)";
}
