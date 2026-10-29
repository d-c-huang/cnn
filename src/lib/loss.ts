/** Current-parameter preview with the same curve domain and sampling as /api/loss. */
export function squaredErrorPreview(prediction: number, target: number) {
  const error = prediction - target;
  const left = Math.min(target - 5, prediction);
  const right = Math.max(target + 5, prediction);
  const step = (right - left) / 120;
  const xs = Array.from({ length: 121 }, (_, i) => i === 120 ? right : left + i * step);
  xs.push(target, prediction);
  return {
    loss: error ** 2,
    gradient: 2 * error,
    curve: [...new Set(xs)].sort((a, b) => a - b).map(x => ({ x, y: (x - target) ** 2 })),
  };
}
