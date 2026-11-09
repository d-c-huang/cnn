export type LossKind = "mse" | "mae" | "huber" | "bce";

export function lossValue(prediction: number, target: number, kind: LossKind = "mse", delta = 1) {
  const error = prediction - target;
  const absolute = Math.abs(error);
  if (kind === "bce") {
    // Stable BCE from logits; the derivative is with respect to z, not p.
    const probability = Math.exp(-Math.max(-prediction, 0) - Math.log1p(Math.exp(-Math.abs(prediction))));
    return { loss: Math.max(prediction, 0) - target * prediction + Math.log1p(Math.exp(-Math.abs(prediction))), gradient: probability - target, differentiable: true };
  }
  if (kind === "mae") return { loss: absolute, gradient: Math.sign(error), differentiable: error !== 0 };
  if (kind === "huber") return {
    loss: absolute <= delta ? 0.5 * error ** 2 : delta * (absolute - 0.5 * delta),
    gradient: absolute <= delta ? error : delta * Math.sign(error),
    differentiable: true,
  };
  return { loss: error ** 2, gradient: 2 * error, differentiable: true };
}

/** Current-parameter preview with the same curve domain and sampling as /api/loss. */
export function lossPreview(prediction: number, target: number, kind: LossKind = "mse", delta = 1) {
  const left = kind === "bce" ? Math.min(-8, prediction) : Math.min(target - 5, prediction);
  const right = kind === "bce" ? Math.max(8, prediction) : Math.max(target + 5, prediction);
  const step = (right - left) / 120;
  const xs = Array.from({ length: 121 }, (_, i) => i === 120 ? right : left + i * step);
  xs.push(kind === "bce" ? 0 : target, prediction);
  if (kind === "huber") xs.push(...[target - delta, target + delta].filter(x => x >= left && x <= right));
  return {
    ...lossValue(prediction, target, kind, delta),
    curve: [...new Set(xs)].sort((a, b) => a - b).map(x => ({ x, y: lossValue(x, target, kind, delta).loss })),
  };
}

export function squaredErrorPreview(prediction: number, target: number) {
  return lossPreview(prediction, target);
}
