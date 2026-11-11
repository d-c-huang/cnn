export function LossChart({ history, label = "交叉熵" }: { history: { step: number; train_loss: number; validation_loss: number }[]; label?: string }) {
  const maximum = Math.max(...history.flatMap(row => [row.train_loss, row.validation_loss]), 0.1);
  const points = (key: "train_loss" | "validation_loss") => history.filter((_, i) => i % Math.max(1, Math.floor(history.length / 200)) === 0 || i === history.length - 1)
    .map(row => `${40 + row.step / Math.max(1, history.at(-1)!.step) * 510},${180 - row[key] / maximum * 150}`).join(" ");
  return <figure className="language-chart"><svg viewBox="0 0 580 215" role="img" aria-label={`训练与验证${label}曲线，共 ${history.at(-1)!.step} 次更新；下方表格提供精确数值`}>
    <path d="M40 25V180H550" className="language-axis" />
    <text x="5" y="32">{maximum.toFixed(1)}</text><text x="15" y="184">0</text>
    <text x="40" y="205">0</text><text x="495" y="205">{history.at(-1)!.step} 步</text>
    <polyline points={points("train_loss")} className="language-train-line" />
    <polyline points={points("validation_loss")} className="language-validation-line" />
  </svg><figcaption><span>实线：训练{label}</span><span>虚线：验证{label}</span></figcaption></figure>;
}

