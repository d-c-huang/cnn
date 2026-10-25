import type { ModelPrediction } from "../../lib/inference";
import "./inference-visuals.css";

export default function PredictionBars({
  prediction,
}: {
  prediction: ModelPrediction;
}) {
  const scores = Array.from({ length: 10 }, (_, label) => ({
    label,
    score: Math.max(0, Math.min(1, prediction.probabilities[label] ?? 0)),
  }));
  const topScore = scores[prediction.predicted_label]?.score ?? 0;
  return (
    <section
      className="iv-prediction"
      aria-label={`${prediction.architecture.toUpperCase()} 预测结果`}
    >
      <div className="iv-prediction-heading">
        <div>
          <span className="iv-architecture">
            {prediction.architecture.toUpperCase()}
          </span>
          <span className="iv-checkpoint">
            验证集最佳权重 · Epoch {prediction.checkpoint_epoch}
          </span>
        </div>
        <span className="iv-forward-latency">
          前向 {prediction.latency_ms.toFixed(2)} ms
        </span>
      </div>
      <div className="iv-predicted-digit" aria-live="polite">
        <strong>{prediction.predicted_label}</strong>
        <div>
          <span>模型预测数字</span>
          <b>{(topScore * 100).toFixed(2)}%</b>
          <small>该类别 softmax 分数</small>
        </div>
      </div>
      <div className="iv-class-scores" aria-label="十个数字类别的模型分数">
        {scores.map(({ label, score }) => (
          <div
            key={label}
            className={`iv-score-row ${label === prediction.predicted_label ? "iv-is-predicted" : ""}`}
          >
            <span className="iv-score-label">{label}</span>
            <div
              className="iv-score-track"
              role="meter"
              aria-label={`数字 ${label} 的模型分数`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={score * 100}
              aria-valuetext={`${(score * 100).toFixed(2)}%${label === prediction.predicted_label ? "，预测类别" : ""}`}
            >
              <i style={{ width: `${score * 100}%` }} />
            </div>
            <strong>{(score * 100).toFixed(2)}%</strong>
            {label === prediction.predicted_label && (
              <span className="iv-score-winner">预测</span>
            )}
          </div>
        ))}
      </div>
      <div className="iv-top-three">
        <span>前三类</span>
        {prediction.top_k.slice(0, 3).map((item, i) => (
          <span key={item.label}>
            <b>{i + 1}.</b> {item.label}
            <small>{(item.score * 100).toFixed(2)}%</small>
          </span>
        ))}
      </div>
      <p className="iv-prediction-note">
        分数是模型的 softmax
        输出，不代表经过校准的可靠性保证。图中各类别使用相同的 0–100% 比例。
      </p>
    </section>
  );
}
