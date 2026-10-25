import { ArrowLeft, ArrowRight } from "lucide-react";
import type { ErrorPage, ErrorSample } from "../../lib/mnist";
import { ErrorMessage } from "../../components/ui";
import DigitImage from "./DigitImage";
import "./mnist-visuals.css";

export default function ErrorSamples({
  page,
  pending,
  error,
  onPage,
  onSelect,
}: {
  page: ErrorPage | null;
  pending: boolean;
  error: string | null;
  onPage: (offset: number) => void;
  onSelect: (sample: ErrorSample) => void;
}) {
  if (error) return <ErrorMessage error={error} />;
  if (!page || pending)
    return (
      <div className="mv-error-loading" role="status">
        {pending
          ? "正在读取独立测试集错误样本…"
          : "选择一个已完成的训练任务，查看独立测试集错例。"}
      </div>
    );
  if (!page.total)
    return (
      <div className="mv-error-empty">
        <span>没有符合筛选条件的错例</span>
        <p>
          可以清除混淆矩阵筛选，查看全部错误样本。正确分类样本不会出现在这里。
        </p>
      </div>
    );
  if (!page.items.length)
    return (
      <div className="mv-error-empty">
        <span>当前页没有错误样本</span>
        <p>该筛选下共有 {page.total} 个错例，可以回到第一页重新查看。</p>
        <button className="secondary-button" onClick={() => onPage(0)}>
          回到第一页
        </button>
      </div>
    );
  return (
    <div className="mv-errors" aria-busy={pending}>
      <div className="mv-errors-heading">
        <span>
          第 {page.offset + 1}–
          {Math.min(page.offset + page.items.length, page.total)} 个 / 共{" "}
          {page.total} 个错例
        </span>
        <span>点击图片，查看十个类别的输出分数</span>
      </div>
      <div className="mv-error-grid">
        {page.items.map((sample) => (
          <button
            className="mv-error-card"
            key={sample.index}
            onClick={() => onSelect(sample)}
            aria-label={`测试样本 ${sample.index}，真实标签 ${sample.label}，预测标签 ${sample.prediction}，模型分数 ${(sample.confidence * 100).toFixed(1)}%，查看详情`}
          >
            <DigitImage
              pixels={sample.pixels}
              label={`MNIST 测试样本 ${sample.index}，真实数字 ${sample.label}`}
              size={78}
            />
            <span className="mv-error-index">测试集 #{sample.index}</span>
            <span className="mv-error-labels">
              真实 <strong>{sample.label}</strong>
              <span>→</span>预测 <strong>{sample.prediction}</strong>
            </span>
            <span className="mv-error-confidence">
              模型分数 {(sample.confidence * 100).toFixed(1)}%
            </span>
          </button>
        ))}
      </div>
      <div className="mv-error-pagination">
        <button
          className="secondary-button"
          disabled={pending || page.offset <= 0}
          onClick={() => onPage(Math.max(0, page.offset - page.limit))}
        >
          <ArrowLeft size={14} />
          上一页
        </button>
        <span>
          第 {Math.floor(page.offset / page.limit) + 1} /{" "}
          {Math.max(1, Math.ceil(page.total / page.limit))} 页
        </span>
        <button
          className="secondary-button"
          disabled={pending || page.offset + page.limit >= page.total}
          onClick={() => onPage(page.offset + page.limit)}
        >
          下一页
          <ArrowRight size={14} />
        </button>
      </div>
      <p className="mv-note">
        图像来自该模型的独立测试集预测。模型分数为 softmax
        输出，不能保证一次预测的可靠性。
      </p>
    </div>
  );
}
