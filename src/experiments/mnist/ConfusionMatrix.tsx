import { useState } from "react";
import type { Evaluation } from "../../lib/mnist";
import "./mnist-visuals.css";

type Cell = { truth: number; predicted: number };

export default function ConfusionMatrix({
  evaluation,
  selected,
  onSelect,
}: {
  evaluation: Evaluation;
  selected: Cell | null;
  onSelect: (cell: Cell | null) => void;
}) {
  const [normalized, setNormalized] = useState(false);
  const matrix = evaluation.confusion_matrix;
  const maximum = Math.max(1, ...matrix.flat());
  const selectedCount = selected
    ? (matrix[selected.truth]?.[selected.predicted] ?? 0)
    : 0;
  return (
    <div className="mv-confusion">
      <div className="mv-matrix-toolbar">
        <div
          className="segmented mv-matrix-mode"
          role="group"
          aria-label="混淆矩阵数值显示"
        >
          <button
            aria-pressed={!normalized}
            onClick={() => setNormalized(false)}
          >
            样本数量
          </button>
          <button aria-pressed={normalized} onClick={() => setNormalized(true)}>
            按行百分比
          </button>
        </div>
        {selected && (
          <button className="text-button" onClick={() => onSelect(null)}>
            清除筛选
          </button>
        )}
      </div>
      <p className="mv-matrix-axis">行：真实标签 y　↓　列：预测标签 ŷ　→</p>
      <div className="mv-matrix-scroll">
        <table className="mv-matrix-table">
          <caption>
            独立测试集混淆矩阵，行是真实标签，列是预测标签；点击单元格筛选下方错误样本。
          </caption>
          <thead>
            <tr>
              <th scope="col">
                真实 ↓<br />
                预测 →
              </th>
              {Array.from({ length: 10 }, (_, label) => (
                <th key={label} scope="col">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, truth) => {
              const total = row.reduce((sum, value) => sum + value, 0);
              return (
                <tr key={truth}>
                  <th scope="row">{truth}</th>
                  {row.map((count, predicted) => {
                    const fraction = total ? count / total : 0;
                    const strength = count
                      ? 0.08 +
                        0.58 *
                          Math.sqrt(normalized ? fraction : count / maximum)
                      : 0;
                    const active =
                      selected?.truth === truth &&
                      selected.predicted === predicted;
                    return (
                      <td key={predicted}>
                        <button
                          className={`${truth === predicted ? "mv-diagonal" : "mv-off-diagonal"} ${active ? "mv-cell-selected" : ""}`}
                          style={{
                            backgroundColor: strength
                              ? truth === predicted
                                ? `rgba(117, 153, 77, ${strength})`
                                : `rgba(200, 149, 108, ${strength})`
                              : undefined,
                          }}
                          aria-label={`真实 ${truth}，预测 ${predicted}，${count} 个样本，占真实 ${truth} 的 ${(fraction * 100).toFixed(1)}%，${truth === predicted ? "正确分类" : "错误分类"}`}
                          aria-pressed={active}
                          onClick={() =>
                            onSelect(active ? null : { truth, predicted })
                          }
                        >
                          {normalized
                            ? `${Math.round(fraction * 100)}%`
                            : count}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mv-matrix-caption">
        <span>
          <i className="mv-diagonal-key" />
          对角线：预测正确
        </span>
        <span>
          <i className="mv-error-key" />
          非对角线：预测错误
        </span>
        <span>颜色越深，{normalized ? "该行占比" : "样本数量"}越大</span>
      </div>
      <p className="mv-matrix-selection" role="status">
        {selected
          ? selected.truth === selected.predicted
            ? `真实 ${selected.truth} → 预测 ${selected.predicted}：${selectedCount} 个正确分类样本，不在错例列表中。`
            : `正在筛选：真实 ${selected.truth} → 预测 ${selected.predicted}，共 ${selectedCount} 个错误样本。`
          : "点击非对角线单元格，查看对应的错误样本。"}
      </p>
    </div>
  );
}
