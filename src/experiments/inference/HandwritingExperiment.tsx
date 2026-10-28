import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, Download, RefreshCw, ScanLine } from "lucide-react";
import { ErrorMessage, ExperimentPanel, Result } from "../../components/ui";
import { format } from "../../lib/api";
import { clearPreferredInferenceRun, inferenceRequest, preferredInferenceRun } from "../../lib/inference";
import { modelLabel } from "../../lib/mnist";
import type { ModelType } from "../../lib/mnist";
import type {
  CanvasInput,
  DigitExample,
  ModelCatalog,
  PredictionResponse,
} from "../../lib/inference";
import { downloadText } from "../../lib/useMnist";
import DigitImage from "../mnist/DigitImage";
import DigitCanvas from "./DigitCanvas";
import PredictionBars from "./PredictionBars";
import InputAttribution from "./InputAttribution";
import "./inference.css";
import "../custom-models.css";

type Architecture = ModelType;
const architectures = ["mlp", "cnn", "custom"] as const;
export default function HandwritingExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [catalogPending, setCatalogPending] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [chosen, setChosen] = useState<Record<Architecture, string | null>>({
    mlp: null,
    cnn: null,
    custom: null,
  });
  const [enabled, setEnabled] = useState<Record<Architecture, boolean>>({
    mlp: true,
    cnn: true,
    custom: false,
  });
  const [input, setInput] = useState<CanvasInput | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [exampleLabel, setExampleLabel] = useState(7);
  const [example, setExample] = useState<DigitExample | null>(null);
  const [examplePending, setExamplePending] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explain, setExplain] = useState(false);
  const revision = useRef(0);
  const request = useRef<AbortController | null>(null);
  const exampleRequest = useRef<AbortController | null>(null);
  const catalogRequest = useRef<AbortController | null>(null);
  const catalogVersion = useRef(0);
  const desiredRun = useRef(preferredInferenceRun());

  const invalidate = useCallback(() => {
    revision.current++;
    request.current?.abort();
    request.current = null;
    exampleRequest.current?.abort();
    exampleRequest.current = null;
    setPrediction(null);
    setPending(false);
    setExamplePending(false);
    setError(null);
  }, []);
  const onDirty = useCallback(() => {
    invalidate();
    setDrawing(true);
  }, [invalidate]);
  const onInput = useCallback(
    (next: CanvasInput) => {
      invalidate();
      setInput(next);
      setDrawing(false);
    },
    [invalidate],
  );

  const loadModels = useCallback(async () => {
    catalogRequest.current?.abort();
    const token = ++catalogVersion.current;
    const controller = new AbortController();
    catalogRequest.current = controller;
    setCatalogPending(true);
    setCatalogError(null);
    try {
      const result = await inferenceRequest<ModelCatalog>("/models", {
        signal: controller.signal,
      });
      if (token !== catalogVersion.current) return;
      setCatalog(result);
      const desired = result.models.find(model => model.run_id === desiredRun.current);
      setChosen((previous) => ({
        mlp: result.models.some(
          (model) => model.id === previous.mlp && model.architecture === "mlp",
        )
          ? previous.mlp
          : result.defaults.mlp,
        cnn: result.models.some(
          (model) => model.id === previous.cnn && model.architecture === "cnn",
        )
          ? previous.cnn
          : result.defaults.cnn,
        custom: result.models.some(model => model.id === previous.custom && model.architecture === "custom") ? previous.custom : result.models.find(model => model.architecture === "custom")?.id ?? null,
        ...(desired ? { [desired.architecture]: desired.id } : {}),
      }));
      if (desired) {
        setEnabled({ mlp: desired.architecture === "mlp", cnn: desired.architecture === "cnn", custom: desired.architecture === "custom" });
        desiredRun.current = null;
        clearPreferredInferenceRun();
      }
    } catch (cause) {
      if (token === catalogVersion.current && !controller.signal.aborted)
        setCatalogError(
          cause instanceof Error ? cause.message : "无法读取模型列表。",
        );
    } finally {
      if (token === catalogVersion.current) {
        setCatalogPending(false);
        catalogRequest.current = null;
      }
    }
  }, []);
  useEffect(() => {
    void loadModels();
    return () => {
      catalogVersion.current++;
      revision.current++;
      catalogRequest.current?.abort();
      request.current?.abort();
      exampleRequest.current?.abort();
    };
  }, [loadModels]);

  async function loadExample() {
    invalidate();
    setDrawing(false);
    const token = revision.current;
    const controller = new AbortController();
    exampleRequest.current = controller;
    setExamplePending(true);
    try {
      const result = await inferenceRequest<DigitExample>(
        `/examples?label=${exampleLabel}`,
        { signal: controller.signal },
      );
      if (token === revision.current && !controller.signal.aborted) {
        setExample(result);
        onInteract();
      }
    } catch (cause) {
      if (token === revision.current && !controller.signal.aborted)
        setError(
          cause instanceof Error
            ? cause.message
            : "示例加载失败，请直接在画布中书写。",
        );
    } finally {
      if (token === revision.current) setExamplePending(false);
      if (exampleRequest.current === controller) exampleRequest.current = null;
    }
  }

  const modelIds = architectures.flatMap((architecture) =>
    enabled[architecture] && chosen[architecture]
      ? [chosen[architecture]!]
      : [],
  );
  const canPredict =
    !!input &&
    !input.empty &&
    !drawing &&
    !pending &&
    !examplePending &&
    !catalogPending &&
    !!catalog?.runtime.available &&
    modelIds.length > 0 && modelIds.length <= 2;
  async function predict() {
    if (!canPredict || !input) return;
    const token = ++revision.current;
    const controller = new AbortController();
    request.current = controller;
    setPending(true);
    setPrediction(null);
    setError(null);
    try {
      const result = await inferenceRequest<PredictionResponse>("/predict", {
        body: { model_ids: modelIds, pixels: input.pixels, mode: input.mode, explain },
        signal: controller.signal,
      });
      if (token === revision.current && !controller.signal.aborted) {
        setPrediction(result);
        onInteract();
      }
    } catch (cause) {
      if (token === revision.current && !controller.signal.aborted)
        setError(cause instanceof Error ? cause.message : "识别失败，请重试。");
    } finally {
      if (token === revision.current) setPending(false);
      if (request.current === controller) request.current = null;
    }
  }

  return (
    <div className="if-experiment">
      <ExperimentPanel
        title="选择真正训练过的最佳权重"
        eyebrow="BEST CHECKPOINTS · NO RETRAINING"
        action={
          <button
            className="text-button"
            disabled={catalogPending}
            onClick={() => {
              invalidate();
              void loadModels();
              onInteract();
            }}
          >
            <RefreshCw size={14} />
            刷新模型
          </button>
        }
      >
        <div className="if-model-grid">
          {architectures.map((architecture) => {
            const models =
              catalog?.models.filter(
                (model) => model.architecture === architecture,
              ) || [];
            const descriptor = models.find(
              (model) => model.id === chosen[architecture],
            );
            return (
              <div
                className={`if-model ${descriptor && enabled[architecture] ? "selected" : ""}`}
                key={architecture}
              >
                <button
                  className="if-model-toggle"
                  aria-pressed={!!descriptor && enabled[architecture]}
                  disabled={!descriptor || !enabled[architecture] && modelIds.length >= 2}
                  onClick={() => {
                    invalidate();
                    setEnabled((previous) => ({
                      ...previous,
                      [architecture]: !previous[architecture],
                    }));
                    onInteract();
                  }}
                >
                  <span>
                    <strong>{modelLabel(architecture)}</strong>
                    <small>
                      {architecture === "custom" ? "你设计并训练的网络" : architecture === "mlp" ? "全连接网络" : "卷积神经网络"}
                    </small>
                  </span>
                  <span className="if-check">
                    {descriptor && enabled[architecture] ? "已选择" : "未选择"}
                  </span>
                </button>
                {descriptor ? (
                  <>
                    <p>
                      最佳权重 Epoch {descriptor.checkpoint_epoch}{" "}
                      <span>
                        ·{" "}
                        {descriptor.source === "run"
                          ? architecture === "custom" ? "自定义结构训练记录" : "训练记录"
                          : "第三期导出模型包"}
                      </span>
                    </p>
                    {descriptor.structure && <p>{descriptor.display_name ?? `自定义网络 · ${descriptor.structure.length} 层`} · 结构来自该次训练快照</p>}
                    <p>
                      独立测试参考{" "}
                      {descriptor.test_accuracy === null
                        ? "未提供"
                        : `${(descriptor.test_accuracy * 100).toFixed(2)}%`}{" "}
                      <span>
                        · 验证损失 {format(descriptor.validation_loss, 5)}
                      </span>
                    </p>
                    <code>
                      run {descriptor.run_id.slice(0, 8)} · SHA{" "}
                      {descriptor.sha256.slice(0, 12)}
                    </code>
                    {models.length > 1 && (
                      <label>
                        选择权重版本
                        <select
                          value={descriptor.id}
                          onChange={(e) => {
                            invalidate();
                            setChosen((previous) => ({
                              ...previous,
                              [architecture]: e.target.value,
                            }));
                            onInteract();
                          }}
                        >
                          {models.map((model) => (
                            <option key={model.id} value={model.id}>
                              {model.id.slice(0, 8)} · Epoch{" "}
                              {model.checkpoint_epoch}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                  </>
                ) : (
                  <p>
                    {catalogPending
                      ? "正在查找已训练权重…"
                      : architecture === "custom" ? "还没有已完成的自定义训练。到搭建器训练并保存最佳权重。" : "没有可用权重，请先完成训练或导出模型包。"}
                  </p>
                )}
              </div>
            );
          })}
        </div>
        <p className="panel-note">
          一次可选择最多两个模型对比。若已有两个模型被选中，先取消其中一个，再选择自定义网络。所有模型使用验证集最佳权重；测试准确率不代表当前笔迹的正确概率。
        </p>
        <ErrorMessage
          error={
            catalogError ||
            (catalog && !catalog.runtime.available
              ? "PyTorch 运行时不可用。请安装 requirements.txt 中的依赖并重新启动 API。"
              : null)
          }
        />
        {catalog && !catalog.models.length && (
          <a className="secondary-button" href="#mnist-train">
            去第三期训练模型 <ArrowRight size={14} />
          </a>
        )}
      </ExperimentPanel>

      <div className="if-workbench">
        <ExperimentPanel
          title="写下一个数字，交给模型识别"
          eyebrow="YOUR HANDWRITING"
        >
          <DigitCanvas example={example} onChange={onInput} onDirty={onDirty} />
          <div className="if-example-loader">
            <label htmlFor="inference-example-label">
              真实测试样本
              <select
                id="inference-example-label"
                value={exampleLabel}
                onChange={(e) => setExampleLabel(Number(e.target.value))}
              >
                {Array.from({ length: 10 }, (_, label) => (
                  <option key={label} value={label}>
                    数字 {label}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="secondary-button"
              disabled={examplePending}
              onClick={() => void loadExample()}
            >
              {examplePending ? "加载中…" : "载入 MNIST 示例"}
            </button>
          </div>
          <div className="if-identify-row">
            <label className="if-explain-option"><input type="checkbox" checked={explain} onChange={event => { invalidate(); setExplain(event.target.checked); onInteract(); }} /><span>计算输入敏感度热力图<small>识别时计算预测类别对输入的真实梯度；改变此选项后需要重新识别。</small></span></label>
            <button
              className="primary-button"
              disabled={!canPredict}
              onClick={() => void predict()}
            >
              <ScanLine size={17} />
              {pending
                ? "加载权重并识别中…"
                : modelIds.length > 1
                  ? "识别并对比模型"
                  : "识别数字"}
            </button>
            <span role="status">
              {drawing
                ? "正在书写，松开后可识别"
                : input?.empty
                  ? "先写一个 0–9 的数字"
                  : !modelIds.length
                    ? "至少选择一个已训练模型"
                    : pending
                      ? "首次请求可能需要加载 PyTorch 与权重"
                      : input?.mode === "mnist" && example
                        ? `官方测试样本 #${example.index} · 真实标签 ${example.label}`
                        : "手写画布 · 输入已准备"}
            </span>
          </div>
          <ErrorMessage error={error} />
        </ExperimentPanel>
        <ExperimentPanel
          title="看看模型实际收到的输入"
          eyebrow="ONE PREPROCESSING PIPELINE"
        >
          {prediction ? (
            <>
              <div className="if-processed-image">
                <DigitImage
                  pixels={prediction.preprocessing.processed_pixels}
                  size={140}
                  label="后端实际用于推理的 28×28 灰度图"
                />
                <div>
                  <span className="small-label">同一份 28 × 28 输入</span>
                  <strong>
                    {prediction.preprocessing.mode === "mnist"
                      ? "保持官方样本原像素"
                      : "裁剪 → 缩放 → 质心居中"}
                  </strong>
                  <p>
                    原始尺寸{" "}
                    {prediction.preprocessing.original_shape.join(" × ")}
                    <br />
                    居中平移 Δx {prediction.preprocessing.shift[0]} · Δy{" "}
                    {prediction.preprocessing.shift[1]}
                  </p>
                </div>
              </div>
              <div className="formula">
                x标准 = (pixel / 255 −{" "}
                {prediction.preprocessing.normalization.mean}) /{" "}
                {prediction.preprocessing.normalization.std}
              </div>
              <div className="result-grid">
                <Result
                  label="本次请求总用时"
                  value={`${format(prediction.total_latency_ms, 2)} ms`}
                  hint="含预处理与首次模型加载"
                />
                <Result
                  label="参与比较的模型"
                  value={String(prediction.predictions.length)}
                  hint="同一输入，各自真实前向计算"
                />
              </div>
            </>
          ) : (
            <div className="if-input-placeholder">
              <span>28 × 28</span>
              <p>识别后，这里会显示后端处理后的真实图像。</p>
            </div>
          )}
          <p className="panel-note">
            手绘笔迹等比例缩放到最长边 20 像素，置于 28×28 黑底并居中；官方
            MNIST 示例保留原始 28×28
            像素。两种输入之后都使用权重中保存的标准化参数。
          </p>
        </ExperimentPanel>
      </div>

      <ExperimentPanel
        title="每个数字的分数，都来自模型输出"
        eyebrow="PREDICTIONS & MODEL COMPARISON"
        action={
          prediction ? (
            <button
              className="text-button"
              onClick={() =>
                downloadText(
                  `zhiwei-prediction-${prediction.request_id.slice(0, 8)}.json`,
                  JSON.stringify(prediction, null, 2),
                )
              }
            >
              <Download size={14} />
              导出预测 JSON
            </button>
          ) : undefined
        }
      >
        {prediction ? (
          <>
            <div className="if-prediction-grid">
              {prediction.predictions.map((item) => (
                <div key={item.model_id}>
                  <PredictionBars prediction={item} />
                  {item.attribution && <InputAttribution key={`${prediction.request_id}-${item.model_id}`} attribution={item.attribution} pixels={prediction.preprocessing.processed_pixels} />}
                  {input?.mode === "mnist" && example && (
                    <p
                      className={`if-example-check ${item.predicted_label === example.label ? "correct" : ""}`}
                    >
                      示例真实标签 {example.label} ·{" "}
                      {item.predicted_label === example.label
                        ? "本次预测正确"
                        : "本次预测有误"}
                    </p>
                  )}
                </div>
              ))}
            </div>
            <p className="panel-note">
              编辑画布或切换模型后，旧结果会立即清除；每份结果对应当前输入与明确的最佳权重版本。
            </p>
          </>
        ) : (
          <div className="if-results-empty">
            <ScanLine size={26} />
            <p>
              {pending
                ? "正在运行真实模型，等待十个类别的输出…"
                : "点击识别，查看所选模型的真实预测。"}
            </p>
          </div>
        )}
      </ExperimentPanel>
    </div>
  );
}
