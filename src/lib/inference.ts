export type InferenceModel = {
  id: string;
  architecture: "mlp" | "cnn";
  run_id: string;
  checkpoint_epoch: number;
  validation_loss: number;
  test_accuracy: number | null;
  source: "run" | "bundle";
  sha256: string;
};
export type ModelCatalog = {
  runtime: { available: boolean; torch_version: string | null; device: "cpu" };
  models: InferenceModel[];
  defaults: { mlp: string | null; cnn: string | null };
};
export type CanvasInput = {
  pixels: number[][];
  mode: "canvas" | "mnist";
  empty: boolean;
};
export type DigitExample = {
  index: number;
  label: number;
  pixels: number[][];
  source: "mnist-test";
};
export type ModelPrediction = {
  model_id: string;
  architecture: "mlp" | "cnn";
  run_id: string;
  checkpoint_epoch: number;
  predicted_label: number;
  probabilities: number[];
  top_k: { label: number; score: number }[];
  latency_ms: number;
};
export type PredictionResponse = {
  request_id: string;
  preprocessing: {
    mode: "canvas" | "mnist";
    original_shape: [number, number];
    bounding_box: [number, number, number, number] | null;
    shift: [number, number];
    processed_pixels: number[][];
    normalization: { mean: number; std: number };
    ink_pixels: number;
  };
  predictions: ModelPrediction[];
  total_latency_ms: number;
};
export async function inferenceRequest<T>(
  path: string,
  options: { body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(30000)])
    : AbortSignal.timeout(30000);
  const response = await fetch(`/api/inference${path}`, {
    method: options.body === undefined ? "GET" : "POST",
    headers:
      options.body === undefined
        ? undefined
        : { "Content-Type": "application/json" },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal,
  }).catch((cause) => {
    if (cause instanceof DOMException && cause.name === "TimeoutError")
      throw new Error("模型加载或推理超时，请稍后重试。");
    if (cause instanceof TypeError)
      throw new Error("无法连接推理服务，请确认本机 API 已启动。");
    throw cause;
  });
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* Preserve status below. */
    }
    if (typeof detail === "string") throw new Error(detail);
    throw new Error(
      response.status === 422
        ? "输入图像或模型选择无效，请检查后重试。"
        : `推理服务返回错误（${response.status}）。`,
    );
  }
  return response.json() as Promise<T>;
}
