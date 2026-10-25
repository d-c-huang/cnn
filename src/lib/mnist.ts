export type MnistStatus = {
  runtime: {
    available: boolean;
    torch_version: string | null;
    torchvision_version: string | null;
    device: "cpu";
    threads: number;
  };
  dataset: {
    status: "missing" | "preparing" | "ready" | "failed";
    message: string;
    error: string | null;
    official_train: number;
    official_test: number;
  };
  active_run_id: string | null;
};
export type SplitConfig = {
  seed: number;
  validation_ratio: number;
  train_limit: number | null;
};
export type RunConfig = SplitConfig & {
  model: "mlp" | "cnn";
  epochs: number;
  batch_size: number;
  learning_rate: number;
};
export type DigitSample = {
  split: "train" | "validation" | "test";
  index: number;
  label: number;
  pixels: number[][];
};
export type DataSplit = SplitConfig & {
  train_count: number;
  validation_count: number;
  test_count: number;
  effective_train_count: number;
  train_class_counts: number[];
  validation_class_counts: number[];
  test_class_counts: number[];
  samples: DigitSample[];
};
export type EpochPoint = {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  validation_loss: number;
  validation_accuracy: number;
  seconds: number;
};
export type Evaluation = {
  loss: number;
  accuracy: number;
  total: number;
  confusion_matrix: number[][];
  class_metrics: {
    label: number;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }[];
  error_count: number;
  checkpoint_epoch: number;
};
export type Run = {
  id: string;
  created_at: string;
  status:
    | "queued"
    | "preparing"
    | "training"
    | "validating"
    | "evaluating"
    | "completed"
    | "cancelling"
    | "cancelled"
    | "failed"
    | "interrupted";
  config: RunConfig;
  epoch: number;
  batch: number;
  batches_per_epoch: number;
  processed_samples: number;
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  history: EpochPoint[];
  best_epoch: number | null;
  best_validation_loss: number | null;
  parameter_count: number;
  device: "cpu";
  message: string;
  error: string | null;
  evaluation: Evaluation | null;
  elapsed_seconds: number;
  checkpoint_available: boolean;
};
export type ErrorSample = {
  index: number;
  label: number;
  prediction: number;
  confidence: number;
  pixels: number[][];
  probabilities: number[];
};
export type ErrorPage = {
  total: number;
  offset: number;
  limit: number;
  items: ErrorSample[];
};
export const activeStatuses = new Set<Run["status"]>([
  "queued",
  "preparing",
  "training",
  "validating",
  "evaluating",
  "cancelling",
]);
export const statusLabels: Record<Run["status"], string> = {
  queued: "等待开始",
  preparing: "准备数据",
  training: "训练中",
  validating: "验证中",
  evaluating: "独立测试评估中",
  completed: "已完成",
  cancelling: "正在取消",
  cancelled: "已取消",
  failed: "失败",
  interrupted: "服务中断",
};
export const defaultRunConfig: RunConfig = {
  model: "mlp",
  seed: 7,
  epochs: 3,
  batch_size: 128,
  learning_rate: 0.001,
  validation_ratio: 0.1,
  train_limit: null,
};
export const splitStorageKey = "zhiwei-mnist-config-v1";
export const selectedRunStorageKey = "zhiwei-mnist-selected-run-v1";
export function loadMnistConfig(): RunConfig {
  try {
    const raw = JSON.parse(localStorage.getItem(splitStorageKey) || "{}");
    return {
      model: raw.model === "cnn" ? "cnn" : "mlp",
      seed:
        Number.isInteger(raw.seed) && raw.seed >= 0 && raw.seed <= 9999
          ? raw.seed
          : 7,
      epochs:
        Number.isInteger(raw.epochs) && raw.epochs >= 1 && raw.epochs <= 20
          ? raw.epochs
          : 3,
      batch_size: [32, 64, 128, 256].includes(raw.batch_size)
        ? raw.batch_size
        : 128,
      learning_rate:
        typeof raw.learning_rate === "number" &&
        raw.learning_rate >= 0.00001 &&
        raw.learning_rate <= 0.1
          ? raw.learning_rate
          : 0.001,
      validation_ratio: raw.validation_ratio === 0.2 ? 0.2 : 0.1,
      train_limit: [6000, 12000].includes(raw.train_limit)
        ? raw.train_limit
        : null,
    };
  } catch {
    return { ...defaultRunConfig };
  }
}
export function storeMnistConfig(config: RunConfig) {
  try {
    localStorage.setItem(splitStorageKey, JSON.stringify(config));
  } catch {
    /* Server-side run history remains available. */
  }
}
export function selectedRunId(): string | null {
  try {
    return localStorage.getItem(selectedRunStorageKey);
  } catch {
    return null;
  }
}
export function selectRunId(id: string) {
  try {
    localStorage.setItem(selectedRunStorageKey, id);
  } catch {
    /* Session can still select a run. */
  }
}

export async function mnistRequest<T>(
  path: string,
  options: { body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(10000)])
    : AbortSignal.timeout(10000);
  const response = await fetch(`/api/mnist${path}`, {
    method: options.body === undefined ? "GET" : "POST",
    headers:
      options.body === undefined
        ? undefined
        : { "Content-Type": "application/json" },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal,
  }).catch((cause) => {
    if (cause instanceof DOMException && cause.name === "TimeoutError")
      throw new Error("训练服务响应超时，请稍后重试。后台任务会继续运行。");
    if (cause instanceof TypeError)
      throw new Error("无法连接训练服务，请确认 npm run dev 已启动。");
    throw cause;
  });
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* HTTP status below remains useful. */
    }
    if (typeof detail === "string") throw new Error(detail);
    if (response.status === 422)
      throw new Error("训练参数无效，请检查数据规模、轮数与学习率。");
    throw new Error(`训练服务返回错误（${response.status}），请稍后重试。`);
  }
  return response.json() as Promise<T>;
}
