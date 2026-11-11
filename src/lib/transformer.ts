export type TransformerMetric = { step: number; train_loss: number; validation_loss: number; perplexity: number };
export type TransformerConfig = {
  corpus: string; context_length: number; d_model: number; heads: number; layers: number;
  batch_size: number; learning_rate: number; seed: number;
};
export type TransformerRun = {
  id: string; created_at: string; config: TransformerConfig; vocab: string[]; step: number;
  best_step: number; parameter_count: number; history: TransformerMetric[];
  train_tokens: number; validation_tokens: number; unknown_validation: number; training_context: number;
};
export type TransformerGeneration = {
  run_id: string; step: number; checkpoint: "latest" | "best"; prompt: string; continuation: string;
  token_ids: number[]; context_tokens: string[]; attention: number[][][]; attention_layer: number;
  distribution: { token: string; probability: number }[];
};

export const transformerDefaults: TransformerConfig = {
  corpus: "", context_length: 16, d_model: 32, heads: 2, layers: 2, batch_size: 16, learning_rate: 0.003, seed: 7,
};
export const displayLanguageToken = (token: string) => token === "\n" ? "↵" : token === " " ? "空格" : token;

export async function transformerApi<T>(path: string, payload?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/transformer${path}`, {
    method: payload === undefined ? "GET" : "POST",
    headers: payload === undefined ? {} : { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(45000)]) : AbortSignal.timeout(45000),
  });
  const result = await response.json().catch(() => ({ detail: `计算服务返回错误 ${response.status}` }));
  if (!response.ok) {
    const detail = result?.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map(item => item.msg).join("；") : `服务返回错误 ${response.status}`);
  }
  return result as T;
}
