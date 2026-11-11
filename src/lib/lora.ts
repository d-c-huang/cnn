import type { TransformerMetric } from "./transformer";

export type LoRAConfig = {
  corpus: string; rank: number; alpha: number; context_length: number; batch_size: number; learning_rate: number; seed: number;
};
export type LoRAStatus = {
  model_id: string; revision: string; state: "idle" | "downloading" | "ready" | "failed";
  message: string; available: boolean; dependencies: Record<string, boolean>; corpus?: string; device: string;
};
export type LoRARun = {
  id: string; created_at: string; config: LoRAConfig; model_id: string; revision: string; weight_sha256: string;
  step: number; best_step: number; history: TransformerMetric[]; base_parameters: number; trainable_parameters: number;
  trainable_ratio: number; train_tokens: number; validation_tokens: number; training_context: number; frozen_verified: boolean;
};
export type LoRAGeneration = {
  run_id: string; model_id: string; revision: string; step: number; checkpoint: "latest" | "best"; prompt: string;
  baseline: { text: string; token_ids: number[]; finish_reason: string };
  adapted: { text: string; token_ids: number[]; finish_reason: string };
  context_tokens: number; frozen_verified: boolean;
};
export const loraDefaults: LoRAConfig = { corpus: "", rank: 4, alpha: 8, context_length: 32, batch_size: 2, learning_rate: 0.003, seed: 7 };

export async function loraApi<T>(path: string, payload?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/lora${path}`, {
    method: payload === undefined ? "GET" : "POST", headers: payload === undefined ? {} : { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(60000)]) : AbortSignal.timeout(60000),
  });
  const result = await response.json().catch(() => ({ detail: `计算服务返回 ${response.status}` }));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : Array.isArray(result.detail) ? result.detail.map((item: { msg?: string }) => item.msg ?? "参数无效").join("；") : `请求失败 ${response.status}`);
  return result as T;
}
