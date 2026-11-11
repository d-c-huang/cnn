import type { ChatMessage, SFTReply } from "./sft";
import type { LoRAStatus } from "./lora";
export type Preference = { prompt: string | ChatMessage[]; chosen: string; rejected: string };
export type DPODraft = { dataset_text: string; system_prompt: string; max_length: number; seed: number; rank: number; alpha: number; batch_size: number; learning_rate: number; beta: number; source_sft_id: string | null; source_checkpoint: "latest" | "best" };
export type DPOMetric = { step: number; train_loss: number; validation_loss: number; train_margin: number; validation_margin: number; train_accuracy: number; validation_accuracy: number };
export type DPOSource = { kind: "pretrained" | "sft"; step: number; run_id?: string; checkpoint?: string };
export type DPORun = { id: string; created_at: string; config: Omit<DPODraft, "dataset_text"> & { dataset: Preference[] }; step: number; best_step: number; history: DPOMetric[];
  base_parameters: number; trainable_parameters: number; frozen_verified: boolean; reference_verified: boolean; reference_digest: string; dataset_fingerprint: string;
  source: DPOSource; train_indices: number[]; validation_indices: number[]; template_version: string;
  pair_scores: { chosen_logp: number; rejected_logp: number; reference_chosen_logp: number; reference_rejected_logp: number; margin: number }[] };
type TokenView = { total_tokens: number; supervised_tokens: number; tokens: { id: number; text: string; label: number; supervised: boolean }[] };
export type DPOPreview = { valid: boolean; errors: string[]; train_indices: number[]; validation_indices: number[]; selected_index: number; chosen: TokenView; rejected: TokenView };
export type DPOStatus = LoRAStatus & { dataset: Preference[]; system_prompt: string; template_version: string };
export type DPOComparison = { run_id: string; step: number; checkpoint: string; source: DPOSource; template_version: string; before: Pick<SFTReply, "reply" | "token_ids" | "finish_reason" | "dropped_turns" | "context_tokens" | "used_messages">; after: Pick<SFTReply, "reply" | "token_ids" | "finish_reason" | "dropped_turns" | "context_tokens" | "used_messages"> };
export const dpoDefaults: DPODraft = { dataset_text: "", system_prompt: "You are Zhiwei. Answer briefly and clearly.", max_length: 128, seed: 7, rank: 8, alpha: 16, batch_size: 2, learning_rate: 0.0001, beta: 0.1, source_sft_id: null, source_checkpoint: "latest" };
export function preferenceQuestion(row: Preference): string { return typeof row.prompt === "string" ? row.prompt : row.prompt.at(-1)!.content; }
export function parsePreferences(raw: string): Preference[] {
  const text = raw.replace(/^\uFEFF/, "").trim();
  if (!text || text.length > 200000) throw new Error("请输入不超过 200,000 字符的 JSON / JSONL 偏好数据。");
  let rows: unknown;
  if (text.startsWith("[")) { try { rows = JSON.parse(text); } catch { throw new Error("JSON 数组无效，请检查括号与引号。"); } }
  else rows = text.split(/\r?\n/).flatMap((line, i) => { if (!line.trim()) return []; try { return [JSON.parse(line)]; } catch { throw new Error(`JSONL 第 ${i + 1} 行无效。`); } });
  if (!Array.isArray(rows) || !rows.length || rows.length > 64) throw new Error("数据集须为 1–64 条记录，训练至少需要 5 个不同首问。");
  rows.forEach((row, i) => {
    if (!row || typeof row !== "object" || Object.keys(row).sort().join(",") !== "chosen,prompt,rejected" || typeof row.chosen !== "string" || typeof row.rejected !== "string" || !row.chosen.trim() || !row.rejected.trim()) throw new Error(`第 ${i + 1} 条须含 prompt、非空 chosen 和 rejected。`);
    if (row.chosen.trim().replace(/\s+/g, " ") === row.rejected.trim().replace(/\s+/g, " ")) throw new Error(`第 ${i + 1} 条的两个回答不能相同。`);
    if (typeof row.prompt === "string" && row.prompt.trim()) return;
    if (Array.isArray(row.prompt) && row.prompt.length >= 1 && row.prompt.length <= 7 && row.prompt.length % 2 === 1 && row.prompt.every((m: ChatMessage, p: number) => m && Object.keys(m).sort().join(",") === "content,role" && m.role === (p % 2 ? "assistant" : "user") && typeof m.content === "string" && m.content.trim())) return;
    throw new Error(`第 ${i + 1} 条 prompt 须为非空问题，或 user/assistant 交替且以 user 结束的消息列表。`);
  });
  return rows as Preference[];
}
export function readDPODraft(): DPODraft | null {
  try {
    const value = JSON.parse(localStorage.getItem("zhiwei-dpo-draft-v1") || "null");
    if (!value || typeof value.dataset_text !== "string" || value.dataset_text.length > 200000) return null;
    return { ...dpoDefaults, dataset_text: value.dataset_text, system_prompt: typeof value.system_prompt === "string" ? value.system_prompt.slice(0, 256) : dpoDefaults.system_prompt,
      max_length: [64, 128, 256].includes(value.max_length) ? value.max_length : 128, rank: [2, 4, 8, 16].includes(value.rank) ? value.rank : 8,
      alpha: [4, 8, 16, 32].includes(value.alpha) ? value.alpha : 16, batch_size: [1, 2, 4].includes(value.batch_size) ? value.batch_size : 2,
      seed: Number.isInteger(value.seed) && value.seed >= 0 && value.seed <= 4294967295 ? value.seed : 7,
      learning_rate: typeof value.learning_rate === "number" && value.learning_rate >= 0.00001 && value.learning_rate <= 0.01 ? value.learning_rate : 0.0001,
      beta: typeof value.beta === "number" && value.beta >= 0.01 && value.beta <= 1 ? value.beta : 0.1,
      source_sft_id: typeof value.source_sft_id === "string" && /^[0-9a-f]{32}$/.test(value.source_sft_id) ? value.source_sft_id : null,
      source_checkpoint: value.source_checkpoint === "best" ? "best" : "latest" };
  } catch { return null; }
}
export async function dpoApi<T>(path: string, payload?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/dpo${path}`, { method: payload === undefined ? "GET" : "POST", headers: payload === undefined ? {} : { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload), signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(60000)]) : AbortSignal.timeout(60000) });
  const result = await response.json().catch(() => ({ detail: `服务返回 ${response.status}` }));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : Array.isArray(result.detail) ? result.detail.map((item: { msg?: string }) => item.msg ?? "参数无效").join("；") : "DPO 请求失败");
  return result as T;
}
