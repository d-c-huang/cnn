import type { TransformerMetric } from "./transformer";
import type { LoRAStatus } from "./lora";

export type ChatMessage = { role: "user" | "assistant"; content: string };
export type SFTRecord = { question: string; answer: string } | { messages: ChatMessage[] };
export type SFTSettings = { system_prompt: string; max_length: number; seed: number; rank: number; alpha: number; batch_size: number; learning_rate: number };
export type SFTDraft = SFTSettings & { dataset_text: string };
export type SFTConfig = SFTSettings & { dataset: SFTRecord[] };
export type SFTStatus = LoRAStatus & { dataset: SFTRecord[]; system_prompt: string; template_version: string };
export type SFTPreview = {
  valid: boolean; errors: string[]; dataset_fingerprint: string; template_version: string; train_indices: number[]; validation_indices: number[];
  records: { index: number; question: string; tokens: number; supervised_tokens: number }[];
  selected_index: number; total_tokens: number; rendered_text: string;
  tokens: { id: number; text: string; label: number; supervised: boolean }[];
};
export type SFTRun = {
  id: string; created_at: string; config: SFTConfig; model_id: string; revision: string; weight_sha256: string; template_version: string;
  step: number; best_step: number; history: TransformerMetric[]; base_parameters: number; trainable_parameters: number; trainable_ratio: number;
  frozen_verified: boolean; dataset_fingerprint: string; train_indices: number[]; validation_indices: number[];
  train_supervised_tokens: number; validation_supervised_tokens: number;
};
export type SFTReply = { run_id: string; model_id: string; revision: string; template_version: string; step: number; checkpoint: "latest" | "best";
  reply: string; token_ids: number[]; finish_reason: string; dropped_turns: number; context_tokens: number; used_messages: ChatMessage[] };

export const sftDefaults: SFTDraft = { dataset_text: "", system_prompt: "You are Zhiwei. Answer briefly and clearly.", max_length: 128,
  rank: 8, alpha: 16, batch_size: 2, learning_rate: 0.003, seed: 7 };

export function parseSFTDataset(raw: string): SFTRecord[] {
  const text = raw.replace(/^\uFEFF/, "").trim();
  if (!text) throw new Error("请先输入或导入问答数据集。");
  if (text.length > 200000) throw new Error("数据文件不能超过 200,000 个字符。");
  let records: unknown;
  if (text.startsWith("[")) {
    try { records = JSON.parse(text); } catch { throw new Error("JSON 数组格式无效，请检查括号与引号。"); }
  } else {
    records = text.split(/\r?\n/).flatMap((line, index) => {
      if (!line.trim()) return [];
      try { return [JSON.parse(line)]; } catch { throw new Error(`JSONL 第 ${index + 1} 行不是有效 JSON。`); }
    });
  }
  if (!Array.isArray(records) || !records.length || records.length > 64) throw new Error("数据集应为 1–64 条记录；训练至少需要 5 条不同首问。");
  for (const [index, record] of records.entries()) {
    if (!record || typeof record !== "object" || Array.isArray(record)) throw new Error(`第 ${index + 1} 条应是问答或 messages 对象。`);
    const item = record as Record<string, unknown>;
    const keys = Object.keys(item).sort().join(",");
    if (keys === "answer,question" && typeof item.question === "string" && typeof item.answer === "string") continue;
    if (keys === "messages" && Array.isArray(item.messages) && item.messages.length >= 2 && item.messages.length <= 8 && item.messages.every((message, position) =>
      message && typeof message === "object" && Object.keys(message).sort().join(",") === "content,role" &&
      message.role === (position % 2 ? "assistant" : "user") && typeof message.content === "string") && item.messages.length % 2 === 0) continue;
    throw new Error(`第 ${index + 1} 条需包含 question/answer，或以 user 开始、assistant 结束的 messages。`);
  }
  return records as SFTRecord[];
}

export function recordQuestion(record: SFTRecord): string {
  return "question" in record ? record.question : record.messages.filter(message => message.role === "user").at(-1)?.content ?? "";
}

export function readSFTDraft(): SFTDraft | null {
  try {
    const value = JSON.parse(localStorage.getItem("zhiwei-sft-draft-v1") || "null");
    if (!value || typeof value.dataset_text !== "string" || value.dataset_text.length > 200000) return null;
    return { ...sftDefaults, dataset_text: value.dataset_text,
      system_prompt: typeof value.system_prompt === "string" ? value.system_prompt.slice(0, 256) : sftDefaults.system_prompt,
      max_length: [64, 128, 256].includes(value.max_length) ? value.max_length : 128,
      rank: [2, 4, 8, 16].includes(value.rank) ? value.rank : 8, alpha: [4, 8, 16, 32].includes(value.alpha) ? value.alpha : 16,
      batch_size: [1, 2, 4].includes(value.batch_size) ? value.batch_size : 2,
      seed: Number.isInteger(value.seed) && value.seed >= 0 && value.seed <= 4294967295 ? value.seed : 7,
      learning_rate: typeof value.learning_rate === "number" && value.learning_rate >= 0.0001 && value.learning_rate <= 0.01 ? value.learning_rate : 0.003 };
  } catch { return null; }
}

export async function sftApi<T>(path: string, payload?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/sft${path}`, { method: payload === undefined ? "GET" : "POST",
    headers: payload === undefined ? {} : { "Content-Type": "application/json" }, body: payload === undefined ? undefined : JSON.stringify(payload),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(60000)]) : AbortSignal.timeout(60000) });
  const result = await response.json().catch(() => ({ detail: `计算服务返回 ${response.status}` }));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : Array.isArray(result.detail) ? result.detail.map((item: { msg?: string }) => item.msg ?? "参数无效").join("；") : `请求失败 ${response.status}`);
  return result as T;
}
