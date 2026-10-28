import { crossCorrelate } from "./convolution.ts";
import type { NetworkAnalysis, NetworkLayer } from "./networkBuilder";
import type { XorSnapshot } from "./xor";

export const CHALLENGE_IDS = ["linear-fit", "edge-detector", "cnn-architect", "xor-master"] as const;
export type ChallengeId = typeof CHALLENGE_IDS[number];
export const CHALLENGE_STORAGE_KEY = "zhiwei-challenges-v1";
export const ACTIVE_CHALLENGE_STORAGE_KEY = "zhiwei-last-challenge-v1";
export const EDGE_INPUT = Array.from({ length: 8 }, () => Array.from({ length: 8 }, (_, column) => column < 4 ? 0 : 1));

export function linearChallenge(weight: number, bias: number) {
  const errors = [-2, -1, 0, 1, 2].map(x => weight * x + bias - (2 * x + 0.5));
  const mse = errors.reduce((sum, error) => sum + error * error, 0) / errors.length;
  return { mse, passed: Number.isFinite(mse) && mse <= 0.01 };
}

export function edgeChallenge(kernel: number[][]) {
  if (kernel.length !== 3 || kernel.some(row => row.length !== 3 || row.some(value => !Number.isFinite(value) || Math.abs(value) > 3))) {
    throw new Error("卷积核需要 3 × 3 个 −3 到 3 之间的有限数值。");
  }
  const output = crossCorrelate(EDGE_INPUT, kernel);
  const uniform = output.flatMap(row => [row[0], row[1], row[4], row[5]]);
  const edge = output.flatMap(row => [row[2], row[3]]);
  const uniformResponse = Math.max(...uniform.map(Math.abs));
  const edgeResponse = Math.max(...edge.map(Math.abs));
  return { output, uniformResponse, edgeResponse, passed: uniformResponse <= 0.05 && edgeResponse >= 1 };
}

export function architectureChallenge(layers: NetworkLayer[] | null, analysis: NetworkAnalysis | null) {
  const required = ["conv", "relu", "pool", "flatten", "linear"] as const;
  const requirements = required.map(type => ({ type, met: Boolean(layers?.some(layer => layer.type === type)) }));
  const valid = Boolean(analysis?.valid && analysis.output?.length === 1 && analysis.output[0] === 10);
  return { requirements, valid, passed: Boolean(layers?.length && valid && requirements.every(item => item.met)) };
}

export function xorChallenge(snapshot: XorSnapshot | null) {
  return Boolean(snapshot && snapshot.iteration > 0 && Number.isFinite(snapshot.loss) && snapshot.loss <= 0.15 && snapshot.accuracy === 1 && snapshot.samples.length === 4 && snapshot.samples.every(sample => sample.prediction === sample.target));
}

export function parseChallengeProgress(value: unknown): ChallengeId[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const stored = value as Record<string, unknown>;
  if (stored.version !== 1 || !Array.isArray(stored.completed)) return [];
  return [...new Set(stored.completed.filter((id): id is ChallengeId => typeof id === "string" && (CHALLENGE_IDS as readonly string[]).includes(id)))];
}

export function readChallengeProgress(): ChallengeId[] {
  try { return parseChallengeProgress(JSON.parse(localStorage.getItem(CHALLENGE_STORAGE_KEY) || "null")); }
  catch { return []; }
}

export function readActiveChallenge(): ChallengeId {
  try {
    const saved = localStorage.getItem(ACTIVE_CHALLENGE_STORAGE_KEY);
    return (CHALLENGE_IDS as readonly string[]).includes(saved || "") ? saved as ChallengeId : "linear-fit";
  } catch { return "linear-fit"; }
}
