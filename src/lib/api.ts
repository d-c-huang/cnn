import { useEffect, useRef, useState } from "react";

export function calculationStatus(error: string | null, pending: boolean) {
  return error ? "当前参数的本地预览，后端计算结果未同步"
    : pending ? "当前参数的即时预览，正在等待后端计算结果" : "已同步后端计算结果";
}

export async function calculate<T>(
  path: string,
  payload: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(8000)])
      : AbortSignal.timeout(8000),
  }).catch((error) => {
    if (error instanceof DOMException && error.name === "TimeoutError")
      throw new Error("计算请求超时，请确认 NumPy 服务仍在运行。");
    throw error;
  });
  if (!response.ok) {
    if (response.status === 422)
      throw new Error("参数超出计算范围，请重置参数后重试。");
    throw new Error(`计算服务返回错误（${response.status}），请稍后重试。`);
  }
  return response.json() as Promise<T>;
}

// A preview must be derived from the same current payload. It is display-only:
// actions that commit an optimizer step must use confirmedData instead.
export function useCalculation<T>(path: string, payload: unknown, preview?: T) {
  const key = JSON.stringify(payload);
  const [state, setState] = useState<{
    path: string;
    key: string;
    data: T | null;
    error: string | null;
    pending: boolean;
  }>({ path, key, data: null, error: null, pending: true });
  const lastSuccess = useRef<{ path: string; data: T } | null>(null);
  const version = useRef(0);
  useEffect(() => {
    const currentVersion = ++version.current;
    const controller = new AbortController();
    // Clear the preceding result so a fast slider movement never pairs new inputs with old values.
    setState({ path, key, data: null, error: null, pending: true });
    const timeout = window.setTimeout(() => {
      calculate<T>(path, JSON.parse(key), controller.signal)
        .then((data) => {
          if (!controller.signal.aborted && version.current === currentVersion) {
            lastSuccess.current = { path, data };
            setState({ path, key, data, error: null, pending: false });
          }
        })
        .catch((error) => {
          if (
            !controller.signal.aborted &&
            version.current === currentVersion
          ) {
            setState({
              path,
              key,
              data: null,
              error:
                error instanceof Error
                  ? error.message
                  : "无法连接计算服务，请确认后端已启动。",
              pending: false,
            });
          }
        });
    }, 70);
    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [path, key]);
  const current = state.path === path && state.key === key ? state : { data: null, error: null, pending: true };
  // Expose the last successful sample separately; callers must explicitly derive
  // a preview rather than presenting old results as answers for new parameters.
  return {
    ...current,
    data: current.data ?? preview ?? null,
    confirmedData: current.data,
    previewing: current.data === null && preview !== undefined,
    previousData: lastSuccess.current?.path === path ? lastSuccess.current.data : null,
  };
}

export function format(value: number, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1e5 || (Math.abs(value) < 0.0001 && value !== 0))
    return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
