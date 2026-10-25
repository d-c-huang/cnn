import { useEffect, useRef, useState } from "react";

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

export function useCalculation<T>(path: string, payload: unknown) {
  const key = JSON.stringify(payload);
  const [state, setState] = useState<{
    key: string;
    data: T | null;
    error: string | null;
    pending: boolean;
  }>({ key, data: null, error: null, pending: true });
  const version = useRef(0);
  useEffect(() => {
    const currentVersion = ++version.current;
    const controller = new AbortController();
    // Clear the preceding result so a fast slider movement never pairs new inputs with old values.
    setState({ key, data: null, error: null, pending: true });
    const timeout = window.setTimeout(() => {
      calculate<T>(path, JSON.parse(key), controller.signal)
        .then((data) => {
          if (version.current === currentVersion)
            setState({ key, data, error: null, pending: false });
        })
        .catch((error) => {
          if (
            !controller.signal.aborted &&
            version.current === currentVersion
          ) {
            setState({
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
  return state.key === key ? state : { data: null, error: null, pending: true };
}

export function format(value: number, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1e5 || (Math.abs(value) < 0.0001 && value !== 0))
    return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
