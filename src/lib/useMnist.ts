import { useCallback, useEffect, useRef, useState } from "react";
import { mnistRequest } from "./mnist";

export function useMnistResource<T>(path: string, interval = 1500) {
  const [data, setData] = useState<T | null>(null);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const version = useRef(0);
  const request = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    if (request.current) return;
    const token = ++version.current;
    const controller = new AbortController();
    request.current = controller;
    try {
      const result = await mnistRequest<T>(path, { signal: controller.signal });
      if (version.current === token) {
        setData(result);
        setError(null);
      }
    } catch (cause) {
      if (version.current === token && !controller.signal.aborted)
        setError(
          cause instanceof Error ? cause.message : "无法读取训练服务状态。",
        );
    } finally {
      if (version.current === token) setPending(false);
      if (request.current === controller) request.current = null;
    }
  }, [path]);
  useEffect(() => {
    setData(null);
    setPending(true);
    void refresh();
    const timer = window.setInterval(() => void refresh(), interval);
    return () => {
      window.clearInterval(timer);
      version.current++;
      request.current?.abort();
      request.current = null;
    };
  }, [refresh, interval]);
  return { data, pending, error, refresh };
}

export function downloadText(
  name: string,
  value: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
