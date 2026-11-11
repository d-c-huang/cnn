import { useEffect, useRef, useState } from "react";
import { useMotionLevel } from "./motion";

/** Only interpolates presentation; computation, exports and ARIA retain the target. */
export function useTween(target: number, duration = 180) {
  const motion = useMotionLevel();
  const [value, setValue] = useState(target);
  const current = useRef(target);
  useEffect(() => {
    if (motion !== "full" || !Number.isFinite(target) || duration <= 0) {
      current.current = target; setValue(target); return;
    }
    const from = current.current;
    const start = performance.now();
    let raf = requestAnimationFrame(function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      current.current = from + (target - from) * (1 - (1 - t) ** 3);
      setValue(current.current);
      if (t < 1) raf = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [target, duration, motion]);
  return motion === "full" ? value : target;
}
