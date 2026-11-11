import { useSyncExternalStore } from "react";

export type MotionLevel = "full" | "reduced" | "off";
const key = "zhiwei-motion-v1";
const query = "(prefers-reduced-motion: reduce)";
export function readMotionLevel(): MotionLevel {
  try {
    const saved = localStorage.getItem(key);
    if (saved === "full" || saved === "reduced" || saved === "off") return saved;
  } catch { /* The system preference remains available without storage. */ }
  return window.matchMedia(query).matches ? "reduced" : "full";
}
export function setMotionLevel(level: MotionLevel) {
  try { localStorage.setItem(key, level); } catch { /* Keep the preference for this session. */ }
  document.documentElement.dataset.motion = level;
  window.dispatchEvent(new Event("zhiwei-motion-change"));
}
function snapshot(): MotionLevel {
  const level = document.documentElement.dataset.motion;
  return level === "full" || level === "reduced" || level === "off" ? level : readMotionLevel();
}
function subscribe(listener: () => void) {
  const media = window.matchMedia(query);
  const update = () => { document.documentElement.dataset.motion = readMotionLevel(); listener(); };
  window.addEventListener("zhiwei-motion-change", listener);
  window.addEventListener("storage", update);
  media.addEventListener("change", update);
  return () => {
    window.removeEventListener("zhiwei-motion-change", listener);
    window.removeEventListener("storage", update);
    media.removeEventListener("change", update);
  };
}
export function useMotionLevel() { return useSyncExternalStore(subscribe, snapshot, () => "reduced" as const); }
