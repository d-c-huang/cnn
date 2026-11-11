import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useMotionLevel } from "../lib/motion";

export default function ExperimentEntry({ children }: { children: ReactNode }) {
  const motion = useMotionLevel();
  const [entered, setEntered] = useState(() => motion !== "full");
  useEffect(() => { if (motion !== "full") setEntered(true); }, [motion]);
  useEffect(() => {
    const timer = window.setTimeout(() => setEntered(true), 600);
    return () => window.clearTimeout(timer);
  }, []);
  return <div className="experiment-entry" data-entered={entered}>{children}</div>;
}
