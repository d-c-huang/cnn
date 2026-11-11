import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { readMotionLevel } from "./lib/motion";
import App from "./App";
import { LearningProgressProvider } from "./lib/learningProgress";
import "./fonts.css";
import "./styles.css";
import "./phase2.css";
import "./design.css";
import "./experiment-theme.css";
import "./playground.css";
import "./readability.css";
import "./neural-upgrade.css";
import "./app-layout.css";

document.documentElement.dataset.motion = readMotionLevel();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LearningProgressProvider><App /></LearningProgressProvider>
  </StrictMode>,
);
