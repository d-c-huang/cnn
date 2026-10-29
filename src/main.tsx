import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { LearningProgressProvider } from "./lib/learningProgress";
import "./styles.css";
import "./phase2.css";
import "./design.css";
import "./experiment-theme.css";
import "./playground.css";
import "./readability.css";
import "./neural-upgrade.css";
import "./app-layout.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LearningProgressProvider><App /></LearningProgressProvider>
  </StrictMode>,
);
