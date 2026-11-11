import { useState } from "react";
import { predictionPrompts } from "../lib/studyQuestions";

export default function PredictionPrompt({ lessonId }: { lessonId: string }) {
  const prompt = predictionPrompts[lessonId];
  const [choice, setChoice] = useState<number | null>(null);
  if (!prompt) return null;
  return <aside className="prediction-prompt" aria-label="运行前先预判"><strong>先预判，再验证</strong><p>{prompt.question}</p><div role="group" aria-label={prompt.question}>{prompt.options.map((option, index) => <button key={option} className="secondary-button" aria-pressed={choice === index} onClick={() => setChoice(index)}>{option}</button>)}</div>{choice !== null && <p role="status">你的预判：{prompt.options[choice]}。{prompt.observation}</p>}</aside>;
}
