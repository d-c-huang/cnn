import { useId, useState } from "react";
import { ArrowUpRight, BookOpen, Check, ChevronDown, CircleHelp, Lightbulb, RotateCcw, Target } from "lucide-react";
import { lessonInsights } from "../lib/lessonInsights";
import type { LessonInsight } from "../lib/lessonInsights";

function KnowledgeCheck({ quiz }: { quiz: LessonInsight["quiz"] }) {
  const [selected, setSelected] = useState<number | null>(null);
  const answered = selected !== null;
  const correct = selected === quiz.answer;
  const feedbackId = useId();

  return (
    <aside className="insight-quiz" aria-label="本节自测">
      <div className="insight-card-label"><CircleHelp size={15} /><span>KNOWLEDGE CHECK</span><span>01 题</span></div>
      <h3>理解了，就挑战一下。</h3>
      <fieldset className="insight-question" aria-describedby={answered ? feedbackId : undefined}>
        <legend>{quiz.question}</legend>
        <div className="insight-options">
          {quiz.options.map((option, index) => {
            const state = !answered ? "" : index === quiz.answer ? "is-correct" : index === selected ? "is-incorrect" : "";
            return (
              <button
                key={option}
                type="button"
                className={`insight-option ${state}`}
                aria-pressed={selected === index}
                onClick={() => setSelected(index)}
              >
                <span className="insight-option-letter">{String.fromCharCode(65 + index)}</span>
                <span>{option}</span>
                {answered && index === quiz.answer && <Check size={15} aria-label="正确选项" />}
              </button>
            );
          })}
        </div>
      </fieldset>
      {answered ? (
        <div id={feedbackId} className={`insight-feedback ${correct ? "is-correct" : "is-incorrect"}`} role="status">
          <strong>{correct ? "回答正确，继续探索。" : "再想一步，看看原因。"}</strong>
          <p>{quiz.explanation}</p>
          <button type="button" className="insight-retry" onClick={() => setSelected(null)}><RotateCcw size={12} />重新作答</button>
        </div>
      ) : <p className="insight-quiz-note">选择一个答案，查看原理解析。无需连接计算服务。</p>}
    </aside>
  );
}

export default function LessonInsights({ lessonId }: { lessonId: string }) {
  const insight = lessonInsights[lessonId];
  const headingId = useId();
  if (!insight) return null;

  return (
    <section className="insight-section" aria-labelledby={headingId}>
      <div className="insight-heading">
        <div><span className="insight-eyebrow">GO DEEPER</span><h2 id={headingId}>不止观察，更要理解。</h2></div>
        <span className="insight-heading-note"><BookOpen size={14} />原理与自测</span>
      </div>
      <div className="insight-objective"><Target size={17} /><span>学习目标</span><p>{insight.objective}</p><ArrowUpRight size={17} /></div>
      <div className="insight-layout">
        <details key={`theory-${lessonId}`} className="insight-theory" open>
          <summary><span><BookOpen size={16} />把原理连起来</span><ChevronDown size={16} /></summary>
          <div className="insight-theory-body">
            <ol className="insight-steps">
              {insight.steps.map((step, index) => (
                <li key={step.title}><span className="insight-step-number">0{index + 1}</span><div><h3>{step.title}</h3><p>{step.description}</p></div></li>
              ))}
            </ol>
            <div className="insight-formula"><span>CORE EQUATION</span><code>{insight.formula}</code><p>{insight.formulaNote}</p></div>
            <div className="insight-misconception"><Lightbulb size={16} /><div><h3>避开一个常见误区</h3><p>{insight.misconception}</p></div></div>
          </div>
        </details>
        <KnowledgeCheck key={`quiz-${lessonId}`} quiz={insight.quiz} />
      </div>
    </section>
  );
}
