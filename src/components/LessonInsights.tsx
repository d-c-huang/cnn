import { MathFormula } from "./MathFormula";
import { useId, useState } from "react";
import { ArrowUpRight, BookOpen, Check, ChevronDown, CircleHelp, Lightbulb, RotateCcw, Target } from "lucide-react";
import { lessonInsights } from "../lib/lessonInsights";
import { useLearningProgress } from "../lib/learningProgress";
import { recallFeedback, transferQuestions } from "../lib/studyQuestions";
import type { StudyQuestion } from "../lib/studyQuestions";
import { GlossaryText } from "./Glossary";
import LessonDeepDive from "./LessonDeepDive";

function KnowledgeCheck({ quiz, number, onPass }: { quiz: StudyQuestion; number: number; onPass: () => void }) {
  const [selected, setSelected] = useState<number | null>(null);
  const answered = selected !== null;
  const correct = selected === quiz.answer;
  const feedbackId = useId();

  return (
    <div className="insight-quiz" aria-label={number === 1 ? "概念自测" : "迁移自测"}>
      <div className="insight-card-label"><CircleHelp size={15} /><span>{number === 1 ? "概念自测" : "迁移自测"}</span><span>0{number} / 02</span></div>
      <h3>理解了，就挑战一下。</h3>
      <fieldset className="insight-question" aria-describedby={answered ? feedbackId : undefined}>
        <legend>{quiz.question}</legend>
        <div className="insight-options">
          {quiz.options.map((option, index) => {
            const state = selected !== index ? "" : correct ? "is-correct" : "is-incorrect";
            return (
              <button
                key={option}
                type="button"
                className={`insight-option ${state}`}
                aria-pressed={selected === index}
                onClick={() => {
                  setSelected(index);
                  if (index === quiz.answer) onPass();
                }}
              >
                <span className="insight-option-letter">{String.fromCharCode(65 + index)}</span>
                <span>{option}</span>
                {selected === index && correct && <Check size={15} aria-label="回答正确" />}
              </button>
            );
          })}
        </div>
      </fieldset>
      {answered ? (
        <div id={feedbackId} className={`insight-feedback ${correct ? "is-correct" : "is-incorrect"}`} role="status">
          <strong>{correct ? "回答正确，继续探索。" : "再想一步，看看原因。"}</strong>
          <p>{correct ? quiz.explanation : quiz.whyWrong[selected]}</p>
          <button type="button" className="insight-retry" onClick={() => setSelected(null)}><RotateCcw size={12} />重新作答</button>
        </div>
      ) : <p className="insight-quiz-note">选择一个答案，查看对应解析。两题均答对后通过本节自测。</p>}
    </div>
  );
}

export function LessonQuiz({ lessonId }: { lessonId: string }) {
  const { verifyQuiz, quizzes } = useLearningProgress();
  const [passed, setPassed] = useState<number[]>([]);
  const insight = lessonInsights[lessonId];
  if (!insight) return null;
  const pass = (number: number) => {
    const next = [...new Set([...passed, number])];
    setPassed(next);
    if (next.length === 2) verifyQuiz(lessonId);
  };
  return <section className="lesson-quiz-target" id={`lesson-quiz-${lessonId}`} tabIndex={-1} aria-label="本节自测">
    <div className="insight-heading"><h2>本节自测</h2><span>{quizzes[lessonId] ? "已通过，记录已保留" : `${passed.length} / 2 题已通过`}</span></div>
    <div className="quiz-pair"><KnowledgeCheck quiz={{ ...insight.quiz, whyWrong: recallFeedback[lessonId] }} number={1} onPass={() => pass(1)} /><KnowledgeCheck quiz={transferQuestions[lessonId]} number={2} onPass={() => pass(2)} /></div>
  </section>;
}

export default function LessonInsights({ lessonId }: { lessonId: string }) {
  const insight = lessonInsights[lessonId];
  const headingId = useId();
  if (!insight) return null;

  return (
    <>
    <section className="insight-section" aria-labelledby={headingId}>
      <div className="insight-heading">
        <div><h2 id={headingId}>不止观察，更要理解。</h2></div>
        <span className="insight-heading-note"><BookOpen size={14} />原理讲解</span>
      </div>
      <div className="insight-objective"><Target size={17} /><span>学习目标</span><p>{insight.objective}</p><ArrowUpRight size={17} /></div>
      <div className="insight-layout">
        <details key={`theory-${lessonId}`} className="insight-theory" open>
          <summary><span><BookOpen size={16} />把原理连起来</span><ChevronDown size={16} /></summary>
          <div className="insight-theory-body">
            <ol className="insight-steps">
              {insight.steps.map((step, index) => (
                <li key={step.title}><span className="insight-step-number">0{index + 1}</span><div><h3>{step.title}</h3><p><GlossaryText text={step.description} /></p></div></li>
              ))}
            </ol>
            <div className="insight-formula"><span>核心公式</span><MathFormula tex={insight.formula} block /><p>{insight.formulaNote}</p></div>
            <div className="insight-misconception"><Lightbulb size={16} /><div><h3>避开一个常见误区</h3><p>{insight.misconception}</p></div></div>
          </div>
        </details>

      </div>
    </section>
    <LessonDeepDive key={lessonId} lessonId={lessonId} />
    </>
  );
}
