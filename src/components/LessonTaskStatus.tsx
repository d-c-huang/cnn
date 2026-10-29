import { useId } from "react";
import { ArrowRight, CheckCircle2, Circle, ClipboardCheck } from "lucide-react";
import { lessonTasks, useLearningProgress } from "../lib/learningProgress";
import "./lesson-task-status.css";

export default function LessonTaskStatus({ lessonId, historicalCompleted = false }: { lessonId: string; historicalCompleted?: boolean }) {
  const { tasks, quizzes } = useLearningProgress();
  const headingId = useId();
  const task = Object.hasOwn(lessonTasks, lessonId) ? lessonTasks[lessonId] : undefined;
  if (!task) return null;
  const taskDone = !!tasks[lessonId];
  const quizDone = !!quizzes[lessonId];
  const passed = Number(taskDone) + Number(quizDone);

  return (
    <section className={`lesson-task-status ${passed === 2 ? "is-ready" : ""}`} aria-labelledby={headingId}>
      <div className="lesson-task-heading">
        <h2 id={headingId}><ClipboardCheck size={17} aria-hidden="true" />本节完成条件</h2>
        <span className="lesson-task-count" role="status" aria-live="polite" aria-atomic="true">{passed === 2 ? "两项均已通过" : `${passed} / 2 已通过`}</span>
      </div>
      <ol className="lesson-task-checks">
        <li className={taskDone ? "is-passed" : ""}>
          {taskDone ? <CheckCircle2 size={18} aria-hidden="true" /> : <Circle size={18} aria-hidden="true" />}
          <div><strong>{task.title}<span>{taskDone ? "已达标" : "待达标"}</span></strong><p>{task.instruction}</p></div>
        </li>
        <li className={quizDone ? "is-passed" : ""}>
          {quizDone ? <CheckCircle2 size={18} aria-hidden="true" /> : <Circle size={18} aria-hidden="true" />}
          <div>
            <strong>通过本节自测<span>{quizDone ? "已通过" : "待作答"}</span></strong>
            <p>答对下方「本节自测」，验证对本节原理的理解。</p>
            <button className="lesson-task-quiz-link" type="button" onClick={() => {
              const target = document.getElementById(`lesson-quiz-${lessonId}`);
              if (!target) return;
              target.focus({ preventScroll: true });
              target.scrollIntoView({ block: "start", behavior: !window.matchMedia("(prefers-reduced-motion: reduce)").matches && !target.closest(".effects-off") ? "smooth" : "instant" });
            }}>{quizDone ? "重新查看自测" : "前往本节自测"}<ArrowRight size={13} aria-hidden="true" /></button>
          </div>
        </li>
      </ol>
      <p className="lesson-task-note">{historicalCompleted ? "原有完成记录已保留，可以继续验证本节任务与自测。" : passed === 2 ? "已满足本节完成条件，可以记录学习进度。" : "完成实验任务并答对自测后，即可记录本节完成进度。"}{passed > 0 && " 已通过的条件会保留，可以继续调参或重新作答。"}</p>
    </section>
  );
}
