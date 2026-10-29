import { useId, useState } from "react";
import { ArrowRight, BookOpen, Calculator, FlaskConical } from "lucide-react";
import { lessonDeepDives } from "../lib/lessonDeepDives";
import { MathFormula } from "./MathFormula";
import "./lesson-deep-dive.css";

const sections = [
  { id: "example", label: "算一个例子", description: "跟着步骤，理解计算", icon: Calculator },
  { id: "observe", label: "动手观察", description: "调整参数，对照变化", icon: FlaskConical },
  { id: "terms", label: "关键概念", description: "理解术语，连起知识", icon: BookOpen },
] as const;

export default function LessonDeepDive({ lessonId }: { lessonId: string }) {
  const [section, setSection] = useState<"example" | "observe" | "terms">("example");
  const content = lessonDeepDives[lessonId];
  const id = useId();
  if (!content) return null;
  return <section className="lesson-deep-dive" aria-label="本节拓展学习">
    <div className="deep-dive-heading"><div><span>EXPLORE THE DETAILS</span><h2>把知识，再走深一步。</h2></div><span className="deep-dive-caption">教学示例 · 对照实验理解</span></div>
    <div className="deep-dive-switcher" role="group" aria-label="选择拓展内容">
      {sections.map(item => <button key={item.id} type="button" aria-label={item.label} aria-pressed={section === item.id} aria-controls={id} onClick={() => setSection(item.id)}><item.icon size={18} aria-hidden="true" /><span><strong>{item.label}</strong><small>{item.description}</small></span></button>)}
    </div>
    <div id={id} className="deep-dive-content">
      {section === "example" && <><div className="deep-dive-panel-heading"><span>分步计算</span><h3>{content.example.title}</h3></div><div className="deep-dive-example-grid"><ol className="deep-dive-steps">{content.example.steps.map((step, index) => <li key={step}><span className="deep-dive-number" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span><p>{step}</p></li>)}</ol><aside className="deep-dive-result" aria-label="公式与结论"><div className="deep-dive-equation"><span className="deep-dive-label">公式对照</span><MathFormula tex={content.example.tex} block /></div><div className="deep-dive-takeaway"><strong>从计算中理解</strong><p>{content.example.takeaway}</p></div></aside></div></>}
      {section === "observe" && <><div className="deep-dive-panel-heading"><span>实验指引</span><h3>先预测变化，再动手验证</h3></div><ol className="deep-dive-observations">{content.observations.map((item, index) => <li key={item.action}><div className="deep-dive-task-heading"><span className="deep-dive-number" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span><span className="deep-dive-label">调整与操作</span></div><h4>{item.action}</h4><div className="deep-dive-expected"><span className="deep-dive-label">重点观察</span><p>{item.expected}</p></div></li>)}</ol><p className="deep-dive-observation-note">观察方法：每次尽量只改变一个条件，记录变化前后的输入、参数与输出。</p></>}
      {section === "terms" && <><div className="deep-dive-panel-heading"><span>概念速查</span><h3>用准确的词，描述你看到的现象</h3></div><dl>{content.terms.map(item => <div key={item.term}><dt>{item.term}</dt><dd>{item.definition}</dd></div>)}</dl></>}
    </div>
    <div className="deep-dive-connection"><p>{content.connection}</p><a href={`#${content.next}`}>{content.next === "overview" ? "回到学习总览" : "探索下一节"}<ArrowRight size={14} aria-hidden="true" /></a></div>
  </section>;
}
