import { lessonInsights } from "../lib/lessonInsights";
import { lessonTasks } from "../lib/learningProgress";

export default function PhaseReview({ lessonIds }: { lessonIds: string[] }) {
  return <section className="phase-review" aria-label="阶段回顾"><h3>回顾你刚完成的实验</h3><ol>{lessonIds.map(id => <li key={id}><strong>{lessonTasks[id].title}</strong><p>{lessonInsights[id].objective}</p><small>操作回顾：{lessonTasks[id].instruction}</small></li>)}</ol><h3>再辨清两处容易混淆的概念</h3>{lessonIds.slice(-2).map(id => <p key={id}>{lessonInsights[id].misconception}</p>)}</section>;
}
