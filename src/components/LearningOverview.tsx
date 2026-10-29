import { ArrowRight, ArrowUpRight, Check, Cpu, Grid3X3, Layers3, Network, Play, ScanLine, Trophy } from "lucide-react";
import "./learning-overview.css";

type Tool = "convolution" | "network-builder" | "challenges";

interface LearningOverviewProps {
  currentLesson: { name: string; title: string; description: string; index: number };
  phase: number;
  completed: string[];
  lessonIds: string[];
  effects: boolean;
  onContinue: () => void;
  onSelectPhase: (start: number) => void;
  onOpenTool: (tool: Tool) => void;
}

const stages = [
  { name: "理解基础", description: "从像素、神经元到梯度下降", start: 0, end: 5, icon: Layers3 },
  { name: "连接网络", description: "串起计算图、反向传播与 XOR", start: 5, end: 8, icon: Network },
  { name: "训练模型", description: "用 MNIST 数据训练 MLP 与 CNN", start: 8, end: 11, icon: Cpu },
  { name: "创造应用", description: "手写识别、模型推理与部署", start: 11, end: 13, icon: ScanLine },
];

const tools = [
  { id: "convolution" as const, name: "卷积可视化", description: "逐格扫描图像，看清卷积、激活与池化的计算。", icon: Grid3X3, tag: "观察计算" },
  { id: "network-builder" as const, name: "神经网络搭建器", description: "亲手搭建网络，训练并评估自己的数字识别模型。", icon: Layers3, tag: "自由搭建" },
  { id: "challenges" as const, name: "实验挑战关卡", description: "带着具体目标动手，在挑战中检验你的理解。", icon: Trophy, tag: "检验理解" },
];

export default function LearningOverview({ currentLesson, phase, completed, lessonIds, effects, onContinue, onSelectPhase, onOpenTool }: LearningOverviewProps) {
  const completedCount = lessonIds.filter(id => completed.includes(id)).length;
  const currentComplete = completed.includes(lessonIds[currentLesson.index]);
  const progress = lessonIds.length ? Math.round(completedCount / lessonIds.length * 100) : 0;

  return (
    <div className="learning-overview" data-effects={effects ? "on" : "off"}>
      <div className="lo-welcome-grid">
        <section className="lo-welcome" aria-labelledby="lo-title">
          <span className="lo-eyebrow"><span aria-hidden="true" />LEARNING WORKSPACE</span>
          <h1 id="lo-title">让每一次动手，<br />都离理解 AI 更近一步。</h1>
          <p>沿着学习路径逐步探索，或进入工坊自由实验。<br className="lo-desktop-break" />从一个像素开始，把模型的计算看清楚。</p>
          <div className="lo-facts" aria-label="课程概览">
            <span><strong>{lessonIds.length}</strong> 交互实验</span>
            <span><strong>4</strong> 学习阶段</span>
            <span><strong>3</strong> 探索工具</span>
          </div>
        </section>

        <section className="lo-resume" aria-labelledby="lo-resume-title">
          <div className="lo-resume-top"><span className="lo-eyebrow">YOUR NEXT STEP</span><span className="lo-lesson-number">{String(currentLesson.index + 1).padStart(2, "0")} / {lessonIds.length}</span></div>
          <span className="lo-current-lesson">{currentComplete ? "再次探索" : "当前实验"} · {currentLesson.name}</span>
          <h2 id="lo-resume-title">{currentLesson.title}</h2>
          <p>{currentLesson.description}</p>
          <button type="button" className="lo-continue" onClick={onContinue}><Play size={14} fill="currentColor" />{currentComplete ? "再次进入实验" : "继续学习"}<ArrowRight size={17} /></button>
          <div className="lo-progress-heading"><span>已完成 {completedCount} / {lessonIds.length} 个实验</span><strong>{progress}%</strong></div>
          <div className="lo-progress" role="progressbar" aria-label="学习总进度" aria-valuemin={0} aria-valuemax={lessonIds.length} aria-valuenow={completedCount}><span style={{ width: `${progress}%` }} /></div>
        </section>
      </div>

      <section className="lo-section" aria-labelledby="lo-path-title">
        <div className="lo-section-heading"><div><span className="lo-eyebrow">LEARNING PATH</span><h2 id="lo-path-title">循序渐进，连接每一块知识</h2></div><span className="lo-section-note">选择阶段，进入对应实验</span></div>
        <div className="lo-stage-grid">
          {stages.map((stage, index) => {
            const Icon = stage.icon;
            const ids = lessonIds.slice(stage.start, stage.end);
            const done = ids.filter(id => completed.includes(id)).length;
            const finished = ids.length > 0 && done === ids.length;
            return <button key={stage.name} type="button" className={`lo-stage ${phase === index + 1 ? "is-current" : ""}`} onClick={() => onSelectPhase(stage.start)} aria-label={`${stage.name}，已完成 ${done} / ${ids.length} 个实验`}>
              <div className="lo-stage-top"><span className="lo-stage-icon"><Icon size={20} /></span><span className="lo-stage-index">0{index + 1}</span>{finished ? <Check size={17} className="lo-stage-state" aria-label="已完成" /> : phase === index + 1 ? <span className="lo-stage-current">当前阶段</span> : <ArrowUpRight size={16} className="lo-stage-state" />}</div>
              <h3>{stage.name}</h3><p>{stage.description}</p>
              <div className="lo-stage-footer"><span>{done} / {ids.length} 已完成</span><ArrowRight size={15} /></div>
              <div className="lo-stage-progress" aria-hidden="true"><span style={{ width: `${ids.length ? done / ids.length * 100 : 0}%` }} /></div>
            </button>;
          })}
        </div>
      </section>

      <section className="lo-section" aria-labelledby="lo-tools-title">
        <div className="lo-section-heading"><div><span className="lo-eyebrow">EXPLORATION STUDIO</span><h2 id="lo-tools-title">把好奇心，变成自己的实验</h2></div><span className="lo-section-note">从观察到创造</span></div>
        <div className="lo-tool-grid">{tools.map(tool => {
          const Icon = tool.icon;
          return <button key={tool.id} type="button" className="lo-tool" onClick={() => onOpenTool(tool.id)}>
            <span className="lo-tool-icon"><Icon size={21} /></span><div><span className="lo-tool-tag">{tool.tag}</span><h3>{tool.name}</h3><p>{tool.description}</p></div><ArrowUpRight className="lo-tool-arrow" size={17} />
          </button>;
        })}</div>
      </section>
    </div>
  );
}
