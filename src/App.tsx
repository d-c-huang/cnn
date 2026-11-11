import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  ArrowDownRight,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  CheckCheck,
  ChevronRight,
  Code2,
  FlaskConical,
  CircleHelp,
  Lightbulb,
  Network,
  Sparkles,
  LayoutDashboard,
  Menu,
  Type,
  X,
} from "lucide-react";
import { lessons } from "./lib/lessons";
import ExperimentEntry from "./components/ExperimentEntry";
import TraceRibbon from "./components/TraceRibbon";
import { readMotionLevel, setMotionLevel, useMotionLevel } from "./lib/motion";
import { parseLearningRecord } from "./lib/learningRecords";
import LearningOverview from "./components/LearningOverview";
import LessonTaskStatus from "./components/LessonTaskStatus";
import ExperimentBoundary from "./components/ExperimentBoundary";
import { useLearningProgress } from "./lib/learningProgress";
import { phaseCompletion } from "./lib/courseProgress";
const LessonInsights = lazy(() => import("./components/LessonInsights"));
const PredictionPrompt = lazy(() => import("./components/PredictionPrompt"));
const LessonQuiz = lazy(() => import("./components/LessonInsights").then(module => ({ default: module.LessonQuiz })));
const Glossary = lazy(() => import("./components/Glossary"));
const PhaseReview = lazy(() => import("./components/PhaseReview"));
const ConvolutionLab = lazy(() => import("./experiments/playground/ConvolutionLab"));
const NetworkBuilder = lazy(() => import("./experiments/playground/NetworkBuilder"));
const ChallengeHub = lazy(() => import("./experiments/playground/ChallengeHub"));

import {
  type ExplorationWorkspace,
  type Workspace,
  type ReadingSize,
  explorationTools,
  workspaceFromHash,
  readReadingSize
} from "./lib/workspace";

const storageKey = "zhiwei-foundation-v1";

function readProgress(): { active: string; completed: string[] } {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    return {
      active: typeof saved.active === "string" ? saved.active : "data",
      completed: Array.isArray(saved.completed)
        ? saved.completed.filter((id: unknown) =>
            lessons.some((l) => l.id === id),
          )
        : [],
    };
  } catch {
    return { active: "data", completed: [] };
  }
}

export default function App() {
  const learningProgress = useLearningProgress();
  const [active, setActive] = useState(() => {
    const hash = window.location.hash.slice(1);
    const record = readProgress();
    const saved = record.completed.includes(record.active) ? lessons.find(item => !record.completed.includes(item.id))?.id ?? record.active : record.active;
    return lessons.findIndex(
      (l) => l.id === (lessons.some((l) => l.id === hash) ? hash : saved),
    ) >= 0
      ? lessons.findIndex(
          (l) => l.id === (lessons.some((l) => l.id === hash) ? hash : saved),
        )
      : 0;
  });
  const [completed, setCompleted] = useState<string[]>(() => [
    ...new Set(readProgress().completed),
  ]);
  const [explored, setExplored] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [progressEpoch, setProgressEpoch] = useState(0);
  const [engine, setEngine] = useState<"connecting" | "online" | "offline">(
    "connecting",
  );
  const [allDone, setAllDone] = useState(false);
  const [storageWarning, setStorageWarning] = useState(false);
  const motion = useMotionLevel();
  const [recordNotice, setRecordNotice] = useState("");
  const [workspace, setWorkspace] = useState<Workspace>(workspaceFromHash);
  const [readingSize, setReadingSize] = useState<ReadingSize>(readReadingSize);
  const about = useRef<HTMLDialogElement>(null);
  const displaySettings = useRef<HTMLDetailsElement>(null);
  const navigationDialog = useRef<HTMLDialogElement>(null);
  const lesson = lessons[active];
  const Experiment = lesson.component;
  const selectedTool = explorationTools.find(tool => tool.id === workspace);
  const localCompute = workspace === "convolution";
  const phase = active < 5 ? 1 : active < 8 ? 2 : active < 11 ? 3 : active < 13 ? 4 : active < 16 ? 5 : active < 19 ? 6 : active < 22 ? 7 : active < 25 ? 8 : active < 28 ? 9 : 10;
  const engineLabel = localCompute ? "浏览器本地计算" : engine === "online"
    ? phase >= 3 || workspace !== "lessons" ? "API 已连接" : "NumPy 引擎已连接"
    : engine === "offline" ? "计算服务未连接" : "连接计算引擎…";
  const phaseIndices =
    phase === 1
      ? [0, 1, 2, 3, 4]
      : phase === 2
        ? [5, 6, 7]
        : phase === 3
          ? [8, 9, 10]
          : phase === 4 ? [11, 12] : phase === 5 ? [13, 14, 15] : phase === 6 ? [16, 17, 18] : phase === 7 ? [19, 20, 21] : phase === 8 ? [22, 23, 24] : phase === 9 ? [25, 26, 27] : [28, 29, 30];
  const phaseComplete = phaseIndices.every((index) =>
    completed.includes(lessons[index].id),
  );
  const phaseEnd =
    active === phaseIndices.at(-1);
  const historicalCompleted = completed.includes(lesson.id);
  const canComplete = historicalCompleted || Boolean(learningProgress.tasks[lesson.id] && learningProgress.quizzes[lesson.id]);
  const { ready: phaseReady, nextUnfinished } = phaseCompletion(active, completed, lessons.map(item => item.id), phaseIndices);

  useEffect(() => {
    document.documentElement.dataset.readingSize = readingSize;
    try { localStorage.setItem("zhiwei-reading-size-v1", readingSize); } catch { /* Reading preferences also work without storage. */ }
  }, [readingSize]);

  useEffect(() => {
    try {
      localStorage.setItem(
        storageKey,
        JSON.stringify({ active: lesson.id, completed }),
      );
    } catch {
      setStorageWarning(true);
    }
  }, [lesson.id, completed]);

  useEffect(() => {
    let live = true;
    const controllers = new Set<AbortController>();
    const check = async () => {
      const controller = new AbortController();
      controllers.add(controller);
      const timeout = window.setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch("/api/health", {
          signal: controller.signal,
        });
        const data = response.ok ? await response.json() : null;
        if (live)
          setEngine(
            data?.engine === "numpy" && data?.status === "ok"
              ? "online"
              : "offline",
          );
      } catch {
        if (live) setEngine("offline");
      } finally {
        window.clearTimeout(timeout);
        controllers.delete(controller);
      }
    };
    void check();
    const interval = window.setInterval(() => void check(), 12000);
    return () => {
      live = false;
      window.clearInterval(interval);
      controllers.forEach((controller) => controller.abort());
    };
  }, []);

  useEffect(() => {
    const desktop = window.matchMedia("(min-width: 901px)");
    const closeMobileNavigation = () => {
      if (desktop.matches) navigationDialog.current?.close();
    };
    desktop.addEventListener("change", closeMobileNavigation);
    return () => desktop.removeEventListener("change", closeMobileNavigation);
  }, []);

  useEffect(() => {
    const handleHash = () => {
      if (["main-content", "experiment-workbench"].includes(window.location.hash.slice(1))) return;
      const currentWorkspace = workspaceFromHash();
      navigationDialog.current?.close();
      window.scrollTo({ top: 0, behavior: "instant" });
      if (currentWorkspace !== "lessons") {
        setWorkspace(currentWorkspace);
        return;
      }
      const index = lessons.findIndex(
        (l) => l.id === window.location.hash.slice(1),
      );
      if (index >= 0) {
        setWorkspace("lessons");
        setActive(index);
        setExplored(false);
        setShowCode(false);
        setAllDone(false);
      }
    };
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  function navigate(index: number) {
    navigationDialog.current?.close();
    setWorkspace("lessons");
    setActive(index);
    setExplored(false);
    setShowCode(false);
    setAllDone(false);
    if (window.location.hash !== `#${lessons[index].id}`) window.history.pushState(null, "", `#${lessons[index].id}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function openOverview() {
    navigationDialog.current?.close();
    if (completed.includes(lesson.id)) { const next = lessons.findIndex(item => !completed.includes(item.id)); if (next >= 0) setActive(next); }
    setWorkspace("overview");
    if (window.location.hash !== "#overview") window.history.pushState(null, "", "#overview");
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function openTool(tool: ExplorationWorkspace) {
    navigationDialog.current?.close();
    setWorkspace(tool);
    if (window.location.hash !== `#${tool}`) window.history.pushState(null, "", `#${tool}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function complete() {
    if (!canComplete) {
      const target = document.getElementById(learningProgress.tasks[lesson.id] ? `lesson-quiz-${lesson.id}` : `lesson-task-${lesson.id}`);
      target?.focus({ preventScroll: true }); target?.scrollIntoView({ block: "center", behavior: motion === "full" ? "smooth" : "instant" });
      target?.classList.remove("needs-attention");
      requestAnimationFrame(() => target?.classList.add("needs-attention"));
      return;
    }
    setCompleted((previous) => [...new Set([...previous, lesson.id])]);
    if (phaseEnd) {
      if (phaseReady) setAllDone(true);
      else {
        if (nextUnfinished !== undefined) navigate(nextUnfinished);
      }
    }
    else navigate(active + 1);
  }

  function exportLearning() {
    const record = { version: 1, exportedAt: new Date().toISOString(), active: lesson.id, completed, evidence: { tasks: learningProgress.tasks, quizzes: learningProgress.quizzes } };
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "zhiwei-learning-record.json"; anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setRecordNotice("学习记录已导出。包含完成情况和验证记录，不包含模型权重。");
  }
  async function importLearning(file: File) {
    try {
      if (file.size > 1024 * 1024) throw new Error("学习记录不能超过 1 MB。");
      const record = parseLearningRecord(await file.text());
      learningProgress.merge(record.evidence);
      setCompleted(previous => [...new Set([...previous, ...record.completed])]);
      navigate(lessons.findIndex(item => item.id === record.active));
      setRecordNotice("学习记录已合并，原有完成情况已保留。");
    } catch (cause) { setRecordNotice(cause instanceof Error ? cause.message : "导入失败，请检查 JSON 文件。"); }
  }

  const sidebarContent = <>

        <a
          className="brand"
          href="#overview"
          onClick={(e) => {
            e.preventDefault();
            openOverview();
          }}
          aria-label="知微 AI 实验室首页"
        >
          <span className="brand-mark">
            <Network size={25} strokeWidth={1.7} />
          </span>
          <span>
            <strong>
              知微<span className="brand-divider">/</span>
              <span className="brand-english">AI LAB</span>
            </strong>
            <small>EXPLORE · BUILD · UNDERSTAND</small>
          </span>
        </a>
        <button type="button" className={`sidebar-overview-link ${workspace === "overview" ? "active" : ""}`} aria-current={workspace === "overview" ? "page" : undefined} onClick={openOverview}>
          <LayoutDashboard size={17} />学习总览<ChevronRight size={14} />
        </button>
        <div className="course-label">
          <span>学习路径</span>
        </div>
        <div className="phase-switcher" role="group" aria-label="选择学习阶段">
          <button aria-pressed={workspace === "lessons" && phase === 1} onClick={() => navigate(0)}>
            01 基础
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 2} onClick={() => navigate(5)}>
            02 网络
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 3} onClick={() => navigate(8)}>
            03 MNIST
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 4} onClick={() => navigate(11)}>
            04 应用
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 5} onClick={() => navigate(13)}>
            05 语言
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 6} onClick={() => navigate(16)}>
            06 Transformer
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 7} onClick={() => navigate(19)}>
            07 LoRA
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 8} onClick={() => navigate(22)}>
            08 SFT
          </button>
          <button aria-pressed={workspace === "lessons" && phase === 9} onClick={() => navigate(25)}>09 DPO</button>
          <button aria-pressed={workspace === "lessons" && phase === 10} onClick={() => navigate(28)}>10 GRPO</button>
        </div>
        <div className="phase-label">
          <span>{String(phase).padStart(2, "0")}</span>
          <strong>
            {phase === 1
              ? "基础可视化实验"
              : phase === 2
                ? "计算图与神经网络"
                : phase === 3
                  ? "MNIST 模型训练"
                  : phase === 4 ? "模型推理与部署" : phase === 5 ? "语言模型训练" : phase === 6 ? "Transformer 语言模型" : phase === 7 ? "LoRA 预训练模型微调" : phase === 8 ? "SFT 指令微调与对话" : phase === 9 ? "DPO 偏好优化与对比" : "GRPO 强化学习微调"}
          </strong>
          <span className="phase-badge">
            {phaseComplete ? "已完成" : "进行中"}
          </span>
        </div>
        <nav
          className={`lesson-nav ${phase > 1 ? "phase-two-nav" : ""} ${phase === 4 ? "phase-four-nav" : ""}`}
          aria-label="实验导航"
        >
          {phaseIndices.map((i) => {
            const item = lessons[i];
            return (
              <button
                key={item.id}
                className={`lesson-link ${workspace === "lessons" && active === i ? "active" : ""}`}
              aria-current={workspace === "lessons" && active === i ? "step" : undefined}
                onClick={() => navigate(i)}
              >
                <span
                  className={`lesson-number ${completed.includes(item.id) ? "done" : ""}`}
                >
                  {completed.includes(item.id) ? (
                    <Check size={13} />
                  ) : (
                    String(i + 1).padStart(2, "0")
                  )}
                </span>
                <span>{item.name}</span>
                {workspace === "lessons" && active === i && (
                  <ChevronRight className="nav-chevron" size={15} />
                )}
              </button>
            );
          })}
        </nav>
        {phaseComplete && <button className="text-button" onClick={() => { navigate(phaseIndices.at(-1)!); setAllDone(true); }}>查看本阶段回顾</button>}
        <nav className="playground-nav" aria-label="探索工坊">
          <div className="playground-nav-heading"><span>探索工坊</span><span>NEW</span></div>
          {explorationTools.map(tool => {
            const Icon = tool.icon;
            return <button key={tool.id} className={`playground-link ${workspace === tool.id ? "active" : ""}`} aria-current={workspace === tool.id ? "page" : undefined} onClick={() => openTool(tool.id)}><Icon size={17} /><span>{tool.name}</span><ChevronRight size={14} /></button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="progress-heading">
            <span>你的探索进度</span>
            <strong>
              {completed.length}
              <em> / {lessons.length} · {Math.round(completed.length / lessons.length * 100)}%</em>
            </strong>
          </div>
          <div
            className="progress-track"
            role="progressbar"
            aria-label="实验完成进度"
            aria-valuenow={completed.length}
            aria-valuemin={0}
            aria-valuemax={lessons.length}
          >
            <div
              style={{ width: `${(completed.length / lessons.length) * 100}%` }}
            />
          </div>
          <p>
            {storageWarning
              ? "当前浏览器无法保存进度"
              : "学习进度保存在当前浏览器"}
          </p>
          <button
            className="about-button"
            onClick={() => about.current?.showModal()}
          >
            <BookOpen size={15} />
            关于这个实验室
            <ArrowDownRight size={14} />
          </button>
        </div>

  </>;

  return (
    <div className={`app-shell ${motion === "full" ? "effects-on" : "effects-off"} ${workspace === "lessons" ? "focus-mode" : ""}`}>
      <a className="skip-link" href="#main-content" onClick={event => {
        event.preventDefault();
        const target = document.getElementById(workspace === "overview" ? "main-content" : "experiment-workbench");
        target?.focus({ preventScroll: true });
        target?.scrollIntoView({ block: "start" });
      }}>跳转到主要内容</a>
      <aside className="sidebar" aria-label="学习导航"><div className="sidebar-content">{sidebarContent}</div></aside>
      <dialog ref={navigationDialog} id="mobile-navigation" className="navigation-dialog" aria-label="学习导航菜单" onClick={event => {
        if (event.target === event.currentTarget) {
          const bounds = event.currentTarget.getBoundingClientRect();
          if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.currentTarget.close();
        }
      }}>
        <div className="navigation-dialog-heading"><span>学习导航</span><button className="icon-button" type="button" aria-label="关闭导航" onClick={() => navigationDialog.current?.close()}><X size={20} /></button></div>
        <div className="sidebar-content">{sidebarContent}</div>
      </dialog>

      <div className="main-shell">
        <header className="topbar">
          <button className="mobile-menu-button icon-button" type="button" aria-label="打开导航" aria-haspopup="dialog" aria-controls="mobile-navigation" onClick={() => navigationDialog.current?.showModal()}><Menu size={21} /></button>
          <div className="breadcrumb">
            <FlaskConical size={16} />
            <span>
              {workspace === "overview" ? "学习工作台" : selectedTool ? "探索工坊" : phase === 1
                ? "基础实验"
                : phase === 2
                  ? "神经网络"
                  : phase === 3
                    ? "MNIST 训练"
                    : phase === 4 ? "模型应用" : phase === 5 ? "语言模型" : phase === 6 ? "Transformer" : phase === 7 ? "LoRA 微调" : phase === 8 ? "SFT 指令微调" : phase === 9 ? "DPO 偏好优化" : "GRPO 强化学习"}
            </span>
            <ChevronRight size={13} />
            <strong>{workspace === "overview" ? "学习总览" : selectedTool?.name ?? lesson.name}</strong>
          </div>
          <div className="topbar-actions">
            <details className="display-settings" ref={displaySettings}>
              <summary aria-label="调整阅读字号" title="调整阅读字号"><Type size={16} /><span>字号</span></summary>
              <div className="reading-popover" role="group" aria-label="阅读字号设置">
                <strong>阅读字号</strong><p>高分辨率屏幕建议使用舒适或大字</p>
                <div>{([{id:"standard",name:"标准"},{id:"comfortable",name:"舒适"},{id:"large",name:"大字"}] as const).map(option => <button key={option.id} aria-pressed={readingSize === option.id} onClick={() => { setReadingSize(option.id); displaySettings.current?.removeAttribute("open"); }}>{option.name}</button>)}</div>
                <small>自动保存到当前浏览器</small>
              </div>
            </details>
            {workspace !== "overview" && <button className="topbar-button overview-return" type="button" onClick={openOverview} aria-label="学习总览" title="返回学习总览">
              <LayoutDashboard size={15} /><span>学习总览</span>
            </button>}
            <label className="motion-control"><Sparkles size={15} /><span className="sr-only">动效设置</span><select aria-label="动效设置" value={motion} onChange={event => setMotionLevel(event.target.value as ReturnType<typeof readMotionLevel>)}><option value="full">完整动效</option><option value="reduced">减少动效</option><option value="off">关闭动效</option></select></label>
            <div className={`engine-status ${localCompute ? "online" : engine}`} role="status" aria-label={engineLabel} title={engineLabel}>
              <i />
              <span>{engineLabel}</span>
            </div>
            <button
              className="icon-button"
              aria-label="实验说明"
              onClick={() => about.current?.showModal()}
            >
              <CircleHelp size={17} />
            </button>
          </div>
        </header>
        <main className="main-content" id="main-content" tabIndex={-1}>
          {workspace === "overview" ? <LearningOverview
            currentLesson={{ name: lesson.name, title: lesson.title, description: lesson.description, index: active }}
            phase={phase} completed={completed} lessonIds={lessons.map(item => item.id)} effects={motion === "full"}
            onContinue={() => navigate(active)} onSelectPhase={navigate} onOpenTool={openTool}
          /> : workspace === "lessons" ? <>
          <section key={`${lesson.id}-${progressEpoch}`} className="lesson-workspace" id="experiment-workbench" tabIndex={-1} aria-label={`当前实验：${lesson.name}`}>
          <div className="lesson-header">
            <div>
              <div className="lesson-kicker">
                <span><span className="signal-dot" />EXPERIMENT {String(active + 1).padStart(2, "0")}</span>
                <i />
                {lesson.en}
              </div>
              <h1>{lesson.title}</h1>
              <p>{lesson.description}</p>
            </div>
            <div className="chapter-count">
              <strong>{String(active + 1).padStart(2, "0")}</strong>
              <span>/ {lessons.length}</span>
            </div>
          </div>
          <TraceRibbon key={`trace-${lesson.id}`} lessonId={lesson.id} />
          <div className="lesson-toolbar">
            <div className="concepts">
              <span className="small-label">本实验关键词</span>
              <span>{lesson.concept}</span>
            </div>
            <button
              className={`code-toggle ${showCode ? "is-selected" : ""}`}
              onClick={() => setShowCode(!showCode)}
              aria-expanded={showCode}
              aria-controls="lesson-code"
            >
              <Code2 size={16} />
              {showCode
                ? "收起代码"
                : lesson.id === "deployment"
                  ? "查看部署命令"
                  : "查看 Python 代码"}
            </button>
          </div>
          {showCode && (
            <section id="lesson-code" className="code-panel">
              <div>
                <span>
                  <Code2 size={15} />
                  对应计算代码
                </span>
                <small>
                  {lesson.id === "deployment"
                    ? "Shell / HTTP · 部署与接口"
                    : phase === 10
                      ? "Python / PyTorch + PEFT · 在线 GRPO"
                      : phase === 9
                      ? "Python / PyTorch + PEFT · 固定参考 DPO"
                      : phase === 8
                      ? "Python / PyTorch + PEFT · 回答监督 SFT"
                      : phase === 7
                      ? "Python / Transformers + PEFT · LoRA"
                      : phase === 6
                      ? "Python / PyTorch · 因果 Transformer"
                      : phase === 5
                      ? "Python / NumPy · 字符语言模型"
                      : phase >= 3
                      ? "Python / PyTorch · 核心计算代码"
                      : "Python / NumPy · 示例初始参数"}
                </small>
              </div>
              <pre>
                <code>{lesson.code}</code>
              </pre>
            </section>
          )}
          <div className="learning-prompt">
            <span>
              <Lightbulb size={18} />
            </span>
            <p>
              <strong>动手试一试</strong>
              {lesson.task}
            </p>
            {explored && (
              <span className="explored-tag">
                <Check size={13} />
                {learningProgress.tasks[lesson.id] ? "任务已达标" : "已开始探索"}
              </span>
            )}
          </div>
          <Suspense fallback={null}><PredictionPrompt key={`prediction-${lesson.id}`} lessonId={lesson.id} /></Suspense>
          {engine === "offline" && (
            <div className="connection-notice" role="alert">
              计算服务暂未连接。请在项目目录运行 <code>npm run dev</code>
              ，启动前端与 Python 服务。
            </div>
          )}
          <ExperimentBoundary key={`${lesson.id}-${progressEpoch}`}>
            <Suspense fallback={<div className="lesson-loading" role="status"><span className="signal-dot" />正在加载「{lesson.name}」实验…</div>}>
              <ExperimentEntry><Experiment key={`experiment-${lesson.id}`} onInteract={() => setExplored(true)} /></ExperimentEntry>
              <LessonTaskStatus lessonId={lesson.id} historicalCompleted={historicalCompleted} />
              <div className="lesson-study-layout"><LessonQuiz key={`quiz-${lesson.id}`} lessonId={lesson.id} />
              <LessonInsights key={`insights-${lesson.id}`} lessonId={lesson.id} /></div>
              <Glossary />
            </Suspense>
          </ExperimentBoundary>
          {allDone && (
            <div className="completion-message" role="status">
              <span>
                <CheckCheck size={24} />
              </span>
              <div>
                <h2>
                  {phase === 1
                    ? "你已完成第一阶段基础实验。"
                    : phase === 2
                      ? "你已完成第二阶段神经网络实验。"
                      : phase === 3
                        ? "你已完成第三阶段 MNIST 模型训练实验。"
                        : phase === 4 ? "你已完成第四阶段模型应用实验。" : phase === 5 ? "你已完成第五阶段语言模型实验。" : phase === 6 ? "你已完成第六阶段 Transformer 实验。" : phase === 7 ? "你已完成第七阶段 LoRA 微调实验。" : phase === 8 ? "你已完成第八阶段 SFT 指令微调实验。" : phase === 9 ? "你已完成第九阶段 DPO 偏好优化实验。" : "你已完成第十阶段 GRPO 强化学习微调实验。"}
                </h2>
                <p>
                  {phase === 1
                    ? "下一步，把这些计算连成网络，逐步理解反向传播。"
                    : phase === 2
                      ? "你已把前向计算、反向梯度与参数更新连接成一次真实训练。"
                      : phase === 3
                        ? "从数据划分、真实训练到独立测试，你已经完成模型训练与评估的完整流程。"
                        : phase === 4 ? "你已将最佳权重接入真实推理，连接手写输入、模型预测与应用部署。" : phase === 5 ? "你已把文本、下一字符预测、交叉熵训练与自回归生成连接起来。" : phase === 6 ? "你已连接因果注意力、真实训练、检查点与上下文文本生成。" : phase === 7 ? "你已完成预训练基座加载、低秩微调、冻结校验与适配器生成对比。" : phase === 8 ? "你已连接自定义问答、回答掩码、真实 SFT 与多轮对话生成。" : phase === 9 ? "你已连接自定义偏好对、固定参考、真实 DPO 与生成对照。" : "你已连接自定义奖励、在线采样、真实 GRPO 与前后对话对照。"}
                </p>
                <Suspense fallback={null}><PhaseReview lessonIds={phaseIndices.map(index => lessons[index].id)} /></Suspense>
                {phase === 1 && (
                  <button className="text-button" onClick={() => navigate(5)}>
                    进入第二阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 2 && (
                  <button className="text-button" onClick={() => navigate(8)}>
                    进入第三阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 3 && (
                  <button className="text-button" onClick={() => navigate(11)}>
                    进入第四阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 4 && (
                  <button className="text-button" onClick={() => navigate(13)}>
                    进入第五阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 5 && (
                  <button className="text-button" onClick={() => navigate(16)}>
                    进入第六阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 6 && (
                  <button className="text-button" onClick={() => navigate(19)}>
                    进入第七阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 7 && (
                  <button className="text-button" onClick={() => navigate(22)}>
                    进入第八阶段 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 8 && <button className="text-button" onClick={() => navigate(25)}>进入第九阶段 <ArrowRight size={14} /></button>}
                {phase === 9 && <button className="text-button" onClick={() => navigate(28)}>进入第十阶段 <ArrowRight size={14} /></button>}
              </div>
            </div>
          )}
          <footer className="lesson-footer">
            <button
              className="text-button"
              disabled={active === 0}
              onClick={() => navigate(active - 1)}
            >
              <ArrowLeft size={15} />
              上一实验
            </button>
            <span id="completion-guidance">
              {completed.includes(lesson.id) ? (
                <>
                  <Check size={14} />
                  本实验已完成
                </>
              ) : (
                canComplete ? "任务与自测已通过，可以完成本实验。" : "完成实验任务并答对本实验自测后，即可记录完成。"
              )}
            </span>
            <button
              className="primary-button"
              onClick={complete}
              aria-describedby="completion-guidance"
              title={canComplete ? "记录本实验学习进度" : "请先完成上方实验任务和本实验自测"}
            >
              {phaseEnd && !phaseReady ? "完成本实验并继续未完成实验" : phaseEnd
                ? phase === 1
                  ? "完成基础实验"
                  : phase === 2
                    ? "完成第二阶段实验"
                    : phase === 3
                      ? "完成第三阶段实验"
                      : phase === 4 ? "完成第四阶段实验" : phase === 5 ? "完成第五阶段实验" : phase === 6 ? "完成第六阶段实验" : phase === 7 ? "完成第七阶段实验" : phase === 8 ? "完成第八阶段实验" : phase === 9 ? "完成第九阶段实验" : "完成第十阶段实验"
                : "完成并继续"}
              <ArrowRight size={16} />
            </button>
          </footer>
          </section>
          <div className="page-caption">
            <span className="caption-brand"><Network size={12} />知微 AI LAB</span><span>MADE FOR CURIOUS MINDS</span><span>从一个数字开始，理解模型如何学习。</span>
          </div>
          </> : <section className="playground-workspace" id="experiment-workbench" tabIndex={-1} aria-label={selectedTool?.name}>
            <div className="playground-page-heading">
              <div><h1>{selectedTool?.name}</h1><p>{selectedTool?.description}</p></div>
              <button className="secondary-button" onClick={() => navigate(active)}><ArrowLeft size={15} />返回课程</button>
            </div>
            <nav className="playground-tabs" aria-label="切换探索工具">{explorationTools.map(tool => <button key={tool.id} aria-pressed={workspace === tool.id} onClick={() => openTool(tool.id)}>{tool.name}</button>)}</nav>
            <ExperimentBoundary key={workspace}>
            <Suspense fallback={<div className="tool-loading" role="status"><span className="signal-dot" />正在加载交互实验…</div>}>
            {workspace === "convolution" && <ConvolutionLab />}
            {workspace === "network-builder" && <NetworkBuilder onOpenLesson={id => { const index = lessons.findIndex(item => item.id === id); if (index >= 0) navigate(index); }} />}
            {workspace === "challenges" && <ChallengeHub onOpenTool={openTool} />}
            </Suspense>
            </ExperimentBoundary>
          </section>}
        </main>
      </div>
      <dialog ref={about} className="about-dialog" aria-labelledby="about-title">
        <div className="dialog-heading">
          <h2 id="about-title">关于知微</h2>
          <button
            className="icon-button"
            onClick={() => about.current?.close()}
            aria-label="关闭说明"
          >
            <X size={20} />
          </button>
        </div>
        <p>
          这是一间从零开始的 AI
          学习实验室。第一阶段理解数据与损失，第二阶段用计算图、反向传播和 XOR
          网络连接训练过程，第三阶段接入真实 MNIST 数据并训练 MLP 与
          CNN，第四阶段使用最佳权重完成手写识别与应用部署，第五阶段从字符分词开始，真实训练 Bigram 语言模型并生成文本，第六阶段进一步训练因果 Transformer，保存检查点并查看真实注意力，第七阶段使用公开预训练基座完成 LoRA 真实微调与生成对比，第八阶段使用自定义问答集完成回答监督 SFT 和多轮对话，第九阶段进一步用偏好数据执行真实 DPO 并对比优化前后的生成，第十阶段使用自定义奖励在线采样，执行真实 GRPO 强化学习微调。
        </p>
        <p>
          课程中的数值实验和模型训练由本机 Python、NumPy 与 PyTorch 真实计算。MNIST
          的训练、验证和测试数据严格分开，后台任务、训练记录与最佳权重保存在本机
          runs/mnist。第四阶段可直接使用 models/ 中导出的两份最佳权重，并提供
          Docker Compose 配置。XOR 模型保留在当前实验页面，可导出 JSON 或 CSV。
        </p>
        <p>
          探索工坊提供卷积实验、立体网络结构图与自动判定的挑战关卡。搭建器支持将自定义结构交给本机 PyTorch 真实训练，并接入独立测试评估与手写识别。
        </p>
        <p>
          完成实验任务并答对本实验自测后，可以点击「完成并继续」记录进度。任务、自测与原有完成记录保存在当前浏览器中，重置学习进度或清除网站数据后会清除。
        </p>
        <a
          className="secondary-button"
          href="/api/docs"
          target="_blank"
          rel="noreferrer"
        >
          查看本机 API 文档
          <ArrowRight size={15} />
        </a>
        <div className="record-actions"><button className="secondary-button" onClick={exportLearning}>导出学习记录（JSON）</button><label className="secondary-button">导入学习记录<input type="file" accept=".json,application/json" onChange={event => { const file = event.target.files?.[0]; if (file) void importLearning(file); event.currentTarget.value = ""; }} /></label></div>
        {recordNotice && <p role="status">{recordNotice}</p>}
        <button
          className="text-button reset-progress"
          onClick={() => {
            setCompleted([]);
            learningProgress.reset();
            setProgressEpoch(previous => previous + 1);
            navigate(0);
            about.current?.close();
          }}
        >
          重置学习进度
        </button>
      </dialog>
    </div>
  );
}
