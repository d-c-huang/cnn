import { useEffect, useRef, useState } from "react";
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
  X,
} from "lucide-react";
import DataExperiment from "./experiments/DataExperiment";
import LinearExperiment from "./experiments/LinearExperiment";
import NeuronExperiment from "./experiments/NeuronExperiment";
import LossExperiment from "./experiments/LossExperiment";
import GradientExperiment from "./experiments/GradientExperiment";
import { phase2Lessons } from "./lib/phase2Lessons";
import { phase3Lessons } from "./lib/phase3Lessons";
import { phase4Lessons } from "./lib/phase4Lessons";

const lessons = [
  {
    id: "data",
    name: "数据与张量",
    en: "DATA & TENSORS",
    title: "数据，是模型的起点。",
    description: "从一张小小的像素图开始，看看计算机眼中的图片是什么样子。",
    task: "选中一个像素，把灰度调到 128。打开归一化后，它会变成多少？",
    concept: "样本 · 特征 · 矩阵 · 归一化",
    component: DataExperiment,
    code: '# 灰度图片 → 归一化矩阵 → 输入向量\nimport numpy as np\n\n# pixels 是你在实验中编辑的 8×8 灰度数组\nmatrix = np.asarray(pixels, dtype=np.float64)\nnormalized = matrix / 255.0\nvector = normalized.ravel(order="C")\n\nprint(matrix.shape)      # (8, 8)\nprint(vector.shape)      # (64,)\nprint(normalized.mean()) # 转换后的均值',
  },
  {
    id: "linear",
    name: "线性模型",
    en: "LINEAR MODEL",
    title: "一个权重，就能改变预测。",
    description: "模型从最简单的直线开始。调整两个参数，看它如何接近真实数据。",
    task: "先只改变权重，再只改变偏置。试着让五个样本的误差都接近 0。",
    concept: "权重 · 偏置 · 预测 · 均方误差",
    component: LinearExperiment,
    code: "import numpy as np\n\nx = np.array([-2., -1., 0., 1., 2.])\ny = 2.0 * x + 0.5     # 构造教学样本\n\nweight, bias = 0.5, 0.0\nprediction = weight * x + bias\nerror = prediction - y\nmse = np.mean(error ** 2)\n\nprint(prediction)\nprint(mse)",
  },
  {
    id: "neuron",
    name: "单个神经元",
    en: "A SINGLE NEURON",
    title: "看看一个神经元，如何计算。",
    description: "输入经过加权求和与激活函数，得到输出。让每一个计算都看得见。",
    task: "把一个权重改成负数，再增大对应输入。观察加权和与最终输出如何变化。",
    concept: "加权求和 · 激活函数 · 前向计算",
    component: NeuronExperiment,
    code: "import numpy as np\n\ninputs = np.array([0.6, 0.9])\nweights = np.array([0.8, -0.4])\nbias = 0.2\n\nproducts = inputs * weights\nz = products.sum() + bias\noutput = np.maximum(0.0, z)  # ReLU\n\n# 其他激活函数\nsigmoid = np.exp(-np.logaddexp(0.0, -z))\ntanh = np.tanh(z)",
  },
  {
    id: "loss",
    name: "损失函数",
    en: "LOSS FUNCTION",
    title: "模型怎么知道，自己错了？",
    description: "用损失衡量预测与真实答案的距离，再用梯度找到改进的方向。",
    task: "让预测值与真实标签相等，再分别向两边移动相同距离，比较损失和梯度。",
    concept: "平方误差 · 损失曲线 · 梯度",
    component: LossExperiment,
    code: "import numpy as np\n\nprediction, target = -1.0, 2.0\n\n# 单样本平方误差\nloss = (prediction - target) ** 2\n\n# 损失对预测值的导数\ngradient = 2.0 * (prediction - target)\n\nprint(loss)      # 9.0\nprint(gradient)  # -6.0",
  },
  {
    id: "gradient",
    name: "梯度下降",
    en: "GRADIENT DESCENT",
    title: "学习，就是一次次微小的更新。",
    description: "让模型自动调整权重。亲眼观察学习率如何决定它走向哪里。",
    task: "先用学习率 0.1 运行，再重置并试试 0.5 和 0.55。记录你看到了什么。",
    concept: "学习率 · 梯度 · 参数更新 · 收敛",
    component: GradientExperiment,
    code: "import numpy as np\n\nx = np.array([-2., -1., 0., 1., 2.])\ny = 2.0 * x + 0.5\nweight, bias, learning_rate = -2.0, 0.5, 0.1\n\nfor step in range(20):\n    error = weight * x + bias - y\n    loss = np.mean(error ** 2)\n    gradient = 2.0 * np.mean(error * x)\n    weight -= learning_rate * gradient\n    print(step, weight, loss)\n\n# b 固定；每一步使用全部 5 个教学样本",
  },
  ...phase2Lessons,
  ...phase3Lessons,
  ...phase4Lessons,
];
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
  const [active, setActive] = useState(() => {
    const hash = window.location.hash.slice(1);
    const saved = readProgress().active;
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
  const [engine, setEngine] = useState<"connecting" | "online" | "offline">(
    "connecting",
  );
  const [allDone, setAllDone] = useState(false);
  const [storageWarning, setStorageWarning] = useState(false);
  const about = useRef<HTMLDialogElement>(null);
  const content = useRef<HTMLElement>(null);
  const lesson = lessons[active];
  const Experiment = lesson.component;
  const phase = active < 5 ? 1 : active < 8 ? 2 : active < 11 ? 3 : 4;
  const phaseIndices =
    phase === 1
      ? [0, 1, 2, 3, 4]
      : phase === 2
        ? [5, 6, 7]
        : phase === 3
          ? [8, 9, 10]
          : [11, 12];
  const phaseComplete = phaseIndices.every((index) =>
    completed.includes(lessons[index].id),
  );
  const phaseEnd =
    active === 4 || active === 7 || active === 10 || active === 12;

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
    const handleHash = () => {
      const index = lessons.findIndex(
        (l) => l.id === window.location.hash.slice(1),
      );
      if (index >= 0) {
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
    setActive(index);
    setExplored(false);
    setShowCode(false);
    setAllDone(false);
    window.history.replaceState(null, "", `#${lessons[index].id}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function complete() {
    setCompleted((previous) => [...new Set([...previous, lesson.id])]);
    if (phaseEnd) setAllDone(true);
    else navigate(active + 1);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#data"
          onClick={(e) => {
            e.preventDefault();
            navigate(0);
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
            <small>从零训练一个 AI</small>
          </span>
        </a>
        <div className="course-label">
          <span className="eyebrow">YOUR LEARNING PATH</span>
          <p>从理解，到创造。</p>
        </div>
        <div className="phase-switcher" role="group" aria-label="选择学习阶段">
          <button aria-pressed={phase === 1} onClick={() => navigate(0)}>
            01 基础
          </button>
          <button aria-pressed={phase === 2} onClick={() => navigate(5)}>
            02 网络
          </button>
          <button aria-pressed={phase === 3} onClick={() => navigate(8)}>
            03 MNIST
          </button>
          <button aria-pressed={phase === 4} onClick={() => navigate(11)}>
            04 应用
          </button>
        </div>
        <div className="phase-label">
          <span>0{phase}</span>
          <strong>
            {phase === 1
              ? "基础可视化实验"
              : phase === 2
                ? "计算图与神经网络"
                : phase === 3
                  ? "MNIST 模型训练"
                  : "模型推理与部署"}
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
                className={`lesson-link ${active === i ? "active" : ""}`}
                aria-current={active === i ? "step" : undefined}
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
                {active === i && (
                  <ChevronRight className="nav-chevron" size={15} />
                )}
              </button>
            );
          })}
        </nav>
        <div className="future-phases">
          <span className="eyebrow">THE COMPLETE LEARNING LOOP</span>
          <div>
            <span>↗</span>
            <p>数据 → 训练 → 评估 → 推理</p>
            <CheckCheck size={13} />
          </div>
          <small>四个阶段 · 从理解到应用</small>
        </div>
        <div className="sidebar-bottom">
          <div className="progress-heading">
            <span>你的探索进度</span>
            <strong>
              {completed.length}
              <em> / {lessons.length}</em>
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
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <FlaskConical size={16} />
            <span>
              {phase === 1
                ? "基础实验"
                : phase === 2
                  ? "神经网络"
                  : phase === 3
                    ? "MNIST 训练"
                    : "模型应用"}
            </span>
            <ChevronRight size={13} />
            <strong>{lesson.name}</strong>
          </div>
          <div className="topbar-actions">
            <div className={`engine-status ${engine}`} role="status">
              <i />
              <span>
                {engine === "online"
                  ? phase >= 3
                    ? "API 已连接"
                    : "NumPy 引擎已连接"
                  : engine === "offline"
                    ? "计算服务未连接"
                    : "连接计算引擎…"}
              </span>
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
        <main className="main-content" ref={content} id="main-content">
          <div className="lesson-header">
            <div>
              <div className="lesson-kicker">
                <span>EXPERIMENT {String(active + 1).padStart(2, "0")}</span>
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
          <div className="lesson-toolbar">
            <div className="concepts">
              <span className="small-label">本节关键词</span>
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
                已开始探索
              </span>
            )}
          </div>
          {engine === "offline" && (
            <div className="connection-notice" role="alert">
              计算服务暂未连接。请在项目目录运行 <code>npm run dev</code>
              ，启动前端与 Python 服务。
            </div>
          )}
          <Experiment key={lesson.id} onInteract={() => setExplored(true)} />
          {allDone && (
            <div className="completion-message" role="status">
              <span>
                <CheckCheck size={24} />
              </span>
              <div>
                <h2>
                  {phase === 1
                    ? "你已完成第一期基础实验。"
                    : phase === 2
                      ? "你已完成第二期神经网络实验。"
                      : phase === 3
                        ? "你已完成第三期 MNIST 模型训练实验。"
                        : "你已完成第四期模型应用实验。"}
                </h2>
                <p>
                  {phase === 1
                    ? "下一步，把这些计算连成网络，逐步理解反向传播。"
                    : phase === 2
                      ? "你已把前向计算、反向梯度与参数更新连接成一次真实训练。"
                      : phase === 3
                        ? "从数据划分、真实训练到独立测试，你已经完成模型训练与评估的完整流程。"
                        : "你已将最佳权重接入真实推理，连接手写输入、模型预测与应用部署。"}
                </p>
                {phase === 1 && (
                  <button className="text-button" onClick={() => navigate(5)}>
                    进入第二期 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 2 && (
                  <button className="text-button" onClick={() => navigate(8)}>
                    进入第三期 <ArrowRight size={14} />
                  </button>
                )}
                {phase === 3 && (
                  <button className="text-button" onClick={() => navigate(11)}>
                    进入第四期 <ArrowRight size={14} />
                  </button>
                )}
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
            <span>
              {completed.includes(lesson.id) ? (
                <>
                  <Check size={14} />
                  本实验已完成
                </>
              ) : (
                "理解一点，动手一点。"
              )}
            </span>
            <button
              className="primary-button"
              onClick={complete}
              disabled={engine !== "online"}
            >
              {phaseEnd
                ? phase === 1
                  ? "完成基础实验"
                  : phase === 2
                    ? "完成第二期实验"
                    : phase === 3
                      ? "完成第三期实验"
                      : "完成第四期实验"
                : "完成并继续"}
              <ArrowRight size={16} />
            </button>
          </footer>
          <div className="page-caption">
            知微 AI LAB <span>·</span> 从一个数字开始，理解模型如何学习。
          </div>
        </main>
      </div>
      <dialog ref={about} className="about-dialog">
        <div className="dialog-heading">
          <h2>关于知微</h2>
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
          学习实验室。第一期理解数据与损失，第二期用计算图、反向传播和 XOR
          网络连接训练过程，第三期接入真实 MNIST 数据并训练 MLP 与
          CNN，第四期使用最佳权重完成手写识别与应用部署。
        </p>
        <p>
          实验结果由本机 Python、NumPy 与 PyTorch 真实计算。MNIST
          的训练、验证和测试数据严格分开，后台任务、训练记录与最佳权重保存在本机
          runs/mnist。第四期可直接使用 models/ 中导出的两份最佳权重，并提供
          Docker Compose 配置。XOR 模型保留在当前实验页面，可导出 JSON 或 CSV。
        </p>
        <p>
          点击「完成并继续」记录学习进度。进度只保存在当前浏览器中，清除网站数据后会重置。
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
        <button
          className="text-button reset-progress"
          onClick={() => {
            setCompleted([]);
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
