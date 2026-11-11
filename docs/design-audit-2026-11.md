# 知微 AI LAB — 内容 / UI / 特效优化方案

审计时间：2026-11-11　对象：`src/` 全量（13 节课程、3 个探索工具、9852 行 CSS）

---

## 0. 先给三条判断

**第一，这个项目的内容质量远高于它的界面质量。** `lessonInsights.ts` 和 `lessonDeepDives.ts` 各 14 条，每条都有学习目标、三步原理、公式对照、常见误区、动手观察的"操作 → 预期"对照，并且反复声明"教学计算不是模型输出""单轮波动不能证明过拟合"。这种诚实度在同类教学项目里很少见。现在的界面没有把这部分价值传达出来——它被放在每节最底部，用和其他面板一样的圆角卡片装起来。

**第二，视觉系统是分裂的，而且分裂得很具体。** `design.css` 声明了 `--accent: #38BDF8`（青蓝）和 `--support: #6EE7B7`（翡翠绿），但实际渲染出的界面是青绿 `#62d7bc` 配柠檬绿 `#c9ef6b`。全站 **1568 个不同的十六进制色值**，`--text-muted` 用在 19 处、对比度 **2.59:1**（WCAG AA 要求 4.5:1）。这不是"需要更多设计"，是"需要一次清算"。

**第三，动效几乎不存在，但"动效开关"已经做出来了。** 全站只有 7 个 `@keyframes`，其中 4 个在 `neural-scene.css` 里——而 `NeuralScene.tsx` 这个组件**从未被任何文件 import**。开关的实现是 `animation-play-state: paused !important`，会把节点冻在半亮状态，比没有动画更奇怪；而且 `effects` 状态不持久化，刷新就回默认开。

---

## 1. 视觉识别：从"暗色 SaaS 卡片"改成"计算示波器"

### 1.1 现状问题

现在的界面落在两个 AI 生成设计的典型套路里：近黑背景 + 单一酸性绿强调色，以及 SaaS 卡片工具包（统一圆角、统一柔和阴影、**几乎每个区块都是一张卡片**）。`.lo-stage`、`.lo-tool`、`.experiment-panel`、`.control-panel`、`.insight`、`.builder-layer-card`——它们用同一种语法说话，于是页面没有层次，只有重复。

更麻烦的是颜色本身在干扰数据。页面底色带绿调（`#0A0D12` 实际渲染出偏绿的 `#101e22`、`#10211c`），而实验区要画**卷积核响应、特征图、混淆矩阵、MNIST 手写热力图**。在有绿偏的中性底上看绿-黄渐变的热力图，人对色相的判断会被拉偏。`experiment-theme.css` 开头自己写了注释「Keep sample pixels, heatmaps and drawing data unaltered.」——说明作者意识到了这个问题，但底色没跟上。

### 1.2 建议

这个学科里最本质的东西是**方向**：前向传播把信号送进去，反向传播把梯度带回来。整套 13 节课就是围绕这条双向通路组织的。所以我建议把强调色从"一种颜色"改成"一对极性"，让颜色承载这个结构：

| 语义 | 色值 | 在 `#0B1017` 上的对比度 | 用途 |
|---|---|---|---|
| 前向 forward | `#4FD1E0` | 10.47:1 | 输入、激活、预测、正向脉冲 |
| 反向 backward | `#E8B563` | 10.19:1 | 梯度、损失回传、参数更新 |
| 达成 / 通过 | `#57D9A3` | 10.77:1 | 任务达标、自测正确、检查点保存 |
| 告警 | `#E8B563` | 10.19:1 | 连接失败、过拟合提示 |
| 错误 | `#F0736F` | 6.71:1 | 校验失败 |

底色改为**去绿的中性冷色**，让热力图、矩阵、手写笔迹的色相不被页面污染：

```css
:root {
  color-scheme: dark;
  /* 去绿中性底：数据可视化优先，页面不参与染色 */
  --surface-0: #070A0F;
  --surface-1: #0B1017;
  --surface-2: #111925;
  --surface-3: #17212F;
  --surface-4: #1F2B3B;

  --text-1: #EAF2F8;   /* 16.86:1 */
  --text-2: #A9C0D4;   /* 10.15:1 */
  --text-3: #7C93A8;   /*  5.99:1 —— 取代现 --text-muted 的 2.59:1 */

  --fwd: #4FD1E0;
  --bwd: #E8B563;
  --ok:  #57D9A3;
}
```

柠檬绿 `#c9ef6b` 不作为界面色，只保留给**输出层节点**和进度条末端（在 `NeuralScene` 里它本来就是这个角色）。

同时要在 `.experiment-panel` 上加一条明确规则：**实验区内部不使用任何页面级渐变**，只用 `--surface-2` 平涂加 1px 边框。让"仪器"和"仪表盘外壳"在材质上分开——这是唯一需要贯彻的层次手段，比加阴影有效。

### 1.3 字体

现在是 `"Segoe UI", "Microsoft YaHei UI"`。中文正文在 14px × 1.12 的 YaHei 下不算差，但整个产品**没有一个统一的数字视觉**，而这恰恰是个满屏数字的产品：权重、损失、梯度、参数量、置信度、坐标。数字在各处跳变时宽度不一致，会左右抖动。

建议三件事，按重要性排序：

1. **数字一律 `font-variant-numeric: tabular-nums`** 并统一用等宽族（`"JetBrains Mono", "Cascadia Code", Consolas`）。目前只有 19 处用了 numeric 相关属性，覆盖不全。
2. **中文改用可变字重的思源黑体子集**（Noto Sans SC Variable，只取 400/500/600 + 常用 3500 字，woff2 约 300–400KB）。现用系统 YaHei 的字重只有 400/700 两档，导致 `.h1 { font-weight: 550 }` 这类中间字重在 Windows 上被量化成 700，标题显得过重——代码里大量 550/650 的字重其实都没生效。
3. **字号级差重排**。现在 `学习总览` 的 h1 是 `clamp(26px, 2.25vw, 40px)`，而最常驻的**课程页 h1 只有 22–28px**。用户 95% 的时间在课程页，标题却是全站最小的。改成 `32px / 38px` 两档，并把 `insight` 的正文从 14px 提到 15px——中文在小字号下的可读性损失比拉丁文大。

### 1.4 版式与行宽

`--content-width: 1720px`，正文 14px×1.12 ≈ 15.7px。在 1720px 宽的内容区里，一行能排下**约 110 个西文等宽字符 / 55 个汉字**，远超 80 字符的可读上限。`.lesson-header p` 有 `max-width: 80ch`，但 `insight-steps`、`deep-dive-steps` 没有——而这两处才是真正要读的段落。

建议把内容区改成**双栏结构而不是单栏加宽**：

```
┌──────────────────────────────────────────────────────────┐
│  trace ribbon:  数据 → 前向 → 损失 → 梯度 → 更新          │  ← 常驻，当前节点高亮
├───────────────────────────────────────┬──────────────────┤
│  实验区（全宽，随窗口伸展）            │  控制面板 320px   │
│  ─────────────────────────────────    │  固定栏           │
│  52ch 正文测量宽度的讲解               │                  │
├───────────────────────────────────────┴──────────────────┤
│  原理（68ch）│ 自测（320px）                              │  ← 两栏，不再上下堆
└──────────────────────────────────────────────────────────┘
```

内容区最大宽度收到 **1440px**，超出部分留白。正文测量宽度硬约束 `68ch`（中文约 34 字），左对齐，不做两端对齐（中英混排两端对齐会产生难看的字间距）。

### 1.5 断点

现在有 **19 个不同的 max-width 断点**（400 / 430 / 520 / 600 / 650 / 700 / 760 / 850 / 900 / 1000 / 1020 / 1040 / 1050 / 1100 / 1150 / 1200 / 1250 / 1450），外加 10 个 min-width。这些数字没有规律，说明是逐次打补丁加的。

收敛为 5 档：`430 / 640 / 900 / 1200 / 1600`。实验区内部（图、矩阵、画布）改用 **container query**——`graph.css` 已经在用 `@container (max-width: 680px)`，这是对的，应该推广到所有实验面板，因为实验区的可用宽度取决于侧栏能不能收起来，而不只是视口宽度。

---

## 2. 内容：把"看懂"变成"先预判"

内容的骨架很好，缺的是**学习闭环**。具体四个缺口：

### 2.1 没有"先预测，再运行"

这是交互式教学里收益最高的一步，而这个项目天然适合：每个实验都有一个"运行/更新/训练"按钮，运行前让用户先押一个判断。

在 `GradientExperiment`，按下"执行更新"前问一句：*损失会下降、上升，还是振荡？* 用户点了才跑，跑完把预测和实际并排显示。在 `ConvolutionLab` 扫描前问：*这个位置的中心值会是正还是负？*

实现成本很低——一个 `prediction: "up" | "down" | "unknown"` 的局部 state，加一段对比文案。但它把页面从"看演示"变成"做实验"，也直接消解了现在最大的体验问题：**用户只要点几下任何按钮就能通过任务，检测不到理解**。`advancedTaskChecks.ts` 里有数值判定，但多数实验的 `onInteract` 一触发就算"已开始探索"。

### 2.2 自测只有一道题，而且位置最差

14 节每节 1 道三选一，放在页面最底部、深处 dive 之后。误答的处理是显示正确答案的解释。

三个改动：

- **移到动手任务下方**，不要等读完长文才答。原理正文可以折叠，自测不能。
- **加一道迁移题**：换一组数字、换一个方向。现在这道题考的是"记住"，第二道考"会用"。例如 `neuron` 现在问的是激活函数选择，迁移题可以问"同样把 w₁ 调成负数，Sigmoid 和 ReLU 哪个输出会更早变成 0"。
- **错项级反馈**。`lessonInsights` 的 quiz 类型是 `answer: 0|1|2` + 一个统一 `explanation`。改成每个错项一条 `whyWrong`，因为三选一的错误答案往往对应三种不同的误解，用一句话同时解释三种会变成什么都没解释。

### 2.3 术语散落，没有汇总

`lessonDeepDives.ts` 里 31 个术语，全部内嵌在各自课程中，"饱和"只出现在 `neuron`，"检查点"只出现在 `mnist-train`。学到第 11 节想回头查"什么是分层抽样"，只能翻回第 9 节。

建议建一个 `glossary.ts`，把这 31 个术语提出来做独立索引（按首字母 + 按出现顺序两种排法），课程正文里的术语做 `[[饱和]]` 形式的轻量跳转。这是纯数据整理，不需要新计算。

### 2.4 阶段结束只有一段鼓励文案

`App.tsx` 里 `allDone` 渲染的是四段固定文本加一个"进入下一期"按钮。建议改成**阶段回顾**：列出本阶段 3–5 节课的核心概念、你亲手调过的关键参数、以及"这一阶段最容易混淆的两组概念"（这个数据 `lessonInsights.misconception` 里已经有了，只是没被聚合）。

另外进度只存在 localStorage 的两个 key（`zhiwei-foundation-v1`、`zhiwei-learning-evidence-v1`），**没有导出**。加一个"导出学习记录（JSON）/ 导入"，让教师能收作业、学生能换机器。这是小改动，但对"能不能真的在课堂上用"是决定性的。

### 2.5 一处内容修正

`README.md` 开头写「四期共 13 个实验」，实际代码 5+3+3+2 = 13，一致。但 `index.html` 的 description 与 `about` 弹窗里的表述分别是「13 个交互实验」和「第一期…第四期」——建议统一术语：**阶段**用于集合，**实验**用于单节，现在"期/阶段/章/节"混用（`phase-label`、`chapter-count`、`lesson`、`experiment` 四处不同叫法）。

---

## 3. UI 与可访问性：可核对的具体缺陷

这些不是审美问题，是能直接用工具验证的：

| 缺陷 | 证据 | 处理 |
|---|---|---|
| 正文对比度不达标 | `--text-muted: #4A5568` 在 `#0A0D12` 上 **2.59:1**，用于 19 处 | 并入 `--text-3`（5.99:1）或删除该 token |
| 浅色 token 遗留且不可达 | `styles.css` 根节点仍声明 `background: #f6f7f3; color: #282d27`，被 `design.css` 的 `body` 覆盖，`--muted: #7b8177` 对比度 2.85:1 | 整体删除；或正式升级为"日间阅读模式"（见 3.2） |
| 死代码 | `design.css` 中 **24 个顶层类名在 TSX 里没有任何出处**：`.lab-hero` `.hero-copy` `.hero-facts` `.roadmap-card` `.roadmap-grid` `.corner-tl` `.corner-br` `.insight-quiz-heading` 等 | 删除，约 400 行 |
| 死组件 | `NeuralScene.tsx`（135 行）+ `neural-scene.css`（4 个 keyframes）从未被 import | 复用或删除（见第 4 节，我建议**复用**） |
| z-index 无体系 | 现有值：`1 / 9 / 10 / 50 / 100 / 1000` | 建 token：`--z-base/--z-sticky/--z-overlay/--z-modal` |
| 圆角无层级 | `--radius-sm/md/lg/full` 中 `--radius-lg` 只被用 4 次，实际文件里到处是 `7px / 9px / 10px / 13px / 14px` | 固定三档：`6px`（控件）/ `10px`（面板）/ `999px`（胶囊），其余清除 |
| 字号倍率与 80ch 冲突 | `--reading-scale` 三档 1 / 1.12 / 1.25 乘在 `14px` 上，但行宽没有跟着变 | `max-width: calc(68ch / var(--reading-scale))` |
| 进度条缺数值标签 | `--z-index` 无关；`.lo-progress` 是纯装饰条，`aria-valuenow` 有但可见文本只在侧栏 | 概览页进度条右侧补百分比文本（现在有 `{progress}%`，但侧栏的没有） |

### 3.1 完成按钮的禁用理由不可见

`App.tsx:583` 的主按钮 `disabled={!canComplete}`，理由只写在 `title` 里。触屏和键盘用户永远看不到。改成按钮上方常驻一行状态（现有 `<span>` 已有文案，但按钮同时 disabled，用户以为卡死）。更好的做法：**按钮不 disabled，点击后滚动到缺失的任务/自测并高亮**。

### 3.2 关于那套浅色 token

`styles.css` 里那套 `#f6f7f3 / #282d27` 不是垃圾，它是一套完整的浅色主题，只是被覆盖了。有两个选择：删掉，或者**认真做成"阅读模式"**——因为 `lessonDeepDives` 里那些长段落，在深色底上读 3000 字确实累。

如果做，我建议不用纯白，用 `#F2F4F1` 底 + `#1A1F1C` 字，并且**只在"原理"板块生效，实验区永远保持深色**。理由是实验区里的画布、热力图、曲线都是按深色底设计的，切浅色要重算所有配色，成本远大于收益。这个"半明半暗"的选择本身也是识别度：**讲解可以亮，仪器必须暗。**

---

## 4. 特效：一次编排，其余全部克制

### 4.1 先把开关的语义修好

现在：

```css
.effects-off *, .effects-off *::before, .effects-off *::after {
  animation-play-state: paused !important;
  transition-duration: 0s !important;
}
```

`animation-play-state: paused` 会让 `neural-node-breathe` 停在动画的任意一帧——节点亮度是悬在中间值的，而且 `neural-signal-flow` 的光点会**冻结在半路上**。关掉动效的正确效果应该是"没有动效"，不是"动效卡住"。

改成三档，并**持久化**，且默认读系统偏好：

```ts
type MotionLevel = "full" | "reduced" | "off";
// 默认：matchMedia("(prefers-reduced-motion: reduce)").matches ? "reduced" : "full"
// 存 localStorage，和 --reading-scale 同级
```

```css
:root[data-motion="reduced"] * { animation-duration: .001ms !important; animation-iteration-count: 1 !important; }
:root[data-motion="off"] *, :root[data-motion="off"] *::before, :root[data-motion="off"] *::after {
  animation: none !important; transition: none !important;
}
```

`reduced` 保留状态变化（按钮按下、面板展开）但去掉所有循环和位移；`off` 全关。`prefers-reduced-motion` 直接映射到 `reduced`，而不是 `off`——把无障碍偏好当成"什么都不许动"是过度响应。

### 4.2 唯一的一次编排：前向脉冲

现在页面进入任何一节实验都是硬切。建议做**且只做一次**编排，用上那个被废弃的 `NeuralScene`：

进入实验的 600ms 内，顶部 trace ribbon（第 1.4 节的常驻条）上的节点从左到右依次点亮（每格 90ms），一道光沿连接线走完全程，同时实验面板以 `opacity 0→1 / translateY(6px)→0` 淡入，**整块面板一次，不是每张卡片各自淡入**。在 `backprop`、`xor`、`mnist-train` 三节，脉冲走完后反向再来一道琥珀色——因为这三节的主题就是梯度回传。

响应 `prefers-reduced-motion` 时直接显示终态。这一处做扎实，比给 40 个组件加 hover 上浮有价值得多。

### 4.3 数值补间（最该做的一项）

这是功能性动效，不是装饰。现在所有读数都是硬跳：损失从 `0.4021` 直接变 `0.3897`，权重从 `−2.00` 变 `−0.40`。用户看不出"变了多少"，只看得出"变了"。

```ts
function useTween(target: number, duration = 180) {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  const start = useRef(0);
  useEffect(() => {
    if (prefersReducedMotion) { setValue(target); return; }
    from.current = value; start.current = performance.now();
    let raf = requestAnimationFrame(function tick(now) {
      const t = Math.min(1, (now - start.current) / duration);
      const e = 1 - Math.pow(1 - t, 3);           // ease-out cubic
      setValue(from.current + (target - from.current) * e);
      if (t < 1) raf = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return value;
}
```

配套：所有数值统一 `toFixed(4)` + `tabular-nums`，容器宽度用 `ch` 固定，避免补间时整行重排。适合接入的地方：`GradientExperiment` 的损失、`OptimizerComparison` 的轨迹坐标、`MnistTrainingExperiment` 的准确率、`PredictionBars` 的置信度。

### 4.4 扫描器改成时间驱动

`ConvolutionLab.tsx:65` 用的是 `setInterval(..., 700)`。问题：不可变速、暂停会丢掉相位、快速点击"下一个"会和定时器打架。

改成 rAF + 时间累加，暴露一个速度档（0.5× / 1× / 2×），暂停时保存已累积时间。同时给 3×3 核窗口加一条**从核元素到乘积项的细连线**，扫描移动时连线跟着走——这让"九个乘积怎么加起来的"这件事在视觉上成立，而现在它只是两张并排的表。

### 4.5 梯度流向要用一次性的"到达"

`neural-upgrade.css` 里 `.xv-signal-stream` 是 `animation: xor-signal 3.8s linear infinite`，无限循环的虚线滚动。无限循环的流动表示的是"一直在传"，但梯度计算的语义是**每一轮算完就停**。

建议：反向时用 400ms 的 `stroke-dashoffset` 过渡让虚线走完一遍，走完触发一次 `stroke` 的短促增亮（120ms），然后静止。用户点"应用更新"时再触发一次。**一次性动效表达"事件"，循环动效只能表达"氛围"**——这个页面上大部分动效都该是事件型。

### 4.6 明令不做的事

写进规范，避免后续反复：

- 不做卡片 hover 上浮 + 阴影加深（现在 `.lo-stage`、`.lo-tool` 都有，改成只变 `border-color` 和 `background`）。
- 不加无限循环的呼吸光/粒子漂浮（`neural-scene.css` 现有 4 个循环动画，压缩到 1 个，且只放在总览页）。
- 不做滚动进场（`fade-and-slide-up on scroll` 是 AI 生成页面的最强标识，而且这个产品的滚动是"读长文"，不是"浏览营销页"）。
- 不给按钮文字加 `→`。
- 不在每个标题上加全大写小型 eyebrow。现在 `.lo-eyebrow`、`.insight-eyebrow`、`.playground-eyebrow`、`.lesson-kicker` 都在做这件事，四个不同名字做同一件事，且大写字母对中文读者不传递任何信息。保留 **lesson-kicker 一处**（它标的是"EXPERIMENT 03"这种真的有序列含义的信息），其余删掉。

统一动效 token：

```css
--ease-out: cubic-bezier(.2, .8, .2, 1);
--dur-1: 120ms;  /* 状态反馈：按下、选中、聚焦 */
--dur-2: 220ms;  /* 面板展开、颜色过渡 */
--dur-3: 420ms;  /* 图表重绘、数值变化 */
--dur-hero: 600ms; /* 唯一允许的编排时长，一节实验只播一次 */
```

---

## 5. 执行顺序

按"改一处、全站受益"排序，前三项是纯清理，风险低、收益立刻可见。

| 顺序 | 事项 | 涉及文件 | 工作量 | 依赖 |
|---|---|---|---|---|
| 1 | 清死代码：24 个死类名、`styles.css` 浅色根、z-index/圆角收敛 | `design.css` `styles.css` `app-layout.css` | 0.5 天 | 无 |
| 2 | 色板换血：去绿中性底 + 前向/反向双极性 + `--text-muted` 并档 | `design.css` `experiment-theme.css` `learning-overview.css` | 1 天 | 1 |
| 3 | 对比度与行宽：68ch 硬约束、字号级差重排、lesson h1 提级 | `app-layout.css` `readability.css` 各 `*.css` | 1 天 | 2 |
| 4 | 动效档位：三档开关 + 持久化 + `prefers-reduced-motion` 映射 | `App.tsx` `design.css` | 0.5 天 | 无 |
| 5 | 数值补间 hook + 接入四处读数 | 新 `lib/useTween.ts` + 4 个实验 | 1 天 | 4 |
| 6 | trace ribbon 常驻条 + 600ms 前向脉冲编排 | `App.tsx` 新 `components/TraceRibbon.tsx` | 2 天 | 2, 4 |
| 7 | 断点收敛 19 → 5，实验区改 container query | 全部 CSS | 2 天 | 3 |
| 8 | 字体：思源黑体子集 + 统一 tabular-nums | `index.html` `design.css` | 0.5 天 | 无 |
| 9 | 内容：预判步骤（选 3 个实验试点） | `GradientExperiment` `ConvolutionLab` `NeuronExperiment` | 2 天 | 无 |
| 10 | 内容：自测上移 + 迁移题 + 错项级反馈 | `lessonInsights.ts` `LessonInsights.tsx` | 2 天 | 无 |
| 11 | 内容：`glossary.ts` 汇总 31 个术语 + 正文跳转 | 新 `lib/glossary.ts` | 1 天 | 无 |
| 12 | 内容：阶段回顾页 + 学习记录导出/导入 | `App.tsx` `learningProgress.tsx` | 1.5 天 | 10 |
| 13 | 扫描器 rAF 化 + 核到乘积项连线 | `ConvolutionLab.tsx` | 1 天 | 5 |
| 14 | 复用 `NeuralScene` 到总览页 hero（替换现在的纯文字欢迎区） | `LearningOverview.tsx` | 1 天 | 2, 6 |

总计约 **17 个工作日**。如果只能做一周，做 1、2、3、4、5 五项——它们解决的是"系统不一致"和"读起来累"，是用户每天都会碰到的问题。

---

## 6. 验收标准

- 全站十六进制色值从 1568 降到 **≤ 40**（全部经 `:root` token 引用），且 CSS 里不再出现裸色值。
- 所有正文与次要文本对比度 **≥ 4.5:1**；`#4A5568` 这类值在代码里不存在。
- `@media` 断点数量 **= 5**；z-index 只出现 token 名。
- 删除死代码后 `design.css` 行数下降 ≥ 15%，`dist/assets/*.css` 从 324KB 降到 ≤ 270KB。
- 关闭动效后，页面上**没有任何元素停在动画中间帧**（用 devtools 的 "Emulate prefers-reduced-motion" 逐页检查）。
- 每节实验进入时，动效只播一次，不循环；离开再进入才重播。
- 每节课都有：预判步骤、≥ 2 道自测（含一道迁移题）、错项级反馈。
- 一个从未见过的用户，能在不读 README 的情况下，从总览页找到"当前该做什么"并进入实验（现在靠侧栏的"当前阶段"和一个 resume 卡片，但两处信息不同步）。

---

## 附：关于"要不要重做"

不建议重做。这个项目的计算层（`lib/neuron.ts`、`lib/loss.ts`、`lib/optimizers.ts`、`lib/convolution.ts`，以及 `scripts/test-*.mjs` 一整套测试）和内容层是真正的资产，界面是资产外面套了一层不匹配的壳。上面 14 项全部做完，等于换了一次壳，而不动一次计算。

唯一的例外是**布局结构**（1.4 节的双栏 + trace ribbon）。这个改动会波及每个实验页的外层容器，但收益最大——它把"这是一串教学卡片"变成"这是一台正在运行的仪器"。如果预算紧张，我愿意用第 6、7、14 项换它。
