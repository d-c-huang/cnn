export type Optimizer = "sgd" | "momentum" | "adagrad" | "rmsprop" | "adam";
export type Objective = "quadratic" | "valley";
export type Vector = [number, number];
export type OptimizerConfig = { objective: Objective; start: Vector; steps: number; learning_rate: number; rates: Partial<Record<Optimizer, number>> };
export type OptimizerStep = { step: number; position: Vector; loss: number; gradient: Vector | null; update: Vector | null; first: Vector; second: Vector };
export type OptimizerRun = { optimizer: Optimizer; learning_rate: number; status: "completed" | "limit"; first_hit: number | null; history: OptimizerStep[] };
export type ComparisonResult = { config: OptimizerConfig; threshold: number; optimum: Vector; runs: OptimizerRun[] };

export const optimizerNames: Optimizer[] = ["sgd", "momentum", "adagrad", "rmsprop", "adam"];
export const optimizerInfo = {
  sgd: { label: "SGD", color: "var(--series-green)", formula: String.raw`\theta_t=\theta_{t-1}-\eta g_t`, memory: "不保存历史梯度", description: "每一步直接沿当前负梯度更新，容易理解。在狭长谷底中，同一个学习率可能让陡峭方向振荡、平缓方向前进缓慢。这里使用全量精确梯度，没有随机抽样噪声。" },
  momentum: { label: "Momentum", color: "var(--series-amber)", formula: String.raw`b_t=0.9b_{t-1}+g_t,\quad\theta_t=\theta_{t-1}-\eta b_t`, memory: "动量缓冲 b（初始为 0）", description: "把之前的更新方向保留下来，像带着惯性移动。方向一致时可能加速，也可能冲过最低点；即使当前梯度为 0，非零动量仍可能推动参数。这里不使用 Nesterov 或 dampening。" },
  adagrad: { label: "AdaGrad", color: "var(--series-blue)", formula: String.raw`s_t=s_{t-1}+g_t^2,\quad\theta_t=\theta_{t-1}-\frac{\eta g_t}{\sqrt{s_t}+\epsilon}`, memory: "累计梯度平方 s（初始为 0）", description: "分别为每个参数累积梯度平方。累计量只增不减，因此对应的有效步长通常越来越小；这有利于区分不同参数的梯度尺度，也可能让后期移动过慢。" },
  rmsprop: { label: "RMSProp", color: "var(--series-violet)", formula: String.raw`v_t=0.9v_{t-1}+0.1g_t^2,\quad\theta_t=\theta_{t-1}-\frac{\eta g_t}{\sqrt{v_t}+\epsilon}`, memory: "梯度平方的滑动平均 v（初始为 0）", description: "用滑动平均代替永久累积，让较早的梯度逐渐淡出。它按坐标调整步长，但固定学习率下也可能在最低点附近持续振荡。这里不使用动量、中心化或偏差校正。" },
  adam: { label: "Adam", color: "var(--series-pink)", formula: String.raw`\begin{aligned}m_t&=0.9m_{t-1}+0.1g_t\\v_t&=0.999v_{t-1}+0.001g_t^2\\\hat m_t&=m_t/(1-0.9^t),\quad\hat v_t=v_t/(1-0.999^t)\\\theta_t&=\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)\end{aligned}`, memory: "一阶矩 m 与二阶矩 v（初始为 0）", description: "结合梯度方向的滑动平均与逐参数缩放，并对零初始化造成的早期偏差进行校正。Adam 仍需调节学习率，也不保证每一步损失下降或在所有问题上最快。这里不使用权重衰减或 AMSGrad。" },
};

export function comparisonRow(run: OptimizerRun, step: number) {
  return run.history[Math.min(step, run.history.length - 1)];
}

export function comparisonCsv(result: ComparisonResult): string {
  const header = "objective,optimizer,learning_rate,requested_steps,start_x,start_y,step,x_after,y_after,loss_after,gradient_x_used,gradient_y_used,update_x,update_y,first_x,first_y,second_x,second_y,run_status";
  return "\uFEFF" + header + "\n" + result.runs.flatMap(run => run.history.map(row => [
    result.config.objective, run.optimizer, run.learning_rate, result.config.steps,
    ...run.history[0].position, row.step, ...row.position, row.loss,
    ...(row.gradient ?? ["", ""]), ...(row.update ?? ["", ""]), ...row.first, ...row.second, run.status,
  ].join(","))).join("\n");
}
