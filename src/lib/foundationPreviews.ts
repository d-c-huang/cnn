import type { GraphRequest, GraphResponse } from "../experiments/ComputationGraph";

export function dataPreview(pixels: number[][], normalize: boolean) {
  const matrix = pixels.map(row => row.map(value => normalize ? value / 255 : value));
  const flat = matrix.flat();
  return {
    matrix, flat, shape: [pixels.length, pixels[0]?.length ?? 0] as [number, number],
    count: flat.length, min: Math.min(...flat), max: Math.max(...flat),
    mean: flat.reduce((sum, value) => sum + value, 0) / flat.length,
  };
}

export function linearPreview(weight: number, bias: number) {
  const points = [-2, -1, 0, 1, 2].map(x => {
    const y = 2 * x + 0.5;
    const prediction = weight * x + bias;
    return { x, y, prediction, error: prediction - y };
  });
  return { points, mse: points.reduce((sum, point) => sum + point.error ** 2, 0) / points.length };
}

export function gradientPreview(weight: number, rate: number) {
  const gradient = 4 * (weight - 2);
  const next = weight - rate * gradient;
  const lossBefore = 2 * (weight - 2) ** 2;
  const lossAfter = 2 * (next - 2) ** 2;
  return {
    weight_before: weight, weight_after: next, gradient,
    loss_before: lossBefore, loss_after: lossAfter,
    converged: Math.abs(next - 2) < 0.001,
    diverged: lossAfter > lossBefore + 1e-12,
  };
}

/** Inspect the same fixed Tanh graph as /api/graph, without committing an update. */
export function graphPreview(request: GraphRequest): GraphResponse {
  const [x1, x2] = request.inputs;
  const [w1, w2] = request.weights;
  const values: Record<string, number> = {
    x1, w1, p1: x1 * w1, x2, w2, p2: x2 * w2,
  };
  values.s = values.p1 + values.p2;
  values.b = request.bias;
  values.z = values.s + request.bias;
  values.a = Math.tanh(values.z);
  values.target = request.target;
  values.e = values.a - request.target;
  values.loss = 0.5 * values.e ** 2;
  const derivatives: [string, string, number][] = [
    ["x1", "p1", w1], ["w1", "p1", x1], ["x2", "p2", w2], ["w2", "p2", x2],
    ["p1", "s", 1], ["p2", "s", 1], ["s", "z", 1], ["b", "z", 1],
    ["z", "a", 1 - values.a ** 2], ["a", "e", 1], ["target", "e", -1], ["e", "loss", values.e],
  ];
  const order = Object.keys(values);
  const gradients: Record<string, number> = Object.fromEntries(order.map(name => [name, 0]));
  gradients.loss = 1;
  for (const destination of [...order].reverse()) {
    for (const [source, target, derivative] of derivatives) {
      if (target === destination) gradients[source] += gradients[target] * derivative;
    }
  }
  const descriptions: Record<string, [string, string, string]> = {
    x1: ["x₁", "input", "输入特征 x₁"], w1: ["w₁", "parameter", "第一路权重 w₁"],
    p1: ["p₁", "multiply", "x₁ × w₁"], x2: ["x₂", "input", "输入特征 x₂"],
    w2: ["w₂", "parameter", "第二路权重 w₂"], p2: ["p₂", "multiply", "x₂ × w₂"],
    s: ["s", "add", "p₁ + p₂"], b: ["b", "parameter", "偏置 b"],
    z: ["z", "add", "s + b"], a: ["a", "tanh", "tanh(z)"],
    target: ["y", "target", "真实标签 y"], e: ["e", "subtract", "a − y"], loss: ["L", "square", "½ × e²"],
  };
  const updates = ["w1", "w2", "b"].map(name => ({
    name, before: values[name], gradient: gradients[name], after: values[name] - request.learning_rate * gradients[name],
  }));
  const next: GraphRequest = {
    ...request, weights: [updates[0].after, updates[1].after], bias: updates[2].after,
  };
  const outputAfter = Math.tanh(x1 * next.weights[0] + x2 * next.weights[1] + next.bias);
  return {
    nodes: order.map(name => ({
      id: name, label: descriptions[name][0], op: descriptions[name][1], value: values[name],
      gradient: gradients[name], expression: descriptions[name][2], trainable: ["w1", "w2", "b"].includes(name),
    })),
    edges: derivatives.map(([source, target, derivative]) => ({
      source, target, local_derivative: derivative, contribution: gradients[target] * derivative,
    })),
    forward_order: order, backward_order: [...order].reverse(),
    output: values.a, loss: values.loss, output_after: outputAfter,
    loss_after: 0.5 * (outputAfter - request.target) ** 2,
    updates, next,
  };
}
