export type Activation = "relu" | "sigmoid" | "tanh" | "linear" | "leaky_relu" | "elu" | "softplus" | "silu";
export type Pair = [number, number];

export function activateNeuron(z: number, activation: Activation): number {
  if (activation === "relu") return Math.max(0, z);
  if (activation === "leaky_relu") return z >= 0 ? z : 0.1 * z;
  if (activation === "elu") return z >= 0 ? z : Math.expm1(z);
  if (activation === "softplus") return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
  if (activation === "silu") return z * activateNeuron(z, "sigmoid");
  if (activation === "tanh") return Math.tanh(z);
  if (activation === "sigmoid") {
    const exp = Math.exp(-Math.abs(z));
    return z >= 0 ? 1 / (1 + exp) : exp / (1 + exp);
  }
  return z;
}

/** At piecewise corners, report the explicit backward convention separately. */
export function activationDerivative(z: number, activation: Activation) {
  const differentiable = !((activation === "relu" || activation === "leaky_relu") && z === 0);
  let derivative = 1;
  if (activation === "relu") derivative = z > 0 ? 1 : 0;
  if (activation === "leaky_relu") derivative = z > 0 ? 1 : 0.1;
  if (activation === "elu") derivative = z >= 0 ? 1 : Math.exp(z);
  if (activation === "sigmoid" || activation === "softplus" || activation === "silu") {
    const p = activateNeuron(z, "sigmoid");
    derivative = activation === "sigmoid" ? p * (1 - p) : activation === "softplus" ? p : p + z * p * (1 - p);
  }
  if (activation === "tanh") derivative = 1 - Math.tanh(z) ** 2;
  return { derivative, differentiable };
}

export function neuronValue(inputs: Pair, weights: Pair, bias: number, activation: Activation) {
  const products: Pair = [inputs[0] * weights[0], inputs[1] * weights[1]];
  const z = products[0] + products[1] + bias;
  return { products, z, output: activateNeuron(z, activation), ...activationDerivative(z, activation) };
}

/** Sweep one input, keeping the other input and all parameters fixed. */
export function neuronOutputCurve(inputs: Pair, weights: Pair, bias: number, activation: Activation, axis: 0 | 1) {
  const fixed = axis === 0 ? 1 : 0;
  const offset = weights[fixed] * inputs[fixed] + bias;
  const xs = Array.from({ length: 241 }, (_, i) => -3 + i / 40);
  xs.push(inputs[axis]);
  // Include the exact ReLU corner so a sampled segment cannot round it off.
  if ((activation === "relu" || activation === "leaky_relu") && weights[axis] !== 0) {
    const corner = -offset / weights[axis];
    if (corner >= -3 && corner <= 3) xs.push(corner);
  }
  return [...new Set(xs)].sort((a, b) => a - b).map(x => ({
    x, y: activateNeuron(weights[axis] * x + offset, activation),
  }));
}

export function neuronActivationCurve(z: number, activation: Activation) {
  const left = Math.min(-6, z) - 0.5;
  const right = Math.max(6, z) + 0.5;
  const xs = Array.from({ length: 241 }, (_, i) => left + (right - left) * i / 240);
  xs.push(z, 0);
  return [...new Set(xs)].sort((a, b) => a - b).map(x => ({ x, y: activateNeuron(x, activation) }));
}

export function neuronDerivativeCurves(z: number, activation: Activation) {
  const points = neuronActivationCurve(z, activation).map(point => ({ x: point.x, y: activationDerivative(point.x, activation).derivative }));
  if (activation === "relu" || activation === "leaky_relu") {
    // Separate one-sided branches so the chart never draws a false sloped transition.
    return [
      [...points.filter(point => point.x < 0), { x: 0, y: activation === "relu" ? 0 : 0.1 }],
      [{ x: 0, y: 1 }, ...points.filter(point => point.x > 0)],
    ];
  }
  return [points];
}
