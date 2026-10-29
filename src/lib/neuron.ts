export type Activation = "relu" | "sigmoid" | "tanh" | "linear";
export type Pair = [number, number];

export function activateNeuron(z: number, activation: Activation): number {
  if (activation === "relu") return Math.max(0, z);
  if (activation === "tanh") return Math.tanh(z);
  if (activation === "sigmoid") {
    const exp = Math.exp(-Math.abs(z));
    return z >= 0 ? 1 / (1 + exp) : exp / (1 + exp);
  }
  return z;
}

export function neuronValue(inputs: Pair, weights: Pair, bias: number, activation: Activation) {
  const products: Pair = [inputs[0] * weights[0], inputs[1] * weights[1]];
  const z = products[0] + products[1] + bias;
  return { products, z, output: activateNeuron(z, activation) };
}

/** Sweep one input, keeping the other input and all parameters fixed. */
export function neuronOutputCurve(inputs: Pair, weights: Pair, bias: number, activation: Activation, axis: 0 | 1) {
  const fixed = axis === 0 ? 1 : 0;
  const offset = weights[fixed] * inputs[fixed] + bias;
  const xs = Array.from({ length: 241 }, (_, i) => -3 + i / 40);
  xs.push(inputs[axis]);
  // Include the exact ReLU corner so a sampled segment cannot round it off.
  if (activation === "relu" && weights[axis] !== 0) {
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
  const xs = Array.from({ length: 121 }, (_, i) => left + (right - left) * i / 120);
  xs.push(z, 0);
  return [...new Set(xs)].sort((a, b) => a - b).map(x => ({ x, y: activateNeuron(x, activation) }));
}
