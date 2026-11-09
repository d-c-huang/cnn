import assert from "node:assert/strict";
import { activateNeuron, activationDerivative, neuronValue, neuronOutputCurve, neuronActivationCurve, neuronDerivativeCurves } from "../src/lib/neuron.ts";
import { ACTIVATIONS, activationInfo } from "../src/lib/activations.ts";
import { renderToString } from "katex";

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
for (const activation of ACTIVATIONS) {
  const inputs = [0.6, 0.9];
  const weights = [0.8, -0.4];
  const value = neuronValue(inputs, weights, 0.2, activation);
  close(value.z, 0.32);
  for (const axis of [0, 1]) {
    const curve = neuronOutputCurve(inputs, weights, 0.2, activation, axis);
    close(curve.find(point => point.x === inputs[axis]).y, value.output);
    assert.equal(curve[0].x, -3);
    assert.equal(curve.at(-1).x, 3);
    assert.ok(curve.every((point, index) => !index || point.x > curve[index - 1].x));
    const constant = neuronOutputCurve(inputs, [0, 0], -0.7, activation, axis);
    assert.ok(constant.every(point => point.y === activateNeuron(-0.7, activation)), "Zero weights must give a horizontal curve");
  }
  const fixedInputChanged = neuronOutputCurve([0.6, 1.9], weights, 0.2, activation, 0);
  const equivalentBias = neuronOutputCurve(inputs, weights, -0.2, activation, 0);
  fixedInputChanged.forEach((point, i) => close(point.y, equivalentBias[i].y));
  const activationCurve = neuronActivationCurve(value.z, activation);
  close(activationCurve.find(point => point.x === value.z).y, value.output);
}
const corner = neuronOutputCurve([0.6, 0.9], [0.7, 0], -0.3, "relu", 0);
assert.ok(corner.some(point => Math.abs(point.x - 3 / 7) < 1e-12 && Math.abs(point.y) < 1e-12), "Include the exact ReLU corner");
const rising = neuronOutputCurve([0, 0], [2, 0], 0, "sigmoid", 0);
const falling = neuronOutputCurve([0, 0], [-2, 0], 0, "sigmoid", 0);
rising.forEach((point, i) => close(point.y + falling[i].y, 1));
assert.equal(activateNeuron(-1000, "sigmoid"), 0);
assert.equal(activateNeuron(1000, "sigmoid"), 1);
const horizontal = neuronOutputCurve([0, 0], [1, 0], 0.8, "tanh", 1);
assert.ok(horizontal.every(point => point.y === Math.tanh(0.8)), "Sweeping x2 must use w2");

for (const activation of ACTIVATIONS) {
  renderToString(activationInfo[activation].formula, { throwOnError: true });
  renderToString(activationInfo[activation].derivative, { throwOnError: true });
  for (const z of [-1000, -6, -3, -1, -0.0001, 0, 0.0001, 1, 3, 6, 1000]) {
    const slope = activationDerivative(z, activation);
    assert.ok(Number.isFinite(activateNeuron(z, activation)) && Number.isFinite(slope.derivative), `${activation}: finite at ${z}`);
    if (slope.differentiable) {
      const eps = 1e-6;
      const numerical = (activateNeuron(z + eps, activation) - activateNeuron(z - eps, activation)) / (2 * eps);
      assert.ok(Math.abs(numerical - slope.derivative) < 1e-6, `${activation} derivative at ${z}: ${slope.derivative} != ${numerical}`);
    } else {
      assert.equal(z, 0);
      assert.ok(["relu", "leaky_relu"].includes(activation));
    }
  }
}
close(activateNeuron(-3, "leaky_relu"), -0.3);
close(activateNeuron(0, "softplus"), Math.log(2));
close(activationDerivative(0, "softplus").derivative, 0.5);
close(activationDerivative(0, "silu").derivative, 0.5);
close(activationDerivative(0, "elu").derivative, 1);
assert.ok(activationDerivative(-3, "silu").derivative < 0, "SiLU is not monotone");
assert.ok(activationDerivative(3, "silu").derivative > 1);
for (const activation of ["relu", "leaky_relu"]) {
  const branches = neuronDerivativeCurves(0, activation);
  assert.equal(branches.length, 2);
  assert.ok(branches[0].every(point => point.x <= 0));
  assert.ok(branches[1].every(point => point.x >= 0));
  assert.equal(branches[0].at(-1).y, activation === "relu" ? 0 : 0.1);
  assert.equal(branches[1][0].y, 1);
  assert.equal(activationDerivative(0, activation).differentiable, false);
}

if (process.argv.includes("--api")) {
  for (const activation of ACTIVATIONS) {
    for (const [inputs, weights, bias] of [[[0.6, 0.9], [0.8, -0.4], 0.2], [[-3, 3], [-3, 3], 3], [[0, 0], [0, 0], 0]]) {
      const response = await fetch("http://127.0.0.1:8000/api/neuron", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs, weights, bias, activation }),
      });
      assert.equal(response.status, 200);
      const actual = await response.json();
      const preview = neuronValue(inputs, weights, bias, activation);
      close(preview.z, actual.z);
      close(preview.output, actual.output);
      close(preview.derivative, actual.derivative);
      assert.equal(preview.differentiable, actual.differentiable);
      const curve = neuronActivationCurve(preview.z, activation);
      assert.equal(curve.length, actual.curve.length);
      curve.forEach((point, i) => { close(point.x, actual.curve[i].x); close(point.y, actual.curve[i].y); });
      actual.curve.forEach(point => close(activateNeuron(point.x, activation), point.y));
    }
  }
}
console.log("Eight activations: input curves, finite-difference derivatives, corners, saturation, stable extremes, SiLU non-monotonicity and valid formulas passed.");
