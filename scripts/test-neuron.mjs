import assert from "node:assert/strict";
import { activateNeuron, neuronValue, neuronOutputCurve, neuronActivationCurve } from "../src/lib/neuron.ts";

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
for (const activation of ["relu", "sigmoid", "tanh", "linear"]) {
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

if (process.argv.includes("--api")) {
  for (const activation of ["relu", "sigmoid", "tanh", "linear"]) {
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
      actual.curve.forEach(point => close(activateNeuron(point.x, activation), point.y));
    }
  }
}
console.log("Neuron curves: both input axes, parameter effects, zero weights, ReLU corner, saturation and current marker passed.");
