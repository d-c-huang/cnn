import assert from "node:assert/strict";
import { dataPreview, linearPreview, gradientPreview, graphPreview } from "../src/lib/foundationPreviews.ts";

function compare(actual, expected, label = "result") {
  if (typeof expected === "number") {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, `${label} length`);
    expected.forEach((item, index) => compare(actual[index], item, `${label}[${index}]`));
  } else if (expected && typeof expected === "object") {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${label} keys`);
    for (const key of Object.keys(expected)) compare(actual[key], expected[key], `${label}.${key}`);
  } else assert.equal(actual, expected, label);
}

const pixels = Array.from({ length: 8 }, (_, row) => Array.from({ length: 8 }, (_, col) => row * 8 + col === 19 ? 128 : (row + col) % 2 ? 255 : 0));
const normalized = dataPreview(pixels, true);
assert.deepEqual(normalized.shape, [8, 8]);
assert.equal(normalized.flat.length, 64);
assert.equal(normalized.flat[19], 128 / 255);
assert.equal(normalized.matrix[2][3], normalized.flat[19], "Selected matrix and flattened cell must match");
assert.equal(normalized.min, 0);
assert.equal(normalized.max, 1);
assert.deepEqual(dataPreview(pixels, false).matrix, pixels);
assert.notEqual(normalized.matrix, pixels, "Preview must not mutate edited pixels");
assert.equal(linearPreview(2, 0.5).mse, 0);
assert.equal(linearPreview(0.5, 0).points.length, 5);
compare(linearPreview(1, 0.5).mse, linearPreview(3, 0.5).mse, "Symmetric weight errors");
const descending = gradientPreview(-2, 0.1);
assert.ok(descending.loss_after < descending.loss_before);
assert.equal(descending.diverged, false);
compare(gradientPreview(-2, 0.5).loss_after, gradientPreview(-2, 0.5).loss_before, "Oscillating rate");
assert.equal(gradientPreview(-2, 0.55).diverged, true);
assert.equal(gradientPreview(2, 0.1).gradient, 0);
assert.equal(gradientPreview(2, 0.1).converged, true);

const graphCases = [
  { inputs: [0.6, 0.9], weights: [0.8, -0.4], bias: 0.2, target: 0.8, learning_rate: 0.1 },
  { inputs: [-3, 3], weights: [3, -3], bias: -3, target: -1, learning_rate: 1 },
  { inputs: [0, 0], weights: [0, 0], bias: 0, target: 0, learning_rate: 0.01 },
  { inputs: [-0.3, 1.1], weights: [-1.2, 0.7], bias: -0.4, target: -0.6, learning_rate: 0.02 },
];
for (const request of graphCases) {
  const graph = graphPreview(request);
  assert.deepEqual(graph.backward_order, [...graph.forward_order].reverse());
  assert.equal(graph.nodes.find(node => node.id === "loss").gradient, 1);
  for (const name of ["w1", "w2", "b"]) {
    const gradient = graph.nodes.find(node => node.id === name).gradient;
    const epsilon = 1e-6;
    const perturb = delta => {
      const next = { ...request, weights: [...request.weights] };
      if (name === "b") next.bias += delta;
      else next.weights[name === "w1" ? 0 : 1] += delta;
      return graphPreview(next).loss;
    };
    const numeric = (perturb(epsilon) - perturb(-epsilon)) / (2 * epsilon);
    assert.ok(Math.abs(numeric - gradient) < 1e-8, `${name}: chain rule must agree with a finite difference`);
    const row = graph.updates.find(update => update.name === name);
    compare(row.after, row.before - request.learning_rate * gradient, `${name}: SGD proposal`);
  }
}

if (process.argv.includes("--api")) {
  const checks = [
    ["data", { pixels, normalize: true }, normalized],
    ["data", { pixels, normalize: false }, dataPreview(pixels, false)],
    ...[[0.5, 0], [2, 0.5], [-3, -3], [4, 3], [1.7, -0.2]].map(([weight, bias]) => ["linear", { weight, bias }, linearPreview(weight, bias)]),
    ...[[-2, 0.1], [-2, 0.5], [-2, 0.55], [2, 0.1], [5.9, 0.01]].map(([weight, rate]) => ["gradient", { weight, learning_rate: rate }, gradientPreview(weight, rate)]),
    ...graphCases.map(request => ["graph", request, graphPreview(request)]),
  ];
  for (const [endpoint, payload, preview] of checks) {
    const response = await fetch(`http://127.0.0.1:8000/api/${endpoint}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    assert.equal(response.status, 200, endpoint);
    compare(preview, await response.json(), endpoint);
  }
}
console.log("Foundation previews: normalization, exact fitting, descent/oscillation, graph finite differences and API parity passed.");
