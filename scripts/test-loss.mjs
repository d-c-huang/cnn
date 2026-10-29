import assert from "node:assert/strict";
import { squaredErrorPreview } from "../src/lib/loss.ts";

const close = (actual, expected, label) => assert.ok(
  Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-12,
  `${label}: ${actual} != ${expected}`,
);

const cases = [
  [-1, 2], [0, 0], [1.3, 1.3], [-3, -3], [3, 3],
  [5, -3], [-5, 3], [-5, -3], [5, 3],
  [0.1, -0.3], [-0.1, 0.3], [5, 1.3], [-5, -1.3],
  [4.9, -2.9], [-4.9, 2.9],
];

for (const [prediction, target] of cases) {
  const value = squaredErrorPreview(prediction, target);
  const error = prediction - target;
  close(value.loss, error ** 2, "Squared error");
  close(value.gradient, 2 * error, "Derivative with respect to prediction");
  assert.ok(value.curve.length >= 121 && value.curve.length <= 123, "Use 121 samples plus the current prediction and exact minimum");
  assert.equal(value.curve[0].x, Math.min(target - 5, prediction));
  assert.equal(value.curve.at(-1).x, Math.max(target + 5, prediction));
  assert.ok(value.curve.every((point, index) => !index || point.x > value.curve[index - 1].x), "Curve coordinates must be strictly sorted and unique");
  for (const point of value.curve) {
    assert.ok(point.y >= 0, "Squared errors cannot be negative");
    close(point.y, (point.x - target) ** 2, "Curve point");
  }
  const minimum = value.curve.find(point => point.x === target);
  assert.ok(minimum, "The exact target must be present, including fractional targets");
  assert.equal(minimum.y, 0);
  const current = value.curve.find(point => point.x === prediction);
  assert.ok(current, "The current prediction must be present, even at an asymmetric domain endpoint");
  close(current.y, value.loss, "Current prediction marker");
  if (prediction === target) {
    assert.equal(value.loss, 0);
    assert.equal(value.gradient, 0);
  }
}

for (const target of [-3, -0.3, 0, 1.3, 3]) {
  for (const distance of [0.1, 0.7, 1.5, 2]) {
    const before = squaredErrorPreview(target - distance, target);
    const after = squaredErrorPreview(target + distance, target);
    close(before.loss, after.loss, "Equal distances on opposite sides give equal loss");
    close(before.gradient, -after.gradient, "Opposite errors give opposite gradients");
    close(after.loss, distance ** 2, "Loss must not include a one-half factor");
  }
}

assert.equal(squaredErrorPreview(5, -3).loss, 64);
assert.equal(squaredErrorPreview(5, -3).gradient, 16);
assert.equal(squaredErrorPreview(-5, 3).loss, 64);
assert.equal(squaredErrorPreview(-5, 3).gradient, -16);
const asymmetric = squaredErrorPreview(-5, 3).curve;
assert.equal(asymmetric[0].x, -5);
assert.equal(asymmetric.at(-1).x, 8);

if (process.argv.includes("--api")) {
  for (const [prediction, target] of cases) {
    const response = await fetch("http://127.0.0.1:8000/api/loss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prediction, target }),
      signal: AbortSignal.timeout(8000),
    });
    assert.equal(response.status, 200, `Loss API for prediction=${prediction}, target=${target}`);
    const actual = await response.json();
    const preview = squaredErrorPreview(prediction, target);
    close(preview.loss, actual.loss, "API loss");
    close(preview.gradient, actual.gradient, "API gradient");
    assert.equal(preview.curve.length, actual.curve.length, "Preview and API must have identical sample counts to avoid a delayed curve jump");
    preview.curve.forEach((point, index) => {
      close(point.x, actual.curve[index].x, `API curve x at ${index}`);
      close(point.y, actual.curve[index].y, `API curve y at ${index}`);
    });
  }
}

console.log("Loss preview: zero, symmetry, gradients, extrema, exact markers and sorted asymmetric curves passed.");
