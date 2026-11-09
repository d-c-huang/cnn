import assert from "node:assert/strict";
import { lossPreview, lossValue, squaredErrorPreview } from "../src/lib/loss.ts";
import { spawnSync } from "node:child_process";

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

const extendedCases = [];
for (const kind of ["mse", "mae", "huber", "bce"]) {
  for (const delta of kind === "huber" ? [0.1, 1, 3] : [1]) {
    for (const target of kind === "bce" ? [0, 1] : [-0.3, 2]) {
      for (const prediction of [-8, -1, 0, 0.1, target, target - delta, target + delta, 8]) {
        const preview = lossPreview(prediction, target, kind, delta);
        extendedCases.push({ prediction, target, kind, delta });
        assert.ok(preview.loss >= 0 && Number.isFinite(preview.loss));
        assert.ok(preview.curve.some(point => point.x === prediction && point.y === preview.loss));
        if (preview.differentiable) {
          const eps = 1e-5;
          const numerical = (lossValue(prediction + eps, target, kind, delta).loss - lossValue(prediction - eps, target, kind, delta).loss) / (2 * eps);
          assert.ok(Math.abs(numerical - preview.gradient) < 4e-6, `${kind}: finite-difference derivative`);
        } else {
          assert.equal(kind, "mae");
          assert.equal(preview.gradient, 0);
        }
      }
    }
  }
}
close(lossValue(3, 0, "mae").loss, 3, "MAE known value");
close(lossValue(3, 0, "huber", 1).loss, 2.5, "Huber linear branch");
close(lossValue(0.5, 0, "huber", 1).loss, 0.125, "Huber quadratic branch");
close(lossValue(0, 1, "bce").loss, Math.log(2), "Uncertain BCE");
for (const prediction of [-1000, 1000]) for (const target of [0, 1]) {
  const value = lossValue(prediction, target, "bce");
  assert.ok(Number.isFinite(value.loss) && Number.isFinite(value.gradient));
  close(value.loss, (prediction > 0 ? 1 - target : target) * 1000, "Extreme logits");
  extendedCases.push({ prediction, target, kind: "bce", delta: 1 });
}
// Compare every point with the Python implementation without a running server.
const python = spawnSync(process.env.PYTHON || "python", ["-c", "import json,sys; from backend.app import LossRequest,loss_experiment; print(json.dumps([loss_experiment(LossRequest(**x)) for x in json.load(sys.stdin)]))"], {
  input: JSON.stringify(extendedCases), encoding: "utf8", maxBuffer: 8 * 1024 * 1024,
});
assert.equal(python.status, 0, python.stderr);
JSON.parse(python.stdout).forEach((actual, index) => {
  const { prediction, target, kind, delta } = extendedCases[index];
  const preview = lossPreview(prediction, target, kind, delta);
  close(actual.loss, preview.loss, `${kind}: backend loss`);
  close(actual.gradient, preview.gradient, `${kind}: backend gradient`);
  assert.equal(actual.differentiable, preview.differentiable);
  assert.equal(actual.curve.length, preview.curve.length);
  actual.curve.forEach((point, i) => {
    close(point.x, preview.curve[i].x, `${kind}: backend curve x`);
    close(point.y, preview.curve[i].y, `${kind}: backend curve y`);
  });
});
console.log(`Loss preview: existing MSE checks plus ${extendedCases.length} multi-loss cases, finite differences, extreme logits and Python parity passed.`);
