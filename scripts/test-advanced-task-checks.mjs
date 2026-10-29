import assert from "node:assert/strict";
import {
  isXorTaskComplete,
  isMnistDataTaskComplete,
  isMnistTrainingTaskComplete,
  isMnistEvaluationTaskComplete,
  isHandwritingTaskComplete,
  isDeploymentTaskComplete,
} from "../src/lib/advancedTaskChecks.ts";

// Server-shaped fixtures test the predicates only; they never enter the UI or create a run.
const copy = value => structuredClone(value);
const changed = (value, edit) => { const next = copy(value); edit(next); return next; };
const modelId = "a".repeat(32);
const parameters = { w1: [[1, 1, 1, 1], [1, 1, 1, 1]], b1: [0, 0, 0, 0], w2: [1, 1, 1, 1], b2: 0 };
const xor = {
  parameters, gradients: copy(parameters), sample_gradients: copy(parameters),
  iteration: 20, loss: 0.1, accuracy: 1,
  samples: [
    { inputs: [0, 0], target: 0, prediction: 0, probability: 0.1 },
    { inputs: [0, 1], target: 1, prediction: 1, probability: 0.9 },
    { inputs: [1, 0], target: 1, prediction: 1, probability: 0.9 },
    { inputs: [1, 1], target: 0, prediction: 0, probability: 0.1 },
  ],
  selected: { index: 0, inputs: [0, 0], target: 0, z1: [0, 0, 0, 0], a1: [0, 0, 0, 0], z2: -2, output: 0.1, loss: 0.1 },
  grid: { size: 2, x_min: 0, x_max: 1, y_min: 0, y_max: 1, probabilities: [0.1, 0.9, 0.9, 0.1] },
};
assert.equal(isXorTaskComplete(xor), true);
for (const invalid of [null, undefined,
  changed(xor, value => { value.iteration = 0; }),
  changed(xor, value => { value.loss = 0.2; }),
  changed(xor, value => { value.loss = NaN; }),
  changed(xor, value => { value.loss = -1; }),
  changed(xor, value => { value.accuracy = 0.75; }),
  changed(xor, value => { value.samples.pop(); }),
  changed(xor, value => { value.samples[0].prediction = 1; }),
  changed(xor, value => { value.samples[1] = value.samples[0]; }),
  changed(xor, value => { value.samples[1].target = value.samples[1].prediction = 0; }),
]) assert.equal(isXorTaskComplete(invalid), false, "Only a trained, correct four-sample XOR result passes");

const status = {
  runtime: { available: true, torch_version: "test", torchvision_version: "test", device: "cpu", threads: 1 },
  dataset: { status: "ready", message: "Official dataset cached", error: null, official_train: 60000, official_test: 10000 },
  active_run_id: null,
};
const split = {
  seed: 7, validation_ratio: 0.1, train_limit: 6000,
  train_count: 54000, validation_count: 6000, test_count: 10000, effective_train_count: 6000,
  train_class_counts: Array(10).fill(600), validation_class_counts: Array(10).fill(600), test_class_counts: Array(10).fill(1000),
  samples: [],
};
assert.equal(isMnistDataTaskComplete(status, split), true);
assert.equal(isMnistDataTaskComplete(null, split), false);
assert.equal(isMnistDataTaskComplete(status, null), false);
for (const invalidStatus of [
  changed(status, value => { value.dataset.status = "missing"; }),
  changed(status, value => { value.dataset.status = "preparing"; }),
  changed(status, value => { value.dataset.status = "failed"; }),
  changed(status, value => { value.runtime.available = false; }),
  changed(status, value => { value.dataset.official_train = 600; }),
]) assert.equal(isMnistDataTaskComplete(invalidStatus, split), false, "Missing, preparing or non-official data cannot pass");
for (const invalidSplit of [
  changed(split, value => { value.train_count = 60000; }),
  changed(split, value => { value.test_count = 0; }),
  changed(split, value => { value.effective_train_count = 0; }),
  changed(split, value => { value.effective_train_count = 60000; }),
  changed(split, value => { value.train_class_counts.pop(); }),
  changed(split, value => { value.validation_class_counts[0]++; }),
]) assert.equal(isMnistDataTaskComplete(status, invalidSplit), false, "Split totals and label distributions must agree");

const history = [1, 2].map(epoch => ({ epoch, train_loss: 1 / epoch, train_accuracy: 0.5,
  validation_loss: 1 / epoch, validation_accuracy: 0.5, seconds: 1 }));
const run = {
  id: modelId, created_at: "2026-01-01T00:00:00Z", status: "completed",
  config: { model: "mlp", seed: 7, validation_ratio: 0.1, train_limit: 6000, epochs: 2, batch_size: 128, learning_rate: 0.001 },
  epoch: 2, batch: 47, batches_per_epoch: 47, processed_samples: 6000,
  train_samples: 6000, validation_samples: 6000, test_samples: 10000, history,
  best_epoch: 2, best_validation_loss: 0.5, parameter_count: 100, device: "cpu",
  message: "Complete", error: null, evaluation: null, elapsed_seconds: 2, checkpoint_available: true,
};
assert.equal(isMnistTrainingTaskComplete(run), true);
assert.equal(isMnistTrainingTaskComplete(null), false);
for (const state of ["queued", "preparing", "training", "validating", "evaluating", "cancelling", "cancelled", "failed", "interrupted"])
  assert.equal(isMnistTrainingTaskComplete({ ...run, status: state }), false, `${state} must not count as complete`);
for (const invalid of [
  changed(run, value => { value.checkpoint_available = false; }),
  changed(run, value => { value.best_epoch = null; }),
  changed(run, value => { value.best_epoch = 0; }),
  changed(run, value => { value.best_validation_loss = Infinity; }),
  changed(run, value => { value.history = []; }),
  changed(run, value => { value.train_samples = 0; }),
]) assert.equal(isMnistTrainingTaskComplete(invalid), false, "A completed record must retain a real best checkpoint and history");

const matrix = Array.from({ length: 10 }, () => Array(10).fill(0));
matrix[0][0] = 9; matrix[0][1] = 1; matrix[1][0] = 2; matrix[1][1] = 8;
const evaluated = { ...run, evaluation: { loss: 0.5, accuracy: 0.85, total: 20,
  confusion_matrix: matrix, class_metrics: Array.from({ length: 10 }, (_, label) => ({ label, precision: 0, recall: 0, f1: 0, support: label < 2 ? 10 : 0 })),
  error_count: 3, checkpoint_epoch: 2 } };
assert.equal(isMnistEvaluationTaskComplete(evaluated), true);
for (const invalid of [null, run,
  changed(evaluated, value => { value.status = "failed"; }),
  changed(evaluated, value => { value.evaluation.total = 0; }),
  changed(evaluated, value => { value.evaluation.total = 21; }),
  changed(evaluated, value => { value.evaluation.confusion_matrix[0][0]++; }),
  changed(evaluated, value => { value.evaluation.confusion_matrix.pop(); }),
  changed(evaluated, value => { value.evaluation.checkpoint_epoch = 1; }),
  changed(evaluated, value => { value.evaluation.accuracy = 1; }),
  changed(evaluated, value => { value.evaluation.error_count = 0; }),
]) assert.equal(isMnistEvaluationTaskComplete(invalid), false, "Non-empty held-out reports must have consistent counts and checkpoint");

const probabilities = Array(10).fill(0.01); probabilities[7] = 0.91;
const prediction = {
  request_id: "confirmed-request", total_latency_ms: 2,
  preprocessing: { mode: "mnist", original_shape: [28, 28], bounding_box: null, shift: [0, 0],
    processed_pixels: Array.from({ length: 28 }, () => Array(28).fill(0)), normalization: { mean: 0.1307, std: 0.3081 }, ink_pixels: 10 },
  predictions: [{ model_id: modelId, architecture: "mlp", run_id: modelId, checkpoint_epoch: 2,
    predicted_label: 7, probabilities, top_k: [{ label: 7, score: 0.91 }], latency_ms: 1 }],
};
assert.equal(isHandwritingTaskComplete(prediction, [modelId]), true);
assert.equal(isHandwritingTaskComplete(null, [modelId]), false);
assert.equal(isHandwritingTaskComplete(prediction, []), false);
assert.equal(isHandwritingTaskComplete(prediction, ["different-model"]), false);
for (const invalid of [
  changed(prediction, value => { value.predictions = []; }),
  changed(prediction, value => { value.request_id = ""; }),
  changed(prediction, value => { value.preprocessing.ink_pixels = 0; }),
  changed(prediction, value => { value.predictions[0].probabilities.pop(); }),
  changed(prediction, value => { value.predictions[0].probabilities[0] = -1; }),
  changed(prediction, value => { value.predictions[0].probabilities[0] = NaN; }),
  changed(prediction, value => { value.predictions[0].probabilities[0] = 0.2; }),
  changed(prediction, value => { value.predictions[0].predicted_label = 2; }),
  changed(prediction, value => { value.predictions[0].checkpoint_epoch = 0; }),
]) assert.equal(isHandwritingTaskComplete(invalid, [modelId]), false, "Empty or mismatched model outputs cannot prove successful inference");

const catalog = {
  runtime: { available: true, torch_version: "test", device: "cpu" }, defaults: { mlp: modelId, cnn: null },
  models: [{ id: modelId, architecture: "mlp", run_id: modelId, checkpoint_epoch: 2,
    validation_loss: 0.5, test_accuracy: 0.85, source: "run", sha256: "b".repeat(64) }],
};
assert.equal(isDeploymentTaskComplete(catalog), true);
for (const invalid of [null,
  changed(catalog, value => { value.runtime.available = false; }),
  changed(catalog, value => { value.models = []; }),
  changed(catalog, value => { value.models[0].id = ""; }),
  changed(catalog, value => { value.models[0].sha256 = ""; }),
  changed(catalog, value => { value.models[0].checkpoint_epoch = 0; }),
  changed(catalog, value => { value.models[0].validation_loss = NaN; }),
]) assert.equal(isDeploymentTaskComplete(invalid), false, "A reachable catalog needs an available runtime and a checkpoint descriptor");

console.log("Advanced learning evidence: XOR, official MNIST split, completed checkpoint, held-out evaluation, inference and API catalog positive/negative branches passed.");
