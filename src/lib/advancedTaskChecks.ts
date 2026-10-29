import type { DataSplit, MnistStatus, Run } from "./mnist";
import type { ModelCatalog, PredictionResponse } from "./inference";
import type { XorSnapshot } from "./xor";

const nonnegative = (value: number) => Number.isFinite(value) && value >= 0;
const positiveInteger = (value: number) => Number.isInteger(value) && value > 0;
const countsMatch = (counts: number[], total: number) => counts.length === 10
  && counts.every(count => Number.isInteger(count) && count >= 0)
  && counts.reduce((sum, count) => sum + count, 0) === total;

/** These predicates consume confirmed server payloads, never optimistic previews. */
export function isXorTaskComplete(snapshot: XorSnapshot | null | undefined): boolean {
  return !!snapshot && positiveInteger(snapshot.iteration) && nonnegative(snapshot.loss) && snapshot.loss < 0.2
    && snapshot.accuracy === 1 && snapshot.samples.length === 4
    && new Set(snapshot.samples.map(sample => sample.inputs.join(","))).size === 4
    && snapshot.samples.every(sample => sample.inputs.every(input => input === 0 || input === 1)
      && sample.target === (sample.inputs[0] === sample.inputs[1] ? 0 : 1)
      && sample.prediction === sample.target && Number.isFinite(sample.probability)
      && sample.probability >= 0 && sample.probability <= 1
      && sample.prediction === (sample.probability >= 0.5 ? 1 : 0));
}

export function isMnistDataTaskComplete(status: MnistStatus | null | undefined, split: DataSplit | null | undefined): boolean {
  return !!status && !!split && status.runtime.available && status.dataset.status === "ready"
    && status.dataset.official_train === 60000 && status.dataset.official_test === 10000
    && positiveInteger(split.train_count) && positiveInteger(split.validation_count)
    && split.train_count + split.validation_count === status.dataset.official_train
    && split.test_count === status.dataset.official_test
    && positiveInteger(split.effective_train_count) && split.effective_train_count <= split.train_count
    && countsMatch(split.train_class_counts, split.effective_train_count)
    && countsMatch(split.validation_class_counts, split.validation_count)
    && countsMatch(split.test_class_counts, split.test_count);
}

export function isMnistTrainingTaskComplete(run: Run | null | undefined): boolean {
  return !!run && run.status === "completed" && run.checkpoint_available
    && run.best_epoch !== null && positiveInteger(run.best_epoch)
    && run.best_validation_loss !== null && nonnegative(run.best_validation_loss)
    && positiveInteger(run.train_samples) && positiveInteger(run.validation_samples)
    && run.history.some(point => point.epoch === run.best_epoch && nonnegative(point.validation_loss));
}

export function isMnistEvaluationTaskComplete(run: Run | null | undefined): boolean {
  const evaluation = run?.evaluation;
  if (!run || run.status !== "completed" || !evaluation || !positiveInteger(evaluation.total)
    || !nonnegative(evaluation.loss) || !Number.isFinite(evaluation.accuracy)
    || evaluation.accuracy < 0 || evaluation.accuracy > 1
    || !positiveInteger(evaluation.checkpoint_epoch) || evaluation.checkpoint_epoch !== run.best_epoch
    || evaluation.confusion_matrix.length !== 10
    || !evaluation.confusion_matrix.every(row => row.length === 10 && row.every(count => Number.isInteger(count) && count >= 0))
    || evaluation.confusion_matrix.flat().reduce((sum, count) => sum + count, 0) !== evaluation.total) return false;
  const correct = evaluation.confusion_matrix.reduce((sum, row, label) => sum + row[label], 0);
  return Math.abs(correct / evaluation.total - evaluation.accuracy) < 1e-8
    && evaluation.error_count === evaluation.total - correct;
}

export function isHandwritingTaskComplete(prediction: PredictionResponse | null | undefined, modelIds: readonly string[]): boolean {
  return !!prediction && prediction.request_id.length > 0 && positiveInteger(prediction.preprocessing.ink_pixels)
    && modelIds.length > 0 && new Set(modelIds).size === modelIds.length
    && prediction.predictions.length === modelIds.length
    && new Set(prediction.predictions.map(item => item.model_id)).size === modelIds.length
    && prediction.predictions.every(item => modelIds.includes(item.model_id) && positiveInteger(item.checkpoint_epoch)
      && Number.isInteger(item.predicted_label) && item.predicted_label >= 0 && item.predicted_label <= 9
      && item.probabilities.length === 10 && item.probabilities.every(value => Number.isFinite(value) && value >= 0 && value <= 1)
      && Math.abs(item.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 0.0001
      && item.predicted_label === item.probabilities.indexOf(Math.max(...item.probabilities)));
}

export function isDeploymentTaskComplete(catalog: ModelCatalog | null | undefined): boolean {
  return !!catalog && catalog.runtime.available && catalog.models.some(model => model.id.length > 0
    && positiveInteger(model.checkpoint_epoch) && nonnegative(model.validation_loss)
    && /^[0-9a-f]{64}$/.test(model.sha256));
}
