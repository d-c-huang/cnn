import assert from "node:assert/strict";
import { phaseCompletion } from "../src/lib/courseProgress.ts";

const ids = ["data", "linear", "neuron", "loss", "gradient", "graph", "backprop", "xor", "mnist-data", "mnist-train", "mnist-evaluation", "handwriting", "deployment"];
for (const phase of [[0,1,2,3,4], [5,6,7], [8,9,10], [11,12]]) {
  const last = phase.at(-1);
  const unrelated = ids.filter((_, index) => !phase.includes(index));
  const skipped = phaseCompletion(last, unrelated, ids, phase);
  assert.equal(skipped.ready, false, "Other phases cannot compensate for skipped lessons");
  assert.equal(skipped.nextUnfinished, phase[0], "Finishing the last lesson first must lead back to the first missing lesson");
  const history = phase.slice(0, -1).map(index => ids[index]);
  const saved = [...history];
  assert.deepEqual(phaseCompletion(last, history, ids, phase), { ready: true, nextUnfinished: undefined });
  assert.deepEqual(history, saved, "Inspecting eligibility must not record completion");
  assert.equal(phaseCompletion(last, [...history, ...history, "unknown"], ids, phase).ready, true, "Existing completion records remain valid");
  history.splice(0, 1);
  assert.equal(phaseCompletion(last, history, ids, phase).ready, false);
}
assert.equal(phaseCompletion(0, ids, ids, []).ready, false);
assert.equal(phaseCompletion(99, ids, ids, [99]).ready, false);
assert.equal(phaseCompletion(0, ids, ids, [0, 99]).ready, false);
assert.equal(phaseCompletion(0, ids, ids, [5, 6, 7]).ready, false);
console.log("Course progress: skipped last lessons, all four phases, existing records, invalid phases and immutable evidence passed.");
