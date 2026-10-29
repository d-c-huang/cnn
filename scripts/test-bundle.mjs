import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const root = fileURLToPath(new URL("../", import.meta.url));
const dist = resolve(root, "dist");
const manifest = JSON.parse(readFileSync(resolve(dist, ".vite/manifest.json"), "utf8"));
const entry = Object.keys(manifest).find(key => manifest[key].isEntry);
assert.ok(entry, "Build first: the Vite manifest must contain an application entry.");
const initial = new Set();
function visit(key) {
  if (initial.has(key)) return;
  assert.ok(manifest[key], `Missing manifest import: ${key}`);
  initial.add(key);
  for (const dependency of manifest[key].imports ?? []) visit(dependency);
}
visit(entry);

const initialFiles = [...initial].map(key => manifest[key].file).filter(file => file.endsWith(".js"));
const initialBytes = initialFiles.reduce((sum, file) => sum + statSync(resolve(dist, file)).size, 0);
const initialGzip = initialFiles.reduce((sum, file) => sum + gzipSync(readFileSync(resolve(dist, file))).length, 0);
assert.ok(initialBytes < 350 * 1024, `Initial JavaScript exceeded 350 KiB: ${initialBytes} bytes`);
assert.ok(!initialFiles.some(file => file.includes("math-typesetting")), "Math typesetting must not load with the overview.");
assert.ok(Object.values(manifest).some(chunk => chunk.file.includes("math-typesetting")), "Keep a reusable, deferred math chunk.");

const lessons = [
  "DataExperiment", "LinearExperiment", "NeuronExperiment", "LossExperiment", "GradientExperiment",
  "GraphExperiment", "BackpropExperiment", "XorExperiment",
  "mnist/MnistDataExperiment", "mnist/MnistTrainingExperiment", "mnist/MnistEvaluationExperiment",
  "inference/HandwritingExperiment", "inference/DeploymentExperiment",
];
for (const lesson of lessons) {
  const key = `src/experiments/${lesson}.tsx`;
  assert.ok(manifest[key]?.isDynamicEntry, `${lesson} must be an on-demand course chunk.`);
  assert.ok(!initial.has(key), `${lesson} must stay out of the initial import graph.`);
}
assert.ok(manifest["src/components/LessonInsights.tsx"]?.isDynamicEntry, "Theory and quiz content should also be deferred.");
console.log(`Bundle checks passed: ${lessons.length} deferred courses; initial JavaScript ${(initialBytes / 1024).toFixed(1)} KiB (${(initialGzip / 1024).toFixed(1)} KiB gzip); math stays deferred.`);
