import assert from "node:assert/strict";
import { analyzeNetwork, createExample, createLayer, MAX_NETWORK_LAYERS, NETWORK_STORAGE_KEY, parseNetwork, readSavedNetwork, toPytorchCode } from "../src/lib/networkBuilder.ts";

const cnn = createExample("cnn");
const cnnAnalysis = analyzeNetwork(cnn);
assert.equal(cnnAnalysis.valid, true);
assert.equal(cnnAnalysis.classifierReady, true);
assert.deepEqual(cnnAnalysis.rows.map(row => row.output), [[8, 28, 28], [8, 28, 28], [8, 14, 14], [1568], [10]]);
assert.deepEqual(cnnAnalysis.rows.map(row => row.parameters), [80, 0, 0, 0, 15690]);
assert.equal(cnnAnalysis.totalParameters, 15770);
assert.deepEqual(cnnAnalysis.output, [10]);

const mlp = createExample("mlp");
const mlpAnalysis = analyzeNetwork(mlp);
assert.equal(mlpAnalysis.valid, true);
assert.deepEqual(mlpAnalysis.rows.map(row => row.output), [[784], [128], [128], [10]]);
assert.equal(mlpAnalysis.totalParameters, (784 + 1) * 128 + (128 + 1) * 10);
assert.equal(mlpAnalysis.totalParameters, 101770);

const strided = [{ id: "conv-strided", type: "conv", outChannels: 8, kernel: 5, stride: 2, padding: 0 }];
const stridedAnalysis = analyzeNetwork(strided);
assert.deepEqual(stridedAnalysis.rows[0].output, [8, 12, 12]);
assert.equal(stridedAnalysis.totalParameters, 208);
assert.equal(stridedAnalysis.shapeValid, true);
assert.equal(stridedAnalysis.valid, false);
assert.match(stridedAnalysis.errors[0], /\[10\]/);

const invalidLinear = analyzeNetwork([createLayer("linear"), createLayer("relu")]);
assert.equal(invalidLinear.shapeValid, false);
assert.match(invalidLinear.rows[0].error, /Flatten/);
assert.equal(invalidLinear.rows[1].input, null);
assert.equal(invalidLinear.rows[1].output, null);
assert.match(invalidLinear.rows[1].error, /前序层/);
assert.equal(invalidLinear.totalParameters, 0);

for (const spatial of ["conv", "pool"]) {
  const invalidSpatial = analyzeNetwork([createLayer("flatten"), createLayer(spatial)]);
  assert.equal(invalidSpatial.shapeValid, false);
  assert.match(invalidSpatial.rows[1].error, /向量/);
}
const shrinking = analyzeNetwork(Array.from({ length: 5 }, () => createLayer("pool")));
assert.deepEqual(shrinking.rows.slice(0, 4).map(row => row.output), [[1, 14, 14], [1, 7, 7], [1, 3, 3], [1, 1, 1]]);
assert.match(shrinking.rows[4].error, /必须大于 0/);
const empty = analyzeNetwork([]);
assert.equal(empty.valid, false);
assert.match(empty.errors[0], /网络为空/);

assert.deepEqual(parseNetwork(cnn), cnn);
assert.equal(parseNetwork(null), null);
assert.equal(parseNetwork([{ ...cnn[0], outChannels: "8" }]), null);
assert.equal(parseNetwork([{ ...cnn[0], kernel: 4 }]), null);
assert.equal(parseNetwork([{ ...cnn[0], extra: true }]), null);
assert.equal(parseNetwork([{ ...cnn[0], id: "bad');print(1)" }]), null);
assert.equal(parseNetwork([cnn[0], cnn[0]]), null);
assert.equal(parseNetwork(Array.from({ length: MAX_NETWORK_LAYERS + 1 }, () => createLayer("relu"))), null);
assert.equal(analyzeNetwork([{ ...cnn[0], stride: 0 }]).valid, false);

const memory = new Map();
Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: key => memory.get(key) ?? null } });
assert.equal(readSavedNetwork(), null);
memory.set(NETWORK_STORAGE_KEY, "broken{");
assert.equal(readSavedNetwork(), null);
memory.set(NETWORK_STORAGE_KEY, JSON.stringify({ version: 2, layers: cnn }));
assert.equal(readSavedNetwork(), null);
memory.set(NETWORK_STORAGE_KEY, JSON.stringify({ version: 1, layers: cnn }));
assert.deepEqual(readSavedNetwork(), cnn);
memory.set(NETWORK_STORAGE_KEY, JSON.stringify({ version: 1, layers: [] }));
assert.deepEqual(readSavedNetwork(), []);
memory.set(NETWORK_STORAGE_KEY, JSON.stringify({ version: 1, layers: [cnn[0], cnn[0]] }));
assert.equal(readSavedNetwork(), null);

const code = toPytorchCode(cnn);
assert.match(code, /nn\.Sequential\(/);
assert.match(code, /nn\.Conv2d\(1, 8, kernel_size=3, stride=1, padding=1\)/);
assert.match(code, /nn\.MaxPool2d\(kernel_size=2, stride=2\)/);
assert.match(code, /nn\.Flatten\(start_dim=1\)/);
assert.match(code, /nn\.Linear\(1568, 10\)/);
assert.match(code, /torch\.zeros\(1, 1, 28, 28\)/);
assert.throws(() => toPytorchCode([createLayer("linear")]), /修复网络结构/);
assert.throws(() => toPytorchCode([]), /修复网络结构/);

console.log("Network builder: shapes, parameter counts, invalid pipelines, persistence and PyTorch export passed.");
