import assert from "node:assert/strict";
import {
  architectureChallenge,
  edgeChallenge,
  linearChallenge,
  parseChallengeProgress,
  xorChallenge,
} from "../src/lib/challenges.ts";
import { analyzeNetwork, createExample } from "../src/lib/networkBuilder.ts";

assert.equal(linearChallenge(2, 0.5).passed, true);
assert.equal(linearChallenge(1.9, 0.5).passed, false);
assert.equal(linearChallenge(Number.NaN, 0).passed, false);
assert.equal(edgeChallenge([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).passed, false);
assert.equal(edgeChallenge(Array.from({ length: 3 }, () => [-1, 0, 1])).passed, true);
assert.equal(edgeChallenge(Array.from({ length: 3 }, () => [1, 0, -1])).passed, true);
assert.equal(edgeChallenge(Array.from({ length: 3 }, () => [0, 0, 0])).passed, false);
assert.throws(() => edgeChallenge([[Number.NaN, 0, 0], [0, 0, 0], [0, 0, 0]]));
const cnn = createExample("cnn");
const mlp = createExample("mlp");
assert.equal(architectureChallenge(cnn, analyzeNetwork(cnn)).passed, true);
assert.equal(architectureChallenge(mlp, analyzeNetwork(mlp)).passed, false);
assert.equal(architectureChallenge(null, null).passed, false);
const invalid = [cnn[4], ...cnn.slice(0, 4)];
assert.equal(architectureChallenge(invalid, analyzeNetwork(invalid)).passed, false);
const snapshot = {
  iteration: 20,
  loss: 0.15,
  accuracy: 1,
  samples: [0, 1, 1, 0].map(target => ({ target, prediction: target })),
};
assert.equal(xorChallenge(snapshot), true);
assert.equal(xorChallenge({ ...snapshot, loss: 0.16 }), false);
assert.equal(xorChallenge({ ...snapshot, iteration: 0 }), false);
assert.equal(xorChallenge({ ...snapshot, samples: snapshot.samples.slice(1) }), false);
assert.deepEqual(parseChallengeProgress({ version: 1, completed: ["linear-fit", "linear-fit", "bad", "xor-master"] }), ["linear-fit", "xor-master"]);
assert.deepEqual(parseChallengeProgress({ version: 2, completed: ["xor-master"] }), []);
assert.deepEqual(parseChallengeProgress(null), []);
console.log("Challenge numerical and persistence checks passed (19 assertions).");

if (process.argv.includes("--api")) {
  const base = process.env.CHALLENGE_API_URL || "http://127.0.0.1:8000";
  const call = async (path, body) => {
    const response = await fetch(`${base}/api${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(8000),
    });
    assert.equal(response.ok, true, `${path}: HTTP ${response.status}`);
    return response.json();
  };
  let current = await call("/xor/init", { seed: 7 });
  assert.equal(xorChallenge(current), false);
  for (let batch = 0; batch < 50 && !xorChallenge(current); batch++) {
    const result = await call("/xor/step", {
      parameters: current.parameters,
      iteration: current.iteration,
      learning_rate: 1,
      steps: 20,
      selected_sample: 0,
    });
    current = result.snapshot;
  }
  assert.equal(xorChallenge(current), true, "Default XOR challenge should reach its explicit goal within 1000 steps.");
  console.log(`Real XOR API challenge passed at ${current.iteration} steps, BCE ${current.loss.toFixed(6)}, predictions ${current.samples.map(sample => sample.prediction).join(",")}.`);
}
