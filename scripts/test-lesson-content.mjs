import assert from "node:assert/strict";
import { renderToString } from "katex";
import { lessonInsights } from "../src/lib/lessonInsights.ts";
import { lessonDeepDives } from "../src/lib/lessonDeepDives.ts";

assert.deepEqual(Object.keys(lessonDeepDives).sort(), Object.keys(lessonInsights).sort(), "Every existing course needs matching enrichment");
for (const [id, content] of Object.entries(lessonDeepDives)) {
  assert.ok(content.example.steps.length >= 3, `${id}: provide a worked sequence`);
  assert.ok(content.observations.length >= 2 && content.terms.length >= 2, `${id}: provide observation prompts and definitions`);
  assert.ok(content.next === "overview" || lessonDeepDives[content.next], `${id}: next course must exist`);
  renderToString(content.example.tex, { throwOnError: true, strict: "ignore" });
}
const close = (value, expected, tolerance = 0.00005) => assert.ok(Math.abs(value - expected) < tolerance, `${value} does not round to ${expected}`);
close(128 / 255, 0.50196);
close(1 / (1 + Math.exp(-0.32)), 0.5793);
close(1 / (1 + Math.exp(0.64)), 0.3452);
close(2 * (-0.4 - 2) ** 2, 11.52);
close(Math.tanh(0.32), 0.3095);
close(1 - Math.tanh(0.32) ** 2, 0.9042);
close((Math.tanh(0.32) - 0.8) * (1 - Math.tanh(0.32) ** 2), -0.4435);
close(-Math.log(0.9), 0.1054);
close(Math.exp(4) / (Math.exp(4) + 9), 0.8585);
assert.equal(Math.ceil(1000 / 64), 16);
assert.equal(1000 - 15 * 64, 40);
console.log("Lesson content: 13 courses, working navigation, valid TeX and numerical examples passed.");
