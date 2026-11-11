import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { readMotionLevel, setMotionLevel } from "../src/lib/motion.ts";
import { overlayLabelColor } from "../src/lib/colorContrast.ts";
const saved = new Map();
let reduced = false;
let changes = 0;
globalThis.localStorage = { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value) };
globalThis.window = { matchMedia: () => ({ matches: reduced }), dispatchEvent: () => { changes++; } };
globalThis.document = { documentElement: { dataset: {} } };
assert.equal(readMotionLevel(), "full");
reduced = true;
assert.equal(readMotionLevel(), "reduced", "Follow the system until the user saves a preference");
for (const level of ["off", "full", "reduced"]) {
  setMotionLevel(level);
  assert.equal(readMotionLevel(), level);
  assert.equal(document.documentElement.dataset.motion, level);
}
assert.equal(changes, 3);
localStorage.getItem = () => { throw new Error("Storage unavailable"); };
assert.equal(readMotionLevel(), "reduced");
localStorage.setItem = () => { throw new Error("Storage unavailable"); };
assert.doesNotThrow(() => setMotionLevel("off"));
assert.equal(document.documentElement.dataset.motion, "off");
for (const rgb of [[84, 214, 183], [225, 158, 100], [117, 153, 77], [200, 149, 108]]) for (let i = 0; i <= 100; i++) {
  const alpha = i / 100;
  const linear = rgb.map((value, index) => (value * alpha + [11, 16, 23][index] * (1 - alpha)) / 255).map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4);
  const luminance = linear.reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index], 0);
  const foreground = overlayLabelColor(rgb, alpha);
  const contrast = foreground.includes("shadow-ink") ? (luminance + .05) / .05 : 1.05 / (luminance + .05);
  assert.ok(contrast >= 4.5, `Heatmap label must meet AA at alpha ${alpha}`);
}
console.log("Display preferences: system defaults, persisted overrides, unavailable storage and 404 heatmap-label contrast cases passed.");

const canvas = readFileSync("src/experiments/inference/DigitCanvas.tsx", "utf8");
assert.ok(!/ctx\.(?:fillStyle|strokeStyle)\s*=\s*["']var\(/.test(canvas), "Canvas pixel paints cannot resolve CSS custom properties");
assert.equal((canvas.match(/ctx\.fillStyle = "rgb\(0, 0, 0\)"/g) ?? []).length, 2, "Initial and cleared canvas must stay black");
assert.ok(canvas.includes('ctx.strokeStyle = "rgb(255, 255, 255)"'), "Handwriting must remain white pixel data");
