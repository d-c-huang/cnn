import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
const walk = directory => readdirSync(directory, { withFileTypes: true }).flatMap(item => item.isDirectory() ? walk(join(directory, item.name)) : [join(directory, item.name)]);
const files = walk("src").filter(file => /\.(css|tsx?)$/.test(file));
const source = files.map(file => readFileSync(file, "utf8")).join("\n");
const design = readFileSync("src/design.css", "utf8");
const colors = new Set(source.match(/#[\da-f]{3,8}\b/gi));
assert.ok(colors.size <= 40, `Keep a bounded palette: ${colors.size} hex colors`);
const breakpoints = new Set();
for (const file of files) {
  let content = readFileSync(file, "utf8");
  if (file.endsWith("design.css")) content = content.replace(/:root\s*\{[^}]*\}/, "");
  assert.ok(!/#[\da-f]{3,8}\b/i.test(content), `${file}: colors belong in root tokens`);
  if (!file.endsWith(".css")) continue;
  for (const [, value] of content.matchAll(/z-index:\s*([^;\n}]+)/g)) assert.ok(value.startsWith("var(--z-"), `${file}: use the stacking tokens`);
  assert.ok(!content.includes("animation-play-state"), `${file}: disabled effects must have a static final state`);
  for (const media of content.matchAll(/@media\s*([^\{]+)/g)) for (const [, width] of media[1].matchAll(/(?:min|max)-width:\s*(\d+)px/g)) breakpoints.add(Number(width));
}
assert.deepEqual([...breakpoints].sort((a, b) => a - b), [430, 640, 900, 1200, 1600]);
const tokens = Object.fromEntries([...design.matchAll(/--([\w-]+):\s*(#[\da-f]{6});/gi)].map(([, name, color]) => [name, color]));
const luminance = hex => [1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16) / 255).map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4).reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index], 0);
for (const level of [1, 2, 3]) for (const surface of [0, 1, 2, 3, 4]) {
  const ratio = (luminance(tokens[`text-${level}`]) + .05) / (luminance(tokens[`surface-${surface}`]) + .05);
  assert.ok(ratio >= 4.5, `text-${level} on surface-${surface}: ${ratio.toFixed(2)}:1`);
}
assert.ok(design.split("\n").length <= 2705 * .85, "Remove the obsolete hero and roadmap styles");
const cssBytes = readdirSync("dist/assets").filter(file => file.endsWith(".css")).reduce((sum, file) => sum + statSync(join("dist/assets", file)).size, 0);
assert.ok(cssBytes <= 270_000, `CSS budget exceeded: ${(cssBytes / 1000).toFixed(1)} KB`);
console.log(`Design system: ${colors.size} token colors, five viewport breakpoints, AA text tokens and ${(cssBytes / 1024).toFixed(1)} KiB CSS passed.`);
