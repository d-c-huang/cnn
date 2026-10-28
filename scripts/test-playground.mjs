import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
for (const suite of ["test-convolution.mjs", "test-network-builder.mjs", "test-challenges.mjs"]) {
  const args = ["--experimental-strip-types", `scripts/${suite}`];
  if (suite === "test-challenges.mjs" && process.argv.includes("--api")) args.push("--api");
  const result = spawnSync(process.execPath, args, { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
