import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const venvPython = fileURLToPath(
  new URL(
    process.platform === "win32"
      ? "../.venv/Scripts/python.exe"
      : "../.venv/bin/python",
    import.meta.url,
  ),
);
const candidates = [
  ...new Set(
    [
      existsSync(venvPython) ? venvPython : null,
      process.env.PYTHON,
      "python",
    ].filter(Boolean),
  ),
];
const python = candidates.find((candidate) => {
  const probe = spawnSync(
    candidate,
    [
      "-c",
      "import sys; assert sys.version_info >= (3, 11); import fastapi, uvicorn, numpy",
    ],
    { cwd: root, encoding: "utf8" },
  );
  return !probe.error && probe.status === 0;
});
if (!python) {
  console.error(
    "请先安装计算服务依赖：python -m pip install -r requirements.txt",
  );
  process.exit(1);
}

const children = [
  spawn(
    python,
    [
      "-m",
      "uvicorn",
      "backend.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      "8000",
    ],
    { cwd: root, stdio: "inherit" },
  ),
  spawn(
    process.execPath,
    ["node_modules/vite/bin/vite.js", "--host", "127.0.0.1"],
    { cwd: root, stdio: "inherit" },
  ),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.exitCode === null) child.kill("SIGTERM");
  }
  process.exitCode = code;
}
for (const child of children) {
  child.on("error", (error) => {
    console.error(error.message);
    stop(1);
  });
  child.on("exit", (code, signal) => stop(code ?? (signal ? 1 : 0)));
}
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
