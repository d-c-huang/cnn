"""Export verified phase-three checkpoints and real MNIST examples for deployment.

The source runs are read-only. Default selection uses the latest completed run
per architecture; test accuracy never participates in model selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def completed_runs(root: Path) -> list[dict]:
    directory = root / "runs" / "mnist"
    records = []
    if not directory.exists():
        return records
    for folder in directory.iterdir():
        if not folder.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", folder.name):
            continue
        try:
            folder.resolve().relative_to(directory.resolve())
            record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            if record["id"] != folder.name or record["status"] != "completed":
                continue
            datetime.fromisoformat(record["created_at"])
            if record["config"]["model"] not in {"mlp", "cnn"} or not (folder / "best.pt").is_file():
                continue
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def select_record(records: list[dict], architecture: str, identifier: str | None) -> dict:
    if identifier and not re.fullmatch(r"[0-9a-f]{32}", identifier):
        raise ValueError(f"{architecture} run ID must be a 32-character hexadecimal UUID")
    candidates = [item for item in records if item["config"]["model"] == architecture and (identifier is None or item["id"] == identifier)]
    if not candidates:
        raise ValueError(f"No completed {architecture.upper()} run with a best checkpoint was found. Finish phase-three training first.")
    return max(candidates, key=lambda item: (datetime.fromisoformat(item["created_at"]), item["id"]))


def export_checkpoint(root: Path, record: dict) -> dict:
    from backend.mnist_engine import build_model, torch_runtime

    torch = torch_runtime()
    architecture = record["config"]["model"]
    source = root / "runs" / "mnist" / record["id"] / "best.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("config", {}).get("model") != architecture:
        raise ValueError(f"{architecture}: checkpoint architecture does not match its run")
    rows = record.get("history", [])
    if not rows:
        raise ValueError(f"{architecture}: no validation history is available")
    if any(not math.isfinite(float(row["validation_loss"])) for row in rows):
        raise ValueError(f"{architecture}: validation history contains non-finite values")
    best = min(rows, key=lambda row: (row["validation_loss"], row["epoch"]))
    epoch = checkpoint.get("epoch")
    if epoch != best["epoch"] or epoch != record.get("best_epoch"):
        raise ValueError(f"{architecture}: checkpoint epoch is not the minimum-validation-loss epoch")
    validation_loss = float(checkpoint["validation_loss"])
    if not math.isclose(validation_loss, float(best["validation_loss"]), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{architecture}: checkpoint validation loss disagrees with history")
    mean = float(checkpoint["normalization"]["mean"])
    std = float(checkpoint["normalization"]["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError(f"{architecture}: invalid normalization metadata")
    model = build_model(architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 1, 28, 28)))
    if tuple(output.shape) != (1, 10) or not torch.isfinite(output).all().item():
        raise ValueError(f"{architecture}: invalid model outputs")
    filename = f"{architecture}-best.pt"
    target = root / "models" / filename
    temporary = target.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    accuracy = (record.get("evaluation") or {}).get("accuracy")
    if accuracy is not None and (not math.isfinite(float(accuracy)) or not 0 <= accuracy <= 1):
        raise ValueError(f"{architecture}: invalid reported test accuracy")
    return {
        "id": record["id"], "architecture": architecture, "run_id": record["id"],
        "checkpoint_epoch": epoch, "validation_loss": validation_loss,
        "test_accuracy": accuracy, "created_at": record["created_at"],
        "filename": filename, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "normalization": {"mean": mean, "std": std},
    }


def real_examples(root: Path) -> list[dict]:
    raw = root / "data" / "MNIST" / "raw"
    images_path = raw / "t10k-images-idx3-ubyte"
    labels_path = raw / "t10k-labels-idx1-ubyte"
    if not images_path.is_file() or not labels_path.is_file():
        print("MNIST test IDX cache is missing; exporting no examples. Handwriting inference still works.", file=sys.stderr)
        return []
    image_bytes, label_bytes = images_path.read_bytes(), labels_path.read_bytes()
    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError("MNIST test IDX headers are incomplete")
    magic, count, height, width = struct.unpack(">4I", image_bytes[:16])
    label_magic, label_count = struct.unpack(">2I", label_bytes[:8])
    if (magic, count, height, width) != (2051, 10000, 28, 28) or (label_magic, label_count) != (2049, 10000):
        raise ValueError("Expected the official 10,000-image MNIST test IDX files")
    if len(image_bytes) != 16 + count * height * width or len(label_bytes) != 8 + count:
        raise ValueError("MNIST test IDX lengths do not match their headers")
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(count, height, width)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    samples = []
    for label in range(10):
        matches = np.flatnonzero(labels == label)
        if not len(matches):
            raise ValueError(f"No real test example exists for label {label}")
        index = int(matches[0])
        samples.append({"index": index, "label": label, "pixels": images[index].tolist(), "source": "mnist-test"})
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY)
    parser.add_argument("--mlp-run")
    parser.add_argument("--cnn-run")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    try:
        records = completed_runs(root)
        selections = [select_record(records, name, getattr(arguments, f"{name}_run")) for name in ("mlp", "cnn")]
        (root / "models").mkdir(parents=True, exist_ok=True)
        models = [export_checkpoint(root, record) for record in selections]
        samples = real_examples(root)
        atomic_json(root / "models" / "examples.json", {"format": "zhiwei-mnist-examples-v1", "samples": samples})
        atomic_json(root / "models" / "manifest.json", {"format": "zhiwei-mnist-models-v1", "models": models})
        print(json.dumps({"exported": [{"model": row["architecture"], "run_id": row["run_id"], "epoch": row["checkpoint_epoch"], "sha256": row["sha256"]} for row in models], "real_examples": len(samples)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"Model export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
