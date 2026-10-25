"""Inference from completed MNIST checkpoints, with one shared input pipeline.

No model is initialized as a substitute for a missing checkpoint. Catalog and
example reads are lightweight; PyTorch is imported only for actual inference.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading
import time
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.mnist_engine import ROOT, build_model, runtime_status, torch_runtime


Pixel = Annotated[float, Field(strict=True, ge=0, le=255, allow_inf_nan=False)]
ModelId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$", strict=True)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ids: list[ModelId] = Field(min_length=1, max_length=2)
    pixels: list[list[Pixel]] = Field(min_length=8, max_length=320)
    mode: Literal["canvas", "mnist"] = "canvas"

    @model_validator(mode="after")
    def image_shape(self):
        width = len(self.pixels[0])
        if not 8 <= width <= 320 or any(len(row) != width for row in self.pixels):
            raise ValueError("pixels must be a rectangular image, each side 8–320")
        if self.mode == "mnist" and (len(self.pixels), width) != (28, 28):
            raise ValueError("mnist mode requires an unchanged 28 × 28 image")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be unique")
        return self


def translate_without_wrap(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate with a black exterior, never cyclically wrapping edge pixels."""
    height, width = image.shape
    result = np.zeros_like(image)
    x0, x1 = max(0, -dx), min(width, width - dx)
    y0, y1 = max(0, -dy), min(height, height - dy)
    if x1 > x0 and y1 > y0:
        result[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = image[y0:y1, x0:x1]
    return result


def preprocess_pixels(pixels, mode: str) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(pixels, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or matrix.min() < 0 or matrix.max() > 255:
        raise HTTPException(422, "图像必须是 0–255 的有限灰度数值。")
    height, width = matrix.shape
    if not 8 <= height <= 320 or not 8 <= width <= 320:
        raise HTTPException(422, "图像的长宽必须在 8–320 像素之间。")
    # Official samples pass through unchanged, including arbitrarily faint ink.
    # The threshold only belongs to the handwriting crop/centering pipeline.
    ink = matrix > (0 if mode == "mnist" else 16)
    if not ink.any():
        raise HTTPException(422, "画布为空，请先写一个数字。")
    metadata = {"mode": mode, "original_shape": [height, width], "bounding_box": None,
                "shift": [0, 0], "ink_pixels": int(ink.sum())}
    if mode == "mnist":
        if matrix.shape != (28, 28):
            raise HTTPException(422, "MNIST 样本必须保持原始 28×28 图像。")
        return matrix.astype(np.float32), metadata
    if mode != "canvas":
        raise HTTPException(422, "未知的预处理模式。")
    from PIL import Image

    ys, xs = np.nonzero(ink)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    metadata["bounding_box"] = [left, top, right, bottom]
    cropped = np.rint(matrix[top:bottom, left:right]).astype(np.uint8)
    scale = 20.0 / max(cropped.shape)
    target_width = max(1, int(round(cropped.shape[1] * scale)))
    target_height = max(1, int(round(cropped.shape[0] * scale)))
    resized = Image.fromarray(cropped).resize((target_width, target_height), Image.Resampling.LANCZOS)
    centered = Image.new("L", (28, 28), 0)
    centered.paste(resized, ((28 - target_width) // 2, (28 - target_height) // 2))
    result = np.array(centered, dtype=np.uint8)
    mass = float(result.sum())
    if not mass or int(result.max()) <= 16:
        raise HTTPException(422, "笔迹过浅，请写清楚一些再识别。")
    yy, xx = np.indices(result.shape)
    dx = int(round(13.5 - float((result * xx).sum() / mass)))
    dy = int(round(13.5 - float((result * yy).sum() / mass)))
    result = translate_without_wrap(result, dx, dy)
    metadata["shift"] = [dx, dy]
    return result, metadata


def _finite_number(value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError("number outside range")
    return float(value)


def _read_json(path: Path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("metadata exceeds size limit")
    return json.loads(path.read_text(encoding="utf-8"))


class InferenceService:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.example_signature = None
        self.raw_examples: dict[int, dict] = {}

    @staticmethod
    def _owned_file(base: Path, candidate: Path) -> Path:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
            raise ValueError("checkpoint path is invalid")
        return resolved

    @staticmethod
    def _descriptor(raw, source: str, path: Path, created_at, expected_sha=None):
        model_id = raw["id"]
        architecture = raw["architecture"]
        if not isinstance(model_id, str) or not re.fullmatch(r"[0-9a-f]{32}", model_id):
            raise ValueError("invalid model ID")
        if architecture not in {"mlp", "cnn"} or raw.get("run_id", model_id) != model_id:
            raise ValueError("invalid model architecture")
        epoch = raw["checkpoint_epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("invalid checkpoint epoch")
        validation_loss = _finite_number(raw["validation_loss"], 0)
        accuracy = raw.get("test_accuracy")
        if accuracy is not None:
            accuracy = _finite_number(accuracy, 0, 1)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"id": model_id, "architecture": architecture, "run_id": model_id,
                "checkpoint_epoch": epoch, "validation_loss": validation_loss,
                "test_accuracy": accuracy, "source": source, "sha256": digest,
                "_path": path, "_created_at": str(created_at), "_expected_sha": expected_sha}

    def _entries(self) -> list[dict]:
        entries = {}
        models_root = self.root / "models"
        manifest_path = models_root / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = _read_json(manifest_path)
                if manifest.get("format") != "zhiwei-mnist-models-v1":
                    raise ValueError("unknown model manifest")
                for raw in manifest.get("models", []):
                    try:
                        # Filenames are relative leaf names in the model bundle.
                        filename = raw["filename"]
                        if filename not in {"mlp-best.pt", "cnn-best.pt"}:
                            raise ValueError("unknown checkpoint filename")
                        path = self._owned_file(models_root, models_root / filename)
                        entry = self._descriptor(raw, "bundle", path, raw.get("created_at", ""), raw.get("sha256"))
                        entries[entry["id"]] = entry
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        runs_root = self.root / "runs" / "mnist"
        if runs_root.is_dir():
            for directory in runs_root.iterdir():
                if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                    continue
                try:
                    record = _read_json(self._owned_file(runs_root, directory / "run.json"))
                    if record.get("status") != "completed" or record.get("id") != directory.name:
                        continue
                    path = self._owned_file(runs_root, directory / "best.pt")
                    raw = {"id": record["id"], "architecture": record["config"]["model"],
                           "checkpoint_epoch": record["best_epoch"], "validation_loss": record["best_validation_loss"],
                           "test_accuracy": (record.get("evaluation") or {}).get("accuracy")}
                    entry = self._descriptor(raw, "run", path, record.get("created_at", ""))
                    entries[entry["id"]] = entry
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    continue
        return sorted(entries.values(), key=lambda row: (row["_created_at"], row["id"]), reverse=True)

    def models(self):
        entries = self._entries()
        runtime = runtime_status()
        return {
            "runtime": {"available": runtime["available"], "torch_version": runtime["torch_version"], "device": "cpu"},
            "models": [{name: value for name, value in row.items() if not name.startswith("_")} for row in entries],
            "defaults": {name: next((row["id"] for row in entries if row["architecture"] == name), None) for name in ("mlp", "cnn")},
        }

    def _load_model(self, entry):
        path = entry["_path"]
        try:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size, entry["sha256"], entry["architecture"], entry["checkpoint_epoch"])
            with self.lock:
                if entry["_expected_sha"] and entry["_expected_sha"] != entry["sha256"]:
                    raise ValueError("checkpoint checksum mismatch")
                cached = self.cache.get(entry["id"])
                if cached and cached["signature"] == signature:
                    self.cache.move_to_end(entry["id"])
                    return cached
                torch = torch_runtime()
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if checkpoint.get("config", {}).get("model") != entry["architecture"]:
                    raise ValueError("checkpoint architecture mismatch")
                if checkpoint.get("epoch") != entry["checkpoint_epoch"]:
                    raise ValueError("checkpoint epoch mismatch")
                checkpoint_loss = _finite_number(checkpoint["validation_loss"], 0)
                if not math.isclose(checkpoint_loss, entry["validation_loss"], abs_tol=1e-10, rel_tol=1e-10):
                    raise ValueError("checkpoint validation loss mismatch")
                normalization = checkpoint["normalization"]
                mean = _finite_number(normalization["mean"])
                std = _finite_number(normalization["std"])
                if std <= 0:
                    raise ValueError("normalization std must be positive")
                # Build shapes on the meta device rather than randomly initializing
                # CPU parameters. Loading a model must not perturb the RNG of a
                # concurrently initialized training experiment.
                with torch.device("meta"):
                    model = build_model(entry["architecture"])
                model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
                model.to(device="cpu", dtype=torch.float32)
                if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
                    raise ValueError("checkpoint has nonfinite parameters")
                model.eval()
                cached = {"model": model, "normalization": {"mean": mean, "std": std}, "signature": signature}
                self.cache[entry["id"]] = cached
                self.cache.move_to_end(entry["id"])
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
                return cached
        except Exception as exc:
            raise HTTPException(503, "模型权重无法安全加载，请重新导出第三期的最佳权重。") from exc

    def predict(self, request: PredictRequest):
        started = time.perf_counter()
        if not runtime_status()["available"]:
            raise HTTPException(503, "缺少 PyTorch / torchvision，请先安装项目依赖。")
        catalog = {row["id"]: row for row in self._entries()}
        try:
            entries = [catalog[model_id] for model_id in request.model_ids]
        except KeyError as exc:
            raise HTTPException(404, "所选模型不存在，或没有已完成训练的最佳权重。") from exc
        image, metadata = preprocess_pixels(request.pixels, request.mode)
        loaded = [self._load_model(entry) for entry in entries]
        normalization = loaded[0]["normalization"]
        if any(item["normalization"] != normalization for item in loaded[1:]):
            raise HTTPException(503, "所选模型的标准化参数不一致，请分别识别。")
        metadata.update(processed_pixels=image.tolist(), normalization=normalization)
        torch = torch_runtime()
        pixels = torch.as_tensor(image.copy(), dtype=torch.float32).reshape(1, 1, 28, 28)
        inputs = pixels.div(255.0).sub(normalization["mean"]).div(normalization["std"])
        predictions = []
        with torch.inference_mode():
            for entry, cached in zip(entries, loaded):
                forward_started = time.perf_counter()
                logits = cached["model"](inputs)
                if tuple(logits.shape) != (1, 10) or not torch.isfinite(logits).all().item():
                    raise HTTPException(503, "模型未产生有效的十分类输出，请重新检查权重。")
                scores = logits.softmax(1)[0].cpu().numpy()
                elapsed = (time.perf_counter() - forward_started) * 1000
                ranking = np.argsort(-scores, kind="stable")
                predictions.append({
                    "model_id": entry["id"], "architecture": entry["architecture"], "run_id": entry["run_id"],
                    "checkpoint_epoch": entry["checkpoint_epoch"], "predicted_label": int(ranking[0]),
                    "probabilities": scores.tolist(),
                    "top_k": [{"label": int(label), "score": float(scores[label])} for label in ranking[:3]],
                    "latency_ms": elapsed,
                })
        return {"request_id": uuid4().hex, "preprocessing": metadata,
                "predictions": predictions, "total_latency_ms": (time.perf_counter() - started) * 1000}

    @staticmethod
    def _validated_example(sample, label):
        if sample.get("source") != "mnist-test" or sample.get("label") != label:
            raise ValueError("invalid example source")
        index = sample["index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 10000:
            raise ValueError("invalid example index")
        pixels = np.asarray(sample["pixels"], dtype=np.float64)
        if pixels.shape != (28, 28) or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("invalid example pixels")
        return {"index": index, "label": label, "pixels": pixels.tolist(), "source": "mnist-test"}

    def example(self, label: int):
        bundle_path = self.root / "models" / "examples.json"
        if bundle_path.is_file():
            try:
                bundle = _read_json(bundle_path)
                if bundle.get("format") == "zhiwei-mnist-examples-v1":
                    for sample in bundle.get("samples", []):
                        if sample.get("label") == label:
                            return self._validated_example(sample, label)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        raw = self.root / "data" / "MNIST" / "raw"
        images_path, labels_path = raw / "t10k-images-idx3-ubyte", raw / "t10k-labels-idx1-ubyte"
        try:
            signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in (images_path, labels_path))
            with self.lock:
                if self.example_signature != signature:
                    images_bytes, labels_bytes = images_path.read_bytes(), labels_path.read_bytes()
                    magic, count, rows, columns = struct.unpack(">IIII", images_bytes[:16])
                    labels_magic, label_count = struct.unpack(">II", labels_bytes[:8])
                    if magic != 2051 or labels_magic != 2049 or count != label_count or not 1 <= count <= 10000 or (rows, columns) != (28, 28):
                        raise ValueError("invalid MNIST IDX metadata")
                    if len(images_bytes) != 16 + count * 784 or len(labels_bytes) != 8 + count:
                        raise ValueError("incomplete MNIST IDX data")
                    images = np.frombuffer(images_bytes, dtype=np.uint8, offset=16).reshape(count, 28, 28)
                    labels = np.frombuffer(labels_bytes, dtype=np.uint8, offset=8)
                    result = {}
                    for digit in range(10):
                        matching = np.flatnonzero(labels == digit)
                        if len(matching):
                            index = int(matching[0])
                            result[digit] = {"index": index, "label": digit, "pixels": images[index].tolist(), "source": "mnist-test"}
                    self.raw_examples = result
                    self.example_signature = signature
                return copy.deepcopy(self.raw_examples[label])
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            raise HTTPException(404, "本地 MNIST 测试样本尚未缓存，请到第三期准备数据，或直接在画布中书写。") from exc


service = InferenceService()
router = APIRouter(prefix="/api/inference", tags=["第四期：模型推理"])


@router.get("/models")
def models():
    return service.models()


@router.post("/predict")
def predict(request: PredictRequest):
    return service.predict(request)


@router.get("/examples")
def example(label: Annotated[int, Query(ge=0, le=9)] = 7):
    return service.example(label)
