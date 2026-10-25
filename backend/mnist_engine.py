"""Real MNIST training with a single cancellable CPU worker.

PyTorch is deliberately imported only when a dataset is prepared. The numerical
experiments in earlier chapters do not pay its startup cost. Tests inject small
synthetic tensors; production always uses torchvision's official MNIST files.
"""

from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

import numpy as np
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_STATUSES = {"queued", "preparing", "training", "validating", "evaluating", "cancelling"}
THREADS = max(1, min(4, os.cpu_count() or 1))
_runtime_lock = threading.Lock()
_torch = None


def torch_runtime():
    global _torch
    with _runtime_lock:
        if _torch is None:
            import torch

            torch.set_num_threads(THREADS)
            _torch = torch
    return _torch


def runtime_status() -> dict:
    versions = {}
    for name in ("torch", "torchvision"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "available": all(versions.values()),
        "torch_version": versions["torch"],
        "torchvision_version": versions["torchvision"],
        "device": "cpu",
        "threads": THREADS,
    }


def build_model(name: str):
    torch = torch_runtime()
    nn = torch.nn
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    raise ValueError("unknown model")


def allocate_counts(counts: np.ndarray, total: int) -> np.ndarray:
    """Allocate exactly total samples proportionally, with deterministic ties."""
    if total < 0 or total > int(counts.sum()):
        raise ValueError("invalid stratified sample count")
    if not total:
        return np.zeros_like(counts)
    desired = counts * (total / counts.sum())
    allocated = np.floor(desired).astype(np.int64)
    priority = np.argsort(-(desired - allocated), kind="stable")
    allocated[priority[:total - int(allocated.sum())]] += 1
    return allocated


def split_indices(labels, seed: int, validation_ratio: float, train_limit: int | None) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=10)
    validation_sizes = allocate_counts(counts, int(round(len(labels) * validation_ratio)))
    train_parts, validation_parts = [], []
    for label in range(10):
        candidates = rng.permutation(np.flatnonzero(labels == label))
        n = validation_sizes[label]
        validation_parts.append(candidates[:n])
        train_parts.append(candidates[n:])
    full_train = rng.permutation(np.concatenate(train_parts))
    validation = rng.permutation(np.concatenate(validation_parts))
    limit = min(train_limit or len(full_train), len(full_train))
    if limit == len(full_train):
        effective = full_train.copy()
    else:
        sizes = allocate_counts(np.bincount(labels[full_train], minlength=10), limit)
        parts = [rng.permutation(part)[:sizes[label]] for label, part in enumerate(train_parts)]
        effective = rng.permutation(np.concatenate(parts))
    return {"full_train": full_train, "validation": validation, "effective_train": effective}


def classification_report(labels: np.ndarray, predictions: np.ndarray, loss: float, checkpoint_epoch: int) -> dict:
    confusion = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    metrics = []
    for label in range(10):
        true_positive = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        metrics.append({
            "label": label, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": support,
        })
    correct = int(np.trace(confusion))
    return {
        "loss": float(loss), "accuracy": correct / len(labels), "total": int(len(labels)),
        "confusion_matrix": confusion.tolist(), "class_metrics": metrics,
        "error_count": int(len(labels) - correct), "checkpoint_epoch": checkpoint_epoch,
    }


class TrainingCancelled(Exception):
    pass


class MnistService:
    def __init__(self, root: Path = ROOT, dataset_provider: Callable | None = None):
        self.root = Path(root)
        self.data_root = self.root / "data"
        self.runs_root = self.root / "runs" / "mnist"
        self.provider = dataset_provider
        self.lock = threading.RLock()
        self.finished = threading.Condition(self.lock)
        self.data = None
        self.raw_test_images = None
        self.dataset_state = "missing"
        self.dataset_message = "准备 MNIST 后即可查看数据划分并训练模型。"
        self.dataset_error = None
        self.active_run_id = None
        self.runs: dict[str, dict] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.start_times: dict[str, float] = {}
        self.workers: dict[str, threading.Thread] = {}
        self.recovered_run_ids: set[str] = set()
        self._restore()

    def _restore(self):
        if not self.runs_root.exists():
            return
        for directory in self.runs_root.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                if run.get("id") != directory.name:
                    continue
                if run["status"] in ACTIVE_STATUSES:
                    run.update(status="interrupted", message="服务重启中断了任务；已完成的记录与最佳权重保留。")
                    # Importing the app (including test discovery) must be read-only.
                    # The actual ASGI startup explicitly persists recovered states.
                    self.recovered_run_ids.add(run["id"])
                run["checkpoint_available"] = (directory / "best.pt").exists()
                self.runs[run["id"]] = run
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def persist_interrupted(self):
        """Called by real API startup, never by constructors or ordinary imports."""
        with self.lock:
            for run_id in list(self.recovered_run_ids):
                path = self.runs_root / run_id / "run.json"
                try:
                    # Do not overwrite a completed record that changed after import.
                    current = json.loads(path.read_text(encoding="utf-8"))
                    if current["status"] in ACTIVE_STATUSES:
                        current.update(status="interrupted", message="服务重启中断了任务；已完成的记录与最佳权重保留。")
                        self._atomic_json(path, current)
                    current["checkpoint_available"] = (path.parent / "best.pt").exists()
                    self.runs[run_id] = current
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                finally:
                    self.recovered_run_ids.discard(run_id)

    @staticmethod
    def _atomic_json(path: Path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)

    def _persist(self, run_id: str):
        self._atomic_json(self.runs_root / run_id / "run.json", self.runs[run_id])

    def status(self) -> dict:
        with self.lock:
            return {
                "runtime": runtime_status(),
                "dataset": {
                    "status": self.dataset_state, "message": self.dataset_message,
                    "error": self.dataset_error, "official_train": 60000, "official_test": 10000,
                },
                "active_run_id": self.active_run_id,
            }

    def prepare(self):
        with self.lock:
            if self.dataset_state in {"preparing", "ready"}:
                return self.status()
            if not runtime_status()["available"]:
                self.dataset_state = "failed"
                self.dataset_message = "缺少 PyTorch 或 torchvision，请安装 requirements.txt 中的依赖。"
                self.dataset_error = self.dataset_message
                return self.status()
            self.dataset_state = "preparing"
            self.dataset_message = "正在加载 PyTorch 并校验或下载官方 MNIST，首次准备需要一些时间。"
            self.dataset_error = None
            threading.Thread(target=self._prepare_worker, name="mnist-dataset", daemon=True).start()
            return self.status()

    def _prepare_worker(self):
        try:
            torch = torch_runtime()
            if self.provider:
                data = self.provider()
            else:
                from torchvision.datasets import MNIST

                train = MNIST(str(self.data_root), train=True, download=True)
                test = MNIST(str(self.data_root), train=False, download=True)
                data = {"train_images": train.data, "train_labels": train.targets,
                        "test_images": test.data, "test_labels": test.targets}
            converted = {}
            for source in ("train", "test"):
                images = torch.as_tensor(data[f"{source}_images"], dtype=torch.uint8)
                labels = torch.as_tensor(data[f"{source}_labels"], dtype=torch.int64)
                if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28) or len(images) != len(labels) or not len(labels):
                    raise ValueError("MNIST 图片形状或标签长度错误")
                if labels.ndim != 1 or labels.min().item() < 0 or labels.max().item() > 9:
                    raise ValueError("MNIST 标签必须是 0–9")
                converted[f"{source}_images"] = images
                converted[f"{source}_labels"] = labels
            with self.lock:
                self.data = converted
                self.dataset_state = "ready"
                self.dataset_message = "MNIST 已就绪；官方训练集用于训练/验证，官方测试集独立保留。"
                self.dataset_error = None
        except Exception as exc:
            with self.lock:
                self.dataset_state = "failed"
                self.dataset_message = "MNIST 准备失败，可重试；请检查依赖、网络和数据目录。"
                self.dataset_error = f"{type(exc).__name__}: {exc}"

    def _require_data(self):
        if self.dataset_state != "ready" or self.data is None:
            raise HTTPException(status_code=409, detail="请先准备 MNIST 数据集，等待数据状态变为就绪。")
        return self.data

    @staticmethod
    def _sample_indices(indices: np.ndarray, labels: np.ndarray, count=12):
        selected = []
        for label in range(10):
            candidates = indices[labels[indices] == label]
            if len(candidates):
                selected.append(int(candidates[0]))
        selected_set = set(selected)
        selected.extend(int(index) for index in indices if int(index) not in selected_set)
        return selected[:count]

    def split(self, config: dict) -> dict:
        data = self._require_data()
        labels = data["train_labels"].numpy()
        test_labels = data["test_labels"].numpy()
        indices = split_indices(labels, config["seed"], config["validation_ratio"], config["train_limit"])
        test_indices = np.arange(len(test_labels), dtype=np.int64)
        samples = []
        for name, source, rows in [
            ("train", "train", indices["effective_train"]),
            ("validation", "train", indices["validation"]),
            ("test", "test", test_indices),
        ]:
            source_labels = data[f"{source}_labels"].numpy()
            for index in self._sample_indices(rows, source_labels):
                samples.append({"split": name, "index": index, "label": int(source_labels[index]),
                                "pixels": data[f"{source}_images"][index].tolist()})
        return {
            "seed": config["seed"], "validation_ratio": config["validation_ratio"],
            "train_count": int(len(indices["full_train"])), "validation_count": int(len(indices["validation"])),
            "test_count": int(len(test_labels)), "effective_train_count": int(len(indices["effective_train"])),
            "train_class_counts": np.bincount(labels[indices["effective_train"]], minlength=10).tolist(),
            "validation_class_counts": np.bincount(labels[indices["validation"]], minlength=10).tolist(),
            "test_class_counts": np.bincount(test_labels, minlength=10).tolist(), "samples": samples,
        }

    def _run(self, run_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id) or run_id not in self.runs:
            raise HTTPException(status_code=404, detail="训练任务不存在。")
        return self.runs[run_id]

    def get_run(self, run_id: str) -> dict:
        with self.lock:
            run = copy.deepcopy(self._run(run_id))
            if run_id in self.start_times:
                run["elapsed_seconds"] = time.monotonic() - self.start_times[run_id]
            return run

    def list_runs(self):
        with self.lock:
            return {"runs": [self.get_run(run_id) for run_id in sorted(self.runs, key=lambda key: self.runs[key]["created_at"], reverse=True)]}

    def create_run(self, config: dict) -> dict:
        self._require_data()
        with self.lock:
            if self.active_run_id is not None:
                raise HTTPException(status_code=409, detail="已有训练任务运行中，请等待完成或先取消。")
            data = self.data
            indices = split_indices(data["train_labels"].numpy(), config["seed"], config["validation_ratio"], config["train_limit"])
            run_id = uuid4().hex
            run = {
                "id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "status": "queued",
                "config": dict(config), "epoch": 0, "batch": 0,
                "batches_per_epoch": math.ceil(len(indices["effective_train"]) / config["batch_size"]),
                "processed_samples": 0, "train_samples": int(len(indices["effective_train"])),
                "validation_samples": int(len(indices["validation"])), "test_samples": int(len(data["test_labels"])),
                "history": [], "best_epoch": None, "best_validation_loss": None,
                "parameter_count": 0, "device": "cpu", "message": "任务已创建，等待后台训练。",
                "error": None, "evaluation": None, "elapsed_seconds": 0.0, "checkpoint_available": False,
            }
            self.runs[run_id] = run
            self.cancel_events[run_id] = threading.Event()
            self.active_run_id = run_id
            self.start_times[run_id] = time.monotonic()
            try:
                self._persist(run_id)
                worker = threading.Thread(target=self._train_worker, args=(run_id, indices), name=f"mnist-{run_id[:8]}", daemon=True)
                self.workers[run_id] = worker
                response = copy.deepcopy(run)
                worker.start()
                return response
            except Exception as exc:
                run.update(status="failed", message="无法创建训练任务，请检查存储目录与可用空间。",
                           error=f"{type(exc).__name__}: {exc}")
                try:
                    self._persist(run_id)
                except Exception:
                    pass
                finally:
                    self.active_run_id = None
                    self.start_times.pop(run_id, None)
                    self.cancel_events.pop(run_id, None)
                    self.workers.pop(run_id, None)
                    self.finished.notify_all()
                raise HTTPException(status_code=500, detail=run["message"]) from exc

    def wait_for_run(self, run_id: str, timeout: float = 20):
        """Internal fixture/smoke-test helper; HTTP clients poll the normal Run."""
        deadline = time.monotonic() + timeout
        with self.finished:
            self._run(run_id)
            while run_id in self.workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("training worker did not finish")
                self.finished.wait(remaining)
            return self.get_run(run_id)

    def _update(self, run_id, persist=True, **fields):
        with self.lock:
            run = self.runs[run_id]
            run.update(fields)
            if run_id in self.start_times:
                run["elapsed_seconds"] = time.monotonic() - self.start_times[run_id]
            if persist:
                self._persist(run_id)

    def _check_cancel(self, run_id):
        if self.cancel_events[run_id].is_set():
            raise TrainingCancelled()

    def cancel(self, run_id):
        with self.lock:
            run = self._run(run_id)
            if run["status"] in ACTIVE_STATUSES:
                self.cancel_events[run_id].set()
                self._update(run_id, status="cancelling", message="正在取消，将在当前批次结束后停止。")
            return self.get_run(run_id)

    def _loader(self, source, indices, batch_size, shuffle=False, generator=None):
        torch = torch_runtime()
        data = self.data
        rows = torch.as_tensor(np.asarray(indices).copy(), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data[f"{source}_images"][rows], data[f"{source}_labels"][rows])
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)

    @staticmethod
    def _normalize(images):
        return images.float().unsqueeze(1).div_(255.0).sub_(0.1307).div_(0.3081)

    def _evaluate(self, model, loader, run_id, collect=False):
        torch = torch_runtime()
        model.eval()
        loss_sum, correct, count = 0.0, 0, 0
        probabilities, all_labels = [], []
        with torch.inference_mode():
            for images, labels in loader:
                self._check_cancel(run_id)
                logits = model(self._normalize(images))
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("评估损失不是有限数，请降低学习率。")
                loss_sum += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
                if collect:
                    probabilities.append(logits.softmax(1).numpy())
                    all_labels.append(labels.numpy())
        return {"loss": loss_sum / count, "accuracy": correct / count,
                "probabilities": np.concatenate(probabilities) if collect else None,
                "labels": np.concatenate(all_labels) if collect else None}

    def _train_worker(self, run_id, indices):
        try:
            self._check_cancel(run_id)
            torch = torch_runtime()
            config = self.runs[run_id]["config"]
            self._update(run_id, status="preparing", message="正在构建模型与互斥的数据加载器。")
            torch.manual_seed(config["seed"])
            model = build_model(config["model"])
            generator = torch.Generator().manual_seed(config["seed"])
            train_loader = self._loader("train", indices["effective_train"], config["batch_size"], True, generator)
            validation_loader = self._loader("train", indices["validation"], 256)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            directory = self.runs_root / run_id
            np.savez_compressed(directory / "split.npz", train_indices=indices["full_train"],
                                validation_indices=indices["validation"], effective_train_indices=indices["effective_train"])
            self._update(run_id, parameter_count=sum(parameter.numel() for parameter in model.parameters()))
            for epoch in range(1, config["epochs"] + 1):
                self._check_cancel(run_id)
                epoch_started = time.monotonic()
                self._update(run_id, status="training", epoch=epoch, batch=0, processed_samples=0,
                             message=f"正在训练第 {epoch}/{config['epochs']} 个 epoch。")
                model.train()
                loss_sum, correct, count = 0.0, 0, 0
                for batch, (images, labels) in enumerate(train_loader, 1):
                    self._check_cancel(run_id)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(self._normalize(images))
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("训练损失不是有限数，请降低学习率。")
                    loss.backward()
                    optimizer.step()
                    count += len(labels)
                    loss_sum += loss.item() * len(labels)
                    correct += (logits.detach().argmax(1) == labels).sum().item()
                    self._update(run_id, persist=batch % 20 == 0, batch=batch, processed_samples=count)
                self._check_cancel(run_id)
                self._update(run_id, status="validating", message=f"正在验证第 {epoch} 个 epoch；测试集未参与。")
                validation = self._evaluate(model, validation_loader, run_id)
                row = {"epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": correct / count,
                       "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                       "seconds": time.monotonic() - epoch_started}
                with self.lock:
                    run = self.runs[run_id]
                    run["history"].append(row)
                    if run["best_validation_loss"] is None or validation["loss"] < run["best_validation_loss"]:
                        temporary = directory / "best.pt.tmp"
                        checkpoint = {
                            "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                            "config": dict(config), "epoch": epoch, "validation_loss": validation["loss"],
                            "normalization": {"mean": 0.1307, "std": 0.3081},
                        }
                        torch.save(checkpoint, temporary)
                        os.replace(temporary, directory / "best.pt")
                        run.update(best_epoch=epoch, best_validation_loss=validation["loss"], checkpoint_available=True)
                    self._persist(run_id)
            self._check_cancel(run_id)
            self._update(run_id, status="evaluating", message="已按验证损失选定最佳模型，正在独立测试集上评估。")
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            test_loader = self._loader("test", np.arange(len(self.data["test_labels"])), 256)
            test = self._evaluate(model, test_loader, run_id, collect=True)
            probabilities, labels = test["probabilities"], test["labels"]
            predictions = probabilities.argmax(1)
            evaluation = classification_report(labels, predictions, test["loss"], checkpoint["epoch"])
            errors = [
                {"index": int(index), "label": int(labels[index]), "prediction": int(predictions[index]),
                 "confidence": float(probabilities[index, predictions[index]]), "probabilities": probabilities[index].tolist()}
                for index in np.flatnonzero(labels != predictions)
            ]
            errors.sort(key=lambda item: (-item["confidence"], item["index"]))
            self._atomic_json(directory / "errors.json", errors)
            self._check_cancel(run_id)
            self._update(run_id, status="completed", evaluation=evaluation,
                         message="训练与独立测试已完成；模型由最低验证损失选择。")
        except TrainingCancelled:
            with self.lock:
                self.runs[run_id].update(status="cancelled", message="训练已取消；已完成 epoch 的记录与最佳权重保留。")
        except Exception as exc:
            with self.lock:
                self.runs[run_id].update(status="failed", error=f"{type(exc).__name__}: {exc}", message="训练任务失败，请查看错误信息后重试。")
        finally:
            with self.lock:
                if run_id in self.start_times:
                    self.runs[run_id]["elapsed_seconds"] = time.monotonic() - self.start_times.pop(run_id)
                try:
                    self._persist(run_id)
                except Exception as exc:
                    run = self.runs[run_id]
                    previous = f"{run['error']}；" if run["error"] else ""
                    run.update(status="failed", error=f"{previous}状态持久化失败：{type(exc).__name__}: {exc}",
                               message="任务状态无法写入磁盘，请检查存储目录与可用空间。")
                finally:
                    # Never leave the service permanently busy, even when the
                    # failure record itself cannot be written to disk.
                    if self.active_run_id == run_id:
                        self.active_run_id = None
                    self.cancel_events.pop(run_id, None)
                    self.workers.pop(run_id, None)
                    self.finished.notify_all()

    def checkpoint_path(self, run_id):
        with self.lock:
            self._run(run_id)
            path = self.runs_root / run_id / "best.pt"
            if not path.is_file():
                raise HTTPException(status_code=409, detail="该任务尚无已验证的最佳模型权重。")
            return path

    def errors(self, run_id: str, true_label=None, predicted_label=None, offset=0, limit=24):
        with self.lock:
            run = self._run(run_id)
            if not run["evaluation"]:
                raise HTTPException(status_code=409, detail="独立测试尚未完成，暂时没有错例结果。")
            path = self.runs_root / run_id / "errors.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
        if true_label is not None:
            rows = [row for row in rows if row["label"] == true_label]
        if predicted_label is not None:
            rows = [row for row in rows if row["prediction"] == predicted_label]
        chosen = copy.deepcopy(rows[offset:offset + limit])
        if chosen:
            if self.data is not None:
                images = self.data["test_images"]
            else:
                if self.raw_test_images is None:
                    image_file = self.data_root / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
                    if not image_file.is_file():
                        raise HTTPException(status_code=409, detail="请先准备 MNIST，以加载错例的原始测试图片。")
                    raw = np.fromfile(image_file, dtype=np.uint8)
                    if len(raw) < 16 or int.from_bytes(raw[:4].tobytes(), "big") != 2051:
                        raise HTTPException(status_code=409, detail="测试图片文件无效，请重新准备 MNIST。")
                    self.raw_test_images = raw[16:].reshape(-1, 28, 28)
                images = self.raw_test_images
            for row in chosen:
                row["pixels"] = images[row["index"]].tolist()
        return {"total": len(rows), "offset": offset, "limit": limit, "items": chosen}


service = MnistService()
