"""Pinned public pretrained weights, downloaded explicitly and reused offline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import threading

from fastapi import HTTPException

from backend.mnist_engine import ROOT

MODEL_ID = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
MODEL_SHA256 = "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"
REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PretrainedAssets:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else ROOT / "data" / "pretrained" / "pythia-14m" / MODEL_REVISION
        self.lock = threading.Lock()
        self.state = "idle"
        self.message = "首次使用需下载公开预训练权重与分词器，之后可离线使用。"

    def cached(self):
        marker = self.directory / "zhiwei-base.json"
        if not marker.is_file() or not all((self.directory / file).is_file() for file in REQUIRED_FILES):
            return False
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            return info.get("model_id") == MODEL_ID and info.get("revision") == MODEL_REVISION and info.get("weight_sha256") == MODEL_SHA256
        except (OSError, ValueError):
            return False

    def status(self):
        dependencies = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "safetensors", "huggingface_hub")}
        cached = self.cached()
        return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "state": "ready" if cached else "idle" if self.state == "ready" else self.state,
                "message": "固定版本预训练模型已缓存，可离线微调。" if cached else self.message,
                "dependencies": dependencies, "available": all(dependencies.values()), "device": "cpu"}

    def ensure(self):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "预训练模型正在准备")
        try:
            if self.cached():
                return self.directory
            if not self.status()["available"]:
                raise HTTPException(503, "请安装 requirements.txt 中的 Transformers、PEFT 与 Safetensors 依赖")
            self.state, self.message = "downloading", "正在从 Hugging Face 下载固定版本的 Pythia-14M 权重与分词器…"
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(self.directory),
                              allow_patterns=[*REQUIRED_FILES, "generation_config.json", "README.md"], token=False, max_workers=2)
            if not all((self.directory / file).is_file() for file in REQUIRED_FILES):
                raise RuntimeError("预训练文件不完整")
            from safetensors import safe_open

            with safe_open(str(self.directory / "model.safetensors"), framework="pt", device="cpu") as weights:
                if not list(weights.keys()):
                    raise RuntimeError("权重文件为空")
            info = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                    "weight_sha256": sha256_file(self.directory / "model.safetensors")}
            if info["weight_sha256"] != MODEL_SHA256:
                raise RuntimeError("预训练权重校验值与固定版本不一致")
            (self.directory / "zhiwei-base.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            self.state, self.message = "ready", "预训练模型准备完成。"
            return self.directory
        except Exception:
            self.state, self.message = "failed", "准备失败，请检查网络、模型文件与依赖后重试。"
            raise
        finally:
            self.lock.release()

    def prepare(self):
        if self.cached() or self.state == "downloading":
            return self.status()
        self.state, self.message = "downloading", "正在准备公开模型下载…"

        def worker():
            try:
                self.ensure()
            except Exception:
                pass  # The status endpoint reports the failure without exposing credentials or paths.

        threading.Thread(target=worker, daemon=True, name="lora-pretrained-download").start()
        return self.status()

    def require(self):
        if not self.cached():
            raise HTTPException(409, "请先准备预训练模型，完成下载后再创建微调实验")
        return self.directory


assets = PretrainedAssets()
