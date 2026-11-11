"""CPU causal Transformer with persistent, bounded training transactions."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
from pathlib import Path
import re
import threading
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path as ApiPath
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.language import CorpusRequest, DEFAULT_CORPUS, vocabulary
from backend.mnist_engine import ROOT, torch_runtime


class RunConfig(CorpusRequest):
    context_length: Literal[8, 16, 32] = 16
    d_model: Literal[32, 64] = 32
    heads: Literal[2, 4] = 2
    layers: Literal[1, 2] = 2
    batch_size: Literal[8, 16, 32] = 16
    learning_rate: float = Field(default=0.003, ge=0.0001, le=0.01, allow_inf_nan=False)
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_step: int = Field(strict=True, ge=0, le=2000)
    steps: int = Field(default=25, strict=True, ge=1, le=25)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default="小猫", min_length=1, max_length=128)
    length: int = Field(default=60, strict=True, ge=1, le=120)
    temperature: float = Field(default=0.8, ge=0.1, le=2, allow_inf_nan=False)
    top_k: Literal[0, 5, 10] = 0
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)
    checkpoint: Literal["latest", "best"] = "latest"


def build_transformer(vocab_size, config):
    """Build without consuming process-global RNG (including during construction)."""
    torch = torch_runtime()
    nn = torch.nn
    width, heads, depth = config["d_model"], config["heads"], config["layers"]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width)
            self.projection = nn.Linear(width, width)
            self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

        def forward(self, x):
            batch, length, _ = x.shape
            q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, heads, width // heads).permute(2, 0, 3, 1, 4).unbind(0)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // heads)
            future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            attention = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
            mixed = (attention @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
            x = x + self.projection(mixed)
            return x + self.ff(self.norm2(x)), attention

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, width)
            self.position_embedding = nn.Embedding(config["context_length"], width)
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(width)
            self.output = nn.Linear(width, vocab_size)

        def forward(self, ids, return_attention=False):
            if not 1 <= ids.shape[1] <= config["context_length"]:
                raise ValueError("输入长度超出上下文窗口")
            x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
            for block in self.blocks:
                x, attention = block(x)
            logits = self.output(self.norm(x))
            return (logits, attention) if return_attention else logits

    with torch.device("meta"):
        model = TinyTransformer()
    model.to_empty(device="cpu")
    generator = torch.Generator().manual_seed(config["seed"])
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.normal_(0, 0.02, generator=generator)
                if getattr(module, "bias", None) is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.fill_(1)
                module.bias.zero_()
    return model


def encode_parts(config, vocab):
    torch = torch_runtime()
    lookup = {char: index for index, char in enumerate(vocab)}
    corpus = config["corpus"]
    cut = int(len(corpus) * 0.8)
    return [torch.tensor([lookup.get(char, 0) for char in text], dtype=torch.long) for text in (corpus[:cut], corpus[cut:])]


def evaluate_model(model, parts, context_length):
    """Each split's N-1 target positions scored once, in independent causal blocks."""
    torch = torch_runtime()
    model.eval()
    losses = []
    with torch.inference_mode():
        for ids in parts:
            total = 0.0
            for start in range(0, len(ids) - 1, context_length):
                stop = min(start + context_length, len(ids) - 1)
                logits = model(ids[start:stop].unsqueeze(0))
                total += torch.nn.functional.cross_entropy(logits[0], ids[start + 1:stop + 1], reduction="sum").item()
            losses.append(total / (len(ids) - 1))
    if not all(math.isfinite(loss) for loss in losses):
        raise FloatingPointError("损失非有限，请降低学习率并重新训练。")
    return {"train_loss": losses[0], "validation_loss": losses[1], "perplexity": math.exp(min(losses[1], 700))}


def report(checkpoint):
    keys = ("id", "created_at", "config", "vocab", "step", "best_step", "parameter_count", "history")
    result = {key: copy.deepcopy(checkpoint[key]) for key in keys}
    corpus = checkpoint["config"]["corpus"]
    cut = int(len(corpus) * 0.8)
    lookup = set(checkpoint["vocab"])
    result.update(train_tokens=cut, validation_tokens=len(corpus) - cut,
                  unknown_validation=sum(char not in lookup for char in corpus[cut:]),
                  training_context=min(checkpoint["config"]["context_length"], cut - 1))
    return result


class TransformerService:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else ROOT / "runs" / "transformer"
        self.lock = threading.RLock()

    def _path(self, run_id):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise HTTPException(404, "训练记录不存在")
        return self.root / run_id / "checkpoint.pt"

    def _load(self, run_id):
        path = self._path(run_id)
        if not path.is_file():
            raise HTTPException(404, "训练记录不存在")
        try:
            return torch_runtime().load(path, map_location="cpu", weights_only=True)
        except (ImportError, ModuleNotFoundError) as error:
            raise HTTPException(503, "请安装 requirements.txt 中的 PyTorch 依赖") from error
        except Exception as error:
            raise HTTPException(500, "检查点无法读取，请检查本机文件") from error

    def _save(self, checkpoint):
        path = self._path(checkpoint["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{uuid4().hex}.tmp")
        try:
            torch_runtime().save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, config):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "Transformer 正在计算，请稍后重试")
        try:
            vocab = vocabulary(config["corpus"])
            try:
                model = build_transformer(len(vocab), config)
            except (ImportError, ModuleNotFoundError) as error:
                raise HTTPException(503, "请安装 requirements.txt 中的 PyTorch 依赖") from error
            torch = torch_runtime()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
            metric = {"step": 0, **evaluate_model(model, encode_parts(config, vocab), config["context_length"])}
            checkpoint = {
                "version": 1, "kind": "causal-character-transformer", "id": uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(), "config": config, "vocab": vocab,
                "step": 0, "best_step": 0, "best_loss": metric["validation_loss"],
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "history": [metric], "state_dict": model.state_dict(),
                "best_state_dict": copy.deepcopy(model.state_dict()), "optimizer": optimizer.state_dict(),
                "generator_state": torch.Generator().manual_seed(config["seed"]).get_state(),
            }
            self._save(checkpoint)
            return report(checkpoint)
        finally:
            self.lock.release()

    def get(self, run_id):
        with self.lock:
            return report(self._load(run_id))

    def list(self):
        with self.lock:
            if not self.root.exists():
                return []
            paths = sorted(self.root.glob("*/checkpoint.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
            results = []
            for path in paths[:30]:
                try:
                    results.append(report(self._load(path.parent.name)))
                except HTTPException as error:
                    if error.status_code != 500:
                        raise
            return results

    def train(self, run_id, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "Transformer 正在计算，请稍后刷新记录")
        try:
            checkpoint = self._load(run_id)
            if checkpoint["step"] != request.expected_step:
                raise HTTPException(409, "训练步数已改变，请刷新记录后继续，避免重复提交")
            if checkpoint["step"] + request.steps > 2000:
                raise HTTPException(422, "每次实验最多 2,000 步，请创建新实验")
            config = checkpoint["config"]
            torch = torch_runtime()
            model = build_transformer(len(checkpoint["vocab"]), config)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
            optimizer.load_state_dict(checkpoint["optimizer"])
            generator = torch.Generator()
            generator.set_state(checkpoint["generator_state"])
            parts = encode_parts(config, checkpoint["vocab"])
            ids = parts[0]
            length = min(config["context_length"], len(ids) - 1)
            model.train()
            for _ in range(request.steps):
                starts = torch.randint(len(ids) - length, (config["batch_size"],), generator=generator)
                indices = starts[:, None] + torch.arange(length)
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids[indices])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(checkpoint["vocab"])), ids[indices + 1].reshape(-1))
                if not torch.isfinite(loss).item():
                    raise HTTPException(422, "训练损失非有限，本批未保存，请重新初始化并降低学习率")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            checkpoint["step"] += request.steps
            metric = {"step": checkpoint["step"], **evaluate_model(model, parts, config["context_length"])}
            if metric["validation_loss"] < checkpoint["best_loss"]:
                checkpoint.update(best_loss=metric["validation_loss"], best_step=checkpoint["step"], best_state_dict=copy.deepcopy(model.state_dict()))
            checkpoint["history"].append(metric)
            checkpoint.update(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), generator_state=generator.get_state())
            self._save(checkpoint)
            return report(checkpoint)
        finally:
            self.lock.release()

    def generate(self, run_id, request):
        with self.lock:
            checkpoint = self._load(run_id)
        config, vocab = checkpoint["config"], checkpoint["vocab"]
        lookup = {char: index for index, char in enumerate(vocab)}
        if any(char not in lookup for char in request.prompt):
            raise HTTPException(422, "提示词包含词表外字符，请从本实验语料取字")
        torch = torch_runtime()
        model = build_transformer(len(vocab), config)
        model.load_state_dict(checkpoint["best_state_dict"] if request.checkpoint == "best" else checkpoint["state_dict"])
        model.eval()
        generator = torch.Generator().manual_seed(request.seed)
        sequence = [lookup[char] for char in request.prompt]
        context_ids = sequence[-config["context_length"]:]
        output = []
        with torch.inference_mode():
            for index in range(request.length):
                context = torch.tensor([sequence[-config["context_length"]:]], dtype=torch.long)
                logits, attention = model(context, return_attention=True)
                scores = logits[0, -1] / request.temperature
                if request.top_k:
                    indices = torch.topk(scores, min(request.top_k, len(vocab))).indices
                    mask = torch.ones_like(scores, dtype=torch.bool)
                    mask[indices] = False
                    scores = scores.masked_fill(mask, float("-inf"))
                probabilities = scores.softmax(dim=-1)
                if index == 0:
                    first = probabilities.tolist()
                    initial_attention = attention[0].tolist()
                token = int(torch.multinomial(probabilities, 1, generator=generator).item())
                sequence.append(token)
                output.append(token)
        return {
            "run_id": run_id, "step": checkpoint["best_step"] if request.checkpoint == "best" else checkpoint["step"],
            "checkpoint": request.checkpoint, "prompt": request.prompt,
            "continuation": "".join(vocab[token] for token in output), "token_ids": output,
            "distribution": [{"token": vocab[i], "probability": first[i]} for i in sorted(range(len(vocab)), key=lambda i: -first[i])[:8]],
            "context_tokens": [vocab[i] for i in context_ids], "attention": initial_attention,
            "attention_layer": config["layers"],
        }


service = TransformerService()
router = APIRouter(prefix="/api/transformer", tags=["Transformer 语言模型"])
RunId = Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")]


@router.get("/status")
def status():
    return {"available": importlib.util.find_spec("torch") is not None, "device": "cpu", "corpus": DEFAULT_CORPUS, "max_steps": 2000}


@router.get("/runs")
def list_runs():
    return service.list()


@router.post("/runs")
def create_run(request: RunConfig):
    return service.create(request.model_dump())


@router.get("/runs/{run_id}")
def get_run(run_id: RunId):
    return service.get(run_id)


@router.post("/runs/{run_id}/train")
def train_run(run_id: RunId, request: TrainingRequest):
    return service.train(run_id, request)


@router.post("/runs/{run_id}/generate")
def generate_text(run_id: RunId, request: GenerationRequest):
    return service.generate(run_id, request)


@router.get("/runs/{run_id}/checkpoint")
def download_checkpoint(run_id: RunId):
    with service.lock:
        service._load(run_id)
        path = service._path(run_id)
    return FileResponse(path, media_type="application/octet-stream", filename=f"transformer-{run_id[:8]}.pt")
