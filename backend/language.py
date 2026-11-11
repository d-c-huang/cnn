"""Small, stateless character bigram language model with exact full-batch SGD."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

router = APIRouter(prefix="/api/language", tags=["语言模型"])
DEFAULT_CORPUS = (
    "春天来了，花开了。小猫在花园里看花。\n"
    "夏天来了，雨落了。小猫在窗边看雨。\n"
    "秋天来了，叶落了。小猫在树下看叶。\n"
    "冬天来了，雪落了。小猫在门前看雪。\n"
    "春天的花园有花，夏天的窗边有雨。\n"
    "秋天的树下有叶，冬天的门前有雪。\n"
    "小猫看花，小猫看雨。小猫看叶，小猫看雪。\n"
    "花开在春天，雨落在夏天。叶落在秋天，雪落在冬天。\n"
)
Scalar = Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]


class CorpusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: str = Field(default=DEFAULT_CORPUS, min_length=40, max_length=2000)

    @model_validator(mode="after")
    def valid_vocabulary(self):
        size = len(set(self.corpus[: int(len(self.corpus) * 0.8)]))
        if not 2 <= size <= 95:
            raise ValueError("训练文本须包含 2–95 个不同字符")
        return self


class ModelRequest(CorpusRequest):
    weights: list[list[Scalar]] = Field(min_length=3, max_length=96)

    @model_validator(mode="after")
    def valid_shape(self):
        size = len(vocabulary(self.corpus))
        if len(self.weights) != size or any(len(row) != size for row in self.weights):
            raise ValueError("权重尺寸必须与训练词表一致")
        return self


class TrainRequest(ModelRequest):
    step: int = Field(default=0, strict=True, ge=0, le=5000)
    steps: int = Field(default=100, strict=True, ge=1, le=200)
    learning_rate: float = Field(default=5, gt=0, le=20, allow_inf_nan=False)

    @model_validator(mode="after")
    def within_budget(self):
        if self.step + self.steps > 5000:
            raise ValueError("最多训练 5000 步，请重新初始化")
        return self


class GenerateRequest(ModelRequest):
    prompt: str = Field(default="小猫", min_length=1, max_length=80)
    temperature: float = Field(default=1, ge=0.1, le=2, allow_inf_nan=False)
    length: int = Field(default=60, strict=True, ge=1, le=120)
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)


def vocabulary(corpus):
    # Build only on the training split; unseen validation characters use ID 0.
    return ["<UNK>", *sorted(set(corpus[: int(len(corpus) * 0.8)]))]


def dataset(corpus):
    vocab = vocabulary(corpus)
    lookup = {char: index for index, char in enumerate(vocab)}
    cut = int(len(corpus) * 0.8)
    parts = [np.array([lookup.get(c, 0) for c in part]) for part in (corpus[:cut], corpus[cut:])]
    counts = np.zeros((len(vocab), len(vocab)))
    np.add.at(counts, (parts[0][:-1], parts[0][1:]), 1)
    return vocab, parts, counts


def log_probs(weights, temperature=1):
    shifted = weights / temperature
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def metrics(weights, parts):
    logs = log_probs(weights)
    losses = [-float(logs[ids[:-1], ids[1:]].mean()) for ids in parts]
    return {"train_loss": losses[0], "validation_loss": losses[1], "perplexity": float(np.exp(losses[1]))}


@router.post("/prepare")
def prepare(request: CorpusRequest):
    vocab, parts, _ = dataset(request.corpus)
    weights = np.zeros((len(vocab), len(vocab)))
    cut = int(len(request.corpus) * 0.8)
    return {
        "corpus": request.corpus,
        "vocab": vocab, "weights": weights.tolist(), "step": 0,
        "train_text": request.corpus[:cut], "validation_text": request.corpus[cut:],
        "train_tokens": len(parts[0]), "validation_tokens": len(parts[1]),
        "unknown_validation": int((parts[1] == 0).sum()),
        "pairs": [{"input": int(x), "target": int(y)} for x, y in zip(parts[0][:-1][:16], parts[0][1:][:16])],
        "history": [{"step": 0, **metrics(weights, parts)}],
    }


@router.post("/train")
def train(request: TrainRequest):
    _, parts, counts = dataset(request.corpus)
    weights = np.array(request.weights, dtype=np.float64)
    total = counts.sum()
    row_counts = counts.sum(axis=1, keepdims=True)
    history = []
    for index in range(request.steps):
        probabilities = np.exp(log_probs(weights))
        gradient = (row_counts * probabilities - counts) / total
        weights -= request.learning_rate * gradient
        history.append({"step": request.step + index + 1, **metrics(weights, parts)})
    return {"weights": weights.tolist(), "step": request.step + request.steps, "history": history}


@router.post("/generate")
def generate(request: GenerateRequest):
    vocab = vocabulary(request.corpus)
    lookup = {char: index for index, char in enumerate(vocab)}
    if any(char not in lookup for char in request.prompt):
        raise HTTPException(422, "提示词包含词表外字符，请从训练文本中选择")
    weights = np.array(request.weights)
    probabilities = np.exp(log_probs(weights, request.temperature))
    rng = np.random.default_rng(request.seed)
    current = lookup[request.prompt[-1]]
    first = probabilities[current].copy()
    output = []
    for _ in range(request.length):
        current = int(rng.choice(len(vocab), p=probabilities[current]))
        output.append(vocab[current])
    return {
        "continuation": "".join(output),
        "distribution": [{"token": vocab[index], "probability": float(first[index])} for index in np.argsort(-first)[:8]],
    }
