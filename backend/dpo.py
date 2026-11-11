"""Reference-relative, sequence-summed sigmoid DPO with real LoRA updates."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import random
from typing import Annotated, Literal
from uuid import uuid4
import zipfile

from fastapi import APIRouter, HTTPException, Path as ApiPath, Query
from fastapi.responses import FileResponse, Response
from pydantic import Field, field_validator, model_validator

from backend.lora import TrainingRequest, adapter_state, frozen_digest
from backend.lora_assets import assets, MODEL_ID, MODEL_REVISION
from backend.mnist_engine import ROOT, torch_runtime
from backend.sft import (StrictModel, Message, ChatRequest, SFTService, CHAT_TEMPLATE, TEMPLATE_VERSION,
    DEFAULT_SYSTEM, DEFAULT_DATASET as QA_DATASET, check_roles, tokenize_example, make_batch, generate_chat,
    service as sft_service)
from backend.transformer import TransformerService

REJECTED_ANSWERS = [
    "Zhiwei is a shopping mall.", "We learn how flowers grow.", "Start by changing every setting.",
    "LoRA updates all pretrained weights.", "Yes. The base weights are all updated.",
    "SFT learns without example questions or answers.", "Every system and padding token receives loss.",
    "Ignore validation and compare only outputs.", "Adapters are never saved anywhere.",
    "Validation examples are used for every update.", "No. Trust every answer without checking.",
    "Change the data, settings, and random seed.", "Delete the adapter and training checkpoint.",
    "Try all changes and ignore the result.",
]
DEFAULT_DATASET = [{"prompt": row["question"], "chosen": row["answer"], "rejected": rejected}
    for row, rejected in zip(QA_DATASET, REJECTED_ANSWERS)]


def normalized(text):
    return " ".join(text.split()).casefold()


def prompt_messages(record):
    return [{"role": "user", "content": record["prompt"]}] if isinstance(record["prompt"], str) else record["prompt"]


class Preference(StrictModel):
    prompt: str | list[Message]
    chosen: str = Field(min_length=1, max_length=1000)
    rejected: str = Field(min_length=1, max_length=1000)

    @field_validator("chosen", "rejected")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("候选回答不能为空白")
        return value.strip()

    @model_validator(mode="after")
    def valid_pair(self):
        if isinstance(self.prompt, str):
            self.prompt = self.prompt.strip()
            if not self.prompt or len(self.prompt) > 1000:
                raise ValueError("prompt 须为 1–1000 字符，或以 user 结束的消息列表")
        else:
            if not 1 <= len(self.prompt) <= 7:
                raise ValueError("prompt 消息列表最多 7 条")
            check_roles(self.prompt, complete=False)
        if " ".join(self.chosen.split()) == " ".join(self.rejected.split()):
            raise ValueError("chosen 与 rejected 必须不同")
        return self


class DatasetConfig(StrictModel):
    dataset: list[Preference] = Field(default_factory=lambda: copy.deepcopy(DEFAULT_DATASET), min_length=5, max_length=64)
    system_prompt: str = Field(default=DEFAULT_SYSTEM, max_length=256)
    max_length: Literal[64, 128, 256] = 128
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def budget_and_groups(self):
        records = [row.model_dump() for row in self.dataset]
        prompts = [prompt_messages(row) for row in records]
        if sum(sum(len(m["content"]) for m in messages) + len(row["chosen"]) + len(row["rejected"]) for messages, row in zip(prompts, records)) > 20000:
            raise ValueError("偏好数据总文本不能超过 20,000 字符")
        keys = [tuple((m["role"], normalized(m["content"])) for m in messages) for messages in prompts]
        if len(set(keys)) != len(keys):
            raise ValueError("存在重复 prompt，请先合并或去重")
        if len({normalized(messages[0]["content"]) for messages in prompts}) < 5:
            raise ValueError("至少需要 5 个不同首问")
        return self


class PreviewRequest(DatasetConfig):
    selected_index: int = Field(default=0, strict=True, ge=0, le=63)


class RunConfig(DatasetConfig):
    rank: Literal[2, 4, 8, 16] = 8
    alpha: Literal[4, 8, 16, 32] = 16
    batch_size: Literal[1, 2, 4] = 2
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=0.01, allow_inf_nan=False)
    beta: float = Field(default=0.1, ge=0.01, le=1, allow_inf_nan=False)
    source_sft_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    source_checkpoint: Literal["latest", "best"] = "latest"


def prepare_dataset(tokenizer, config):
    encoded, groups, errors = [], {}, []
    for index, row in enumerate(config["dataset"]):
        messages = prompt_messages(row)
        pair = []
        for variant in ("chosen", "rejected"):
            # Score only the final completion. Earlier assistant turns are context.
            example = tokenize_example(tokenizer, messages + [{"role": "assistant", "content": row[variant]}], config["system_prompt"])
            final_mask = example["assistant_mask"]
            end = max(i for i, value in enumerate(final_mask) if value)
            start = end
            while start > 0 and final_mask[start - 1]:
                start -= 1
            example["labels"] = [token if start <= i <= end else -100 for i, token in enumerate(example["input_ids"])]
            if len(example["input_ids"]) > config["max_length"]:
                errors.append(f"第 {index + 1} 条 {variant} 长度 {len(example['input_ids'])} 超过 {config['max_length']}；不截断偏好回答")
            pair.append(example)
        encoded.append(pair)
        groups.setdefault(normalized(messages[0]["content"]), []).append(index)
    keys = list(groups)
    random.Random(config["seed"]).shuffle(keys)
    cut = int(len(keys) * 0.8)
    fingerprint = hashlib.sha256(json.dumps({key: config[key] for key in ("dataset", "system_prompt", "max_length", "seed")},
        sort_keys=True, ensure_ascii=False).encode() + TEMPLATE_VERSION.encode()).hexdigest()
    return {"encoded": encoded, "train_indices": [i for key in keys[:cut] for i in groups[key]],
        "validation_indices": [i for key in keys[cut:] for i in groups[key]], "errors": errors, "dataset_fingerprint": fingerprint}


def sequence_logps(logits, labels):
    """One causal shift; sum completion+EOS log probabilities, never length-average."""
    torch = torch_runtime()
    targets = labels[:, 1:]
    mask = targets != -100
    selected = logits[:, :-1].log_softmax(-1).gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~mask, 0).sum(-1)


def pair_logps(model, pairs, pad_id):
    inputs, labels, attention = make_batch([item for pair in pairs for item in pair], pad_id)
    return sequence_logps(model(input_ids=inputs, attention_mask=attention, use_cache=False).logits, labels).reshape(-1, 2)


def preference_loss(policy, reference, beta):
    torch = torch_runtime()
    rewards = beta * (policy - reference)
    margins = rewards[:, 0] - rewards[:, 1]
    return -torch.nn.functional.logsigmoid(margins), margins


def evaluate(model, prepared, reference, config, tokenizer):
    torch = torch_runtime()
    model.eval()
    with torch.inference_mode():
        policy = torch.cat([pair_logps(model, prepared["encoded"][i:i + config["batch_size"]], tokenizer.eos_token_id)
            for i in range(0, len(prepared["encoded"]), config["batch_size"])])
        losses, margins = preference_loss(policy, reference, config["beta"])
    metrics = {}
    for label, split in (("train", "train_indices"), ("validation", "validation_indices")):
        indices = prepared[split]
        metrics.update({f"{label}_loss": losses[indices].mean().item(), f"{label}_margin": margins[indices].mean().item(),
            f"{label}_accuracy": (margins[indices] > 0).float().mean().item()})
    if not all(math.isfinite(value) for value in metrics.values()):
        raise HTTPException(422, "DPO 指标非有限，本批未保存")
    scores = [{"chosen_logp": p[0], "rejected_logp": p[1], "reference_chosen_logp": r[0], "reference_rejected_logp": r[1], "margin": m}
        for p, r, m in zip(policy.tolist(), reference.tolist(), margins.tolist())]
    return metrics, scores


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def report(saved):
    keys = ("id", "created_at", "config", "model_id", "revision", "weight_sha256", "template_version", "step", "best_step", "history",
        "base_parameters", "trainable_parameters", "trainable_ratio", "frozen_verified", "reference_verified", "reference_digest",
        "dataset_fingerprint", "train_indices", "validation_indices", "source", "pair_scores")
    return {key: copy.deepcopy(saved[key]) for key in keys}


class DPOService(SFTService):
    def __init__(self, root=None, base_assets=None, loader=None, source_service=None):
        super().__init__(root if root is not None else ROOT / "runs" / "dpo", base_assets, loader)
        self.source_service = source_service or sft_service

    def _load(self, run_id):
        saved = TransformerService._load(self, run_id)
        if (saved.get("kind"), saved.get("model_id"), saved.get("revision"), saved.get("template_version")) != (
            "dpo-lora", MODEL_ID, MODEL_REVISION, TEMPLATE_VERSION):
            raise HTTPException(409, "DPO 检查点基座或模板不匹配")
        return saved

    def get(self, run_id):
        with self.lock:
            return report(self._load(run_id))

    def list(self):
        with self.lock:
            paths = sorted(self.root.glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if self.root.exists() else []
            return [report(self._load(p.parent.name)) for p in paths[:30]]

    def preview(self, config, selected_index=0):
        with self.lock:
            _, tokenizer = self._runtime()
            prepared = prepare_dataset(tokenizer, config)
        if selected_index >= len(prepared["encoded"]):
            raise HTTPException(422, "预览索引超出数据集范围")
        def view(item):
            return {"total_tokens": len(item["input_ids"]), "supervised_tokens": sum(label != -100 for label in item["labels"][1:]),
                "tokens": [{"id": token, "text": tokenizer.decode([token], skip_special_tokens=False), "label": label, "supervised": label != -100}
                    for token, label in zip(item["input_ids"][:256], item["labels"][:256])]}
        return {"valid": not prepared["errors"], "errors": prepared["errors"], "dataset_fingerprint": prepared["dataset_fingerprint"],
            "train_indices": prepared["train_indices"], "validation_indices": prepared["validation_indices"], "selected_index": selected_index,
            "chosen": view(prepared["encoded"][selected_index][0]), "rejected": view(prepared["encoded"][selected_index][1])}

    def create(self, config):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "DPO 正在计算，请稍后重试")
        try:
            torch = torch_runtime()
            config = copy.deepcopy(config)
            initial, source = None, {"kind": "pretrained", "step": 0}
            if config["source_sft_id"]:
                with self.source_service.lock:
                    original = self.source_service._load(config["source_sft_id"])
                if original["step"] == 0:
                    raise HTTPException(422, "请选择已训练的 SFT 实验")
                best = config["source_checkpoint"] == "best"
                if best and original["best_step"] == 0:
                    raise HTTPException(422, "所选最佳 SFT 权重尚未训练，请选择当前权重")
                initial = original["best_adapter"] if best else original["adapter"]
                config.update(rank=original["config"]["rank"], alpha=original["config"]["alpha"])
                source = {"kind": "sft", "run_id": original["id"], "step": original["best_step"] if best else original["step"], "checkpoint": config["source_checkpoint"]}
            model, tokenizer = self._model(config, initial)
            prepared = prepare_dataset(tokenizer, config)
            if prepared["errors"]:
                raise HTTPException(422, "；".join(prepared["errors"]))
            state = adapter_state(model)
            model.eval()
            with torch.inference_mode():
                reference = torch.cat([pair_logps(model, prepared["encoded"][i:i + config["batch_size"]], tokenizer.eos_token_id)
                    for i in range(0, len(prepared["encoded"]), config["batch_size"])])
            metrics, scores = evaluate(model, prepared, reference, config, tokenizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            if any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
                raise RuntimeError("DPO 基座未冻结")
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
            base_count = sum(p.numel() for name, p in model.named_parameters() if "lora_" not in name)
            count = sum(p.numel() for p in parameters)
            saved = {"version": 1, "kind": "dpo-lora", "id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "weight_sha256": "test-fixture" if self.loader else json.loads((self.assets.require() / "zhiwei-base.json").read_text())["weight_sha256"],
                "template_version": TEMPLATE_VERSION, "step": 0, "best_step": 0, "best_loss": metrics["validation_loss"],
                "adapter": state, "best_adapter": copy.deepcopy(state), "reference_adapter": copy.deepcopy(state), "reference_logps": reference.clone(),
                "reference_digest": state_digest(state), "source": source, "optimizer": optimizer.state_dict(),
                "generator_state": torch.Generator().manual_seed(config["seed"]).get_state(), "history": [{"step": 0, **metrics}],
                "base_parameters": base_count, "trainable_parameters": count, "trainable_ratio": count / (base_count + count),
                "frozen_digest": frozen_digest(model), "frozen_verified": True, "reference_verified": True,
                "dataset_fingerprint": prepared["dataset_fingerprint"], "train_indices": prepared["train_indices"],
                "validation_indices": prepared["validation_indices"], "pair_scores": scores}
            self._save(saved)
            return report(saved)
        finally:
            self.lock.release()

    def train(self, run_id, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "DPO 正在计算，请稍后刷新")
        try:
            saved = self._load(run_id)
            if saved["step"] != request.expected_step:
                raise HTTPException(409, "DPO 步数已改变，请先刷新记录")
            if saved["step"] + request.steps > 500:
                raise HTTPException(422, "DPO 实验最多 500 步")
            torch = torch_runtime()
            config = saved["config"]
            model, tokenizer = self._model(config, saved["adapter"])
            if frozen_digest(model) != saved["frozen_digest"] or state_digest(saved["reference_adapter"]) != saved["reference_digest"]:
                raise HTTPException(409, "基座或固定参考快照不匹配")
            prepared = prepare_dataset(tokenizer, config)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
            optimizer.load_state_dict(saved["optimizer"])
            generator = torch.Generator(); generator.set_state(saved["generator_state"])
            model.train()
            for _ in range(request.steps):
                indices = [saved["train_indices"][i] for i in torch.randint(len(saved["train_indices"]), (config["batch_size"],), generator=generator).tolist()]
                optimizer.zero_grad(set_to_none=True)
                policy = pair_logps(model, [prepared["encoded"][i] for i in indices], tokenizer.eos_token_id)
                losses, _ = preference_loss(policy, saved["reference_logps"][indices], config["beta"])
                loss = losses.mean()
                if not torch.isfinite(loss).item():
                    raise HTTPException(422, "DPO 损失非有限，本批未保存")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
            if frozen_digest(model) != saved["frozen_digest"]:
                raise RuntimeError("基座改变，拒绝保存")
            metrics, scores = evaluate(model, prepared, saved["reference_logps"], config, tokenizer)
            saved["step"] += request.steps
            state = adapter_state(model)
            if metrics["validation_loss"] < saved["best_loss"]:
                saved.update(best_loss=metrics["validation_loss"], best_step=saved["step"], best_adapter=copy.deepcopy(state))
            saved["history"].append({"step": saved["step"], **metrics})
            saved.update(adapter=state, optimizer=optimizer.state_dict(), generator_state=generator.get_state(), pair_scores=scores)
            self._save(saved)
            return report(saved)
        finally:
            self.lock.release()

    def compare(self, run_id, request):
        with self.lock:
            saved = self._load(run_id)
            before, tokenizer = self._model(saved["config"], saved["reference_adapter"])
            state = saved["best_adapter"] if request.checkpoint == "best" else saved["adapter"]
            after, _ = self._model(saved["config"], state)
        for p in before.parameters():
            p.requires_grad_(False)
        return {"run_id": run_id, "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "sampling": request.model_dump(exclude={"messages", "checkpoint"}), "step": saved["best_step"] if request.checkpoint == "best" else saved["step"],
            "checkpoint": request.checkpoint, "source": saved["source"], "template_version": TEMPLATE_VERSION,
            "before": generate_chat(before, tokenizer, saved["config"], request),
            "after": generate_chat(after, tokenizer, saved["config"], request)}

    def export(self, run_id, variant):
        original = super().export(run_id, variant)
        with self.lock:
            saved = self._load(run_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
            for name in source.namelist():
                if name != "sft_format.json":
                    package.writestr(name, source.read(name))
            package.writestr("dpo_format.json", json.dumps({"template_version": TEMPLATE_VERSION, "system_prompt": saved["config"]["system_prompt"],
                "max_length": saved["config"]["max_length"], "beta": saved["config"]["beta"], "objective": "sigmoid DPO; summed final-completion+EOS logps",
                "source": saved["source"], "reference_digest": saved["reference_digest"], "dataset_fingerprint": saved["dataset_fingerprint"]}, indent=2))
        return buffer.getvalue()


service = DPOService()
router = APIRouter(prefix="/api/dpo", tags=["偏好优化 DPO"])
RunId = Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")]


@router.get("/status")
def status():
    return {**assets.status(), "dataset": DEFAULT_DATASET, "system_prompt": DEFAULT_SYSTEM, "template_version": TEMPLATE_VERSION, "max_steps": 500}


@router.post("/prepare")
def prepare():
    return assets.prepare()


@router.post("/preview")
def preview(request: PreviewRequest):
    return service.preview(request.model_dump(exclude={"selected_index"}), request.selected_index)


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


@router.post("/runs/{run_id}/compare")
def compare(run_id: RunId, request: ChatRequest):
    return service.compare(run_id, request)


@router.get("/runs/{run_id}/adapter")
def download_adapter(run_id: RunId, checkpoint: Annotated[Literal["latest", "best"], Query()] = "latest"):
    return Response(service.export(run_id, checkpoint), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="dpo-{run_id[:8]}-{checkpoint}.zip"'})


@router.get("/runs/{run_id}/checkpoint")
def download_checkpoint(run_id: RunId):
    with service.lock:
        service._load(run_id); path = service._path(run_id)
    return FileResponse(path, media_type="application/octet-stream", filename=f"dpo-{run_id[:8]}-training.pt")
