"""Answer-only supervised LoRA fine-tuning and versioned multi-turn chat."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import random
import re
from typing import Annotated, Literal
from uuid import uuid4
import zipfile

from fastapi import APIRouter, HTTPException, Path as ApiPath, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.lora import LoRAService, TrainingRequest, adapter_state, frozen_digest
from backend.lora_assets import assets, MODEL_ID, MODEL_REVISION
from backend.mnist_engine import ROOT, torch_runtime
from backend.transformer import TransformerService

TEMPLATE_VERSION = "zhiwei-sft-v1"
CHAT_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'assistant' %}{{ 'Assistant:' }}"
    "{% generation %}{{ ' ' + m['content'] + eos_token }}{% endgeneration %}{{ '\n' }}"
    "{% elif m['role'] == 'system' %}{{ 'System: ' + m['content'] + '\n' }}"
    "{% else %}{{ 'User: ' + m['content'] + '\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)
DEFAULT_SYSTEM = "You are Zhiwei. Answer briefly and clearly."
DEFAULT_DATASET = [
    {"question": "What is Zhiwei?", "answer": "Zhiwei is a learning lab."},
    {"question": "What do we learn?", "answer": "We learn how models train."},
    {"question": "How should I start?", "answer": "Start with a small experiment."},
    {"question": "What is LoRA?", "answer": "LoRA learns small adapter matrices."},
    {"question": "Are base weights updated?", "answer": "No. The base weights stay frozen."},
    {"question": "What is SFT?", "answer": "SFT learns from example questions and answers."},
    {"question": "Which tokens receive loss?", "answer": "Only assistant answers and EOS receive loss."},
    {"question": "How do I check progress?", "answer": "Compare training and validation loss."},
    {"question": "Where are adapters saved?", "answer": "Adapters are saved locally."},
    {"question": "Why use validation data?", "answer": "Validation checks examples not used for updates."},
    {"question": "Can small models make mistakes?", "answer": "Yes. Check important answers."},
    {"question": "How do I reproduce a run?", "answer": "Keep the data, settings, and random seed."},
    {"question": "How do I save my work?", "answer": "Download the adapter and training checkpoint."},
    {"question": "What is the next step?", "answer": "Try one change and inspect the result."},
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("消息内容不能为空白")
        return value.strip()


class QAExample(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=1000)

    @field_validator("question", "answer")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("问题和回答不能为空白")
        return value.strip()


def check_roles(messages, complete):
    for index, message in enumerate(messages):
        role = message.role if isinstance(message, Message) else message["role"]
        if role != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("消息必须从 user 开始，user/assistant 交替")
    if len(messages) % 2 != (0 if complete else 1):
        raise ValueError("训练样本必须以 assistant 结束" if complete else "对话请求必须以 user 结束")


class ConversationExample(StrictModel):
    messages: list[Message] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def valid_roles(self):
        check_roles(self.messages, complete=True)
        return self


def normalize_records(dataset):
    records = []
    for item in dataset:
        item = item.model_dump() if isinstance(item, BaseModel) else item
        records.append(item["messages"] if "messages" in item else [
            {"role": "user", "content": item["question"]}, {"role": "assistant", "content": item["answer"]}])
    return records


class DatasetConfig(StrictModel):
    dataset: list[QAExample | ConversationExample] = Field(default_factory=lambda: copy.deepcopy(DEFAULT_DATASET), min_length=5, max_length=64)
    system_prompt: str = Field(default=DEFAULT_SYSTEM, max_length=256)
    max_length: Literal[64, 128, 256] = 128
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def unique_prompts_and_budget(self):
        records = normalize_records(self.dataset)
        if sum(len(message["content"]) for messages in records for message in messages) > 20000:
            raise ValueError("数据集消息总字符数不能超过 20,000")
        keys = [tuple(" ".join(m["content"].split()).casefold() for m in messages if m["role"] == "user") for messages in records]
        if len(set(keys)) != len(keys):
            raise ValueError("存在重复问题或重复用户消息序列，请去重以避免划分泄漏")
        if len({key[0] for key in keys}) < 5:
            raise ValueError("至少需要 5 个不同首问，首问相同的对话会分在同一集合")
        return self


class PreviewRequest(DatasetConfig):
    selected_index: int = Field(default=0, strict=True, ge=0, le=63)


class RunConfig(DatasetConfig):
    rank: Literal[2, 4, 8, 16] = 8
    alpha: Literal[4, 8, 16, 32] = 16
    batch_size: Literal[1, 2, 4] = 2
    learning_rate: float = Field(default=0.003, ge=0.0001, le=0.01, allow_inf_nan=False)


class ChatRequest(StrictModel):
    messages: list[Message] = Field(min_length=1, max_length=9)
    length: int = Field(default=32, strict=True, ge=1, le=64)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    top_k: Literal[0, 10, 50] = 10
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)
    checkpoint: Literal["latest", "best"] = "latest"

    @model_validator(mode="after")
    def valid_roles(self):
        check_roles(self.messages, complete=False)
        return self


def template_messages(messages, system_prompt):
    return ([{"role": "system", "content": system_prompt}] if system_prompt.strip() else []) + messages


def tokenize_example(tokenizer, messages, system_prompt):
    encoded = tokenizer.apply_chat_template(template_messages(messages, system_prompt), chat_template=CHAT_TEMPLATE,
        tokenize=True, return_dict=True, return_assistant_tokens_mask=True, add_generation_prompt=False)
    ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
    if len(ids) != len(mask) or not any(mask[1:]):
        raise HTTPException(422, "分词器未生成有效回答掩码，请检查数据与依赖版本")
    return {"input_ids": ids, "labels": [token if supervised else -100 for token, supervised in zip(ids, mask)], "assistant_mask": mask}


def prepare_dataset(tokenizer, config):
    records = normalize_records(config["dataset"])
    encoded = [tokenize_example(tokenizer, messages, config["system_prompt"]) for messages in records]
    groups = {}
    for index, messages in enumerate(records):
        key = " ".join(messages[0]["content"].split()).casefold()
        groups.setdefault(key, []).append(index)
    group_keys = list(groups)
    random.Random(config["seed"]).shuffle(group_keys)
    cut = int(len(group_keys) * 0.8)
    train_indices = [index for key in group_keys[:cut] for index in groups[key]]
    validation_indices = [index for key in group_keys[cut:] for index in groups[key]]
    errors = [f"第 {i + 1} 条样本含 {len(item['input_ids'])} 个 token，超过最大长度 {config['max_length']}；请增大长度或缩短内容。"
              for i, item in enumerate(encoded) if len(item["input_ids"]) > config["max_length"]]
    fingerprint = hashlib.sha256(json.dumps({"records": records, "system_prompt": config["system_prompt"],
        "max_length": config["max_length"], "seed": config["seed"], "template": TEMPLATE_VERSION}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"records": records, "encoded": encoded, "train_indices": train_indices, "validation_indices": validation_indices,
            "dataset_fingerprint": fingerprint, "errors": errors}


def make_batch(examples, pad_id):
    torch = torch_runtime()
    width = max(len(item["input_ids"]) for item in examples)
    inputs = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    attention = torch.zeros_like(inputs)
    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        inputs[row, :length] = torch.tensor(example["input_ids"])
        labels[row, :length] = torch.tensor(example["labels"])
        attention[row, :length] = 1
    return inputs, labels, attention


def answer_loss(logits, labels, reduction="mean"):
    torch = torch_runtime()
    return torch.nn.functional.cross_entropy(logits[:, :-1].contiguous().reshape(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().reshape(-1), ignore_index=-100, reduction=reduction)


def evaluate(model, prepared, tokenizer, batch_size):
    torch = torch_runtime()
    model.eval()
    losses = []
    counts = []
    with torch.inference_mode():
        for split in ("train_indices", "validation_indices"):
            total, count = 0.0, 0
            indices = prepared[split]
            for start in range(0, len(indices), batch_size):
                examples = [prepared["encoded"][index] for index in indices[start:start + batch_size]]
                inputs, labels, attention = make_batch(examples, tokenizer.eos_token_id)
                logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
                total += answer_loss(logits, labels, reduction="sum").item()
                count += int((labels[:, 1:] != -100).sum().item())
            losses.append(total / count)
            counts.append(count)
    if not all(math.isfinite(value) for value in losses):
        raise HTTPException(422, "回答损失非有限，本批未保存")
    return {"train_loss": losses[0], "validation_loss": losses[1], "perplexity": math.exp(min(losses[1], 700))}, counts


def report(saved):
    keys = ("id", "created_at", "config", "model_id", "revision", "weight_sha256", "template_version", "step", "best_step", "history",
        "base_parameters", "trainable_parameters", "trainable_ratio", "frozen_verified", "dataset_fingerprint", "train_indices", "validation_indices",
        "train_supervised_tokens", "validation_supervised_tokens")
    return {key: copy.deepcopy(saved[key]) for key in keys}


def generate_chat(model, tokenizer, config, request):
    torch = torch_runtime()
    model.eval()
    messages = [message.model_dump() for message in request.messages]
    limit = config["max_length"]

    def encode_context():
        return tokenizer.apply_chat_template(template_messages(messages, config["system_prompt"]),
            chat_template=CHAT_TEMPLATE, tokenize=True, add_generation_prompt=True)

    ids = encode_context()
    dropped = 0
    budget = limit - min(request.length, limit // 2)
    while len(messages) > 1 and len(ids) > budget:
        messages = messages[2:]
        dropped += 1
        ids = encode_context()
    if len(ids) >= limit:
        raise HTTPException(422, "系统提示与最新问题超过上下文，请缩短问题或使用更大最大长度的实验")
    maximum = min(request.length, limit - len(ids))
    generator = torch.Generator().manual_seed(request.seed)
    sequence, output = list(ids), []
    reason = "context_limit" if maximum < request.length else "length"
    reply = ""
    with torch.inference_mode():
        for _ in range(maximum):
            scores = model(input_ids=torch.tensor([sequence]), use_cache=False).logits[0, -1]
            if request.temperature == 0:
                token = int(scores.argmax().item())
            else:
                scores = scores / request.temperature
                if request.top_k:
                    keep = torch.topk(scores, min(request.top_k, len(scores))).indices
                    mask = torch.ones_like(scores, dtype=torch.bool)
                    mask[keep] = False
                    scores = scores.masked_fill(mask, float("-inf"))
                token = int(torch.multinomial(scores.softmax(-1), 1, generator=generator).item())
            output.append(token)
            sequence.append(token)
            if token == tokenizer.eos_token_id:
                reason = "eos"
                break
            reply = tokenizer.decode(output, skip_special_tokens=True)
            boundary = re.search(r"(?:^|\n)(?:User|Assistant|System):", reply)
            if boundary:
                reply = reply[:boundary.start()]
                reason = "role_boundary"
                break
        if reason != "role_boundary":
            reply = tokenizer.decode(output, skip_special_tokens=True)
    return {"reply": reply.strip(), "token_ids": output, "finish_reason": reason, "dropped_turns": dropped,
        "context_tokens": len(ids), "used_messages": messages}


class SFTService(LoRAService):
    def __init__(self, root=None, base_assets=None, loader=None):
        super().__init__(root if root is not None else ROOT / "runs" / "sft", base_assets, loader)

    def _load(self, run_id):
        saved = TransformerService._load(self, run_id)
        if (saved.get("kind"), saved.get("model_id"), saved.get("revision"), saved.get("template_version")) != (
            "sft-lora", MODEL_ID, MODEL_REVISION, TEMPLATE_VERSION):
            raise HTTPException(409, "SFT 检查点的基座或模板版本不匹配")
        return saved

    def get(self, run_id):
        with self.lock:
            return report(self._load(run_id))

    def list(self):
        with self.lock:
            paths = sorted(self.root.glob("*/checkpoint.pt"), key=lambda path: path.stat().st_mtime, reverse=True) if self.root.exists() else []
            return [report(self._load(path.parent.name)) for path in paths[:30]]

    def preview(self, config, selected_index=0):
        with self.lock:
            _, tokenizer = self._runtime()
            prepared = prepare_dataset(tokenizer, config)
        if selected_index >= len(prepared["encoded"]):
            raise HTTPException(422, "预览样本索引超出数据集范围")
        selected = prepared["encoded"][selected_index]
        tokens = [{"id": token, "text": tokenizer.decode([token], skip_special_tokens=False), "label": label, "supervised": label != -100}
                  for token, label in zip(selected["input_ids"][:256], selected["labels"][:256])]
        return {"valid": not prepared["errors"], "errors": prepared["errors"], "dataset_fingerprint": prepared["dataset_fingerprint"],
            "train_indices": prepared["train_indices"], "validation_indices": prepared["validation_indices"], "template_version": TEMPLATE_VERSION,
            "records": [{"index": index, "question": [m["content"] for m in messages if m["role"] == "user"][-1],
                "tokens": len(item["input_ids"]), "supervised_tokens": sum(label != -100 for label in item["labels"][1:])}
                for index, (messages, item) in enumerate(zip(prepared["records"], prepared["encoded"]))],
            "selected_index": selected_index, "tokens": tokens, "total_tokens": len(selected["input_ids"]),
            "rendered_text": tokenizer.apply_chat_template(template_messages(prepared["records"][selected_index], config["system_prompt"]),
                chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=False)}

    def create(self, config):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "SFT 正在计算，请稍后重试")
        try:
            torch = torch_runtime()
            model, tokenizer = self._model(config)
            prepared = prepare_dataset(tokenizer, config)
            if prepared["errors"]:
                raise HTTPException(422, "；".join(prepared["errors"]))
            parameters = [p for p in model.parameters() if p.requires_grad]
            if any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
                raise RuntimeError("基座参数未完全冻结")
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
            metrics, counts = evaluate(model, prepared, tokenizer, config["batch_size"])
            state = adapter_state(model)
            base_count = sum(p.numel() for name, p in model.named_parameters() if "lora_" not in name)
            trainable_count = sum(p.numel() for p in parameters)
            weight_sha = "test-fixture" if self.loader else json.loads((self.assets.require() / "zhiwei-base.json").read_text(encoding="utf-8"))["weight_sha256"]
            saved = {"version": 1, "kind": "sft-lora", "id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": weight_sha, "template_version": TEMPLATE_VERSION,
                "step": 0, "best_step": 0, "best_loss": metrics["validation_loss"], "history": [{"step": 0, **metrics}],
                "adapter": state, "best_adapter": copy.deepcopy(state), "optimizer": optimizer.state_dict(),
                "generator_state": torch.Generator().manual_seed(config["seed"]).get_state(), "base_parameters": base_count,
                "trainable_parameters": trainable_count, "trainable_ratio": trainable_count / (base_count + trainable_count),
                "frozen_digest": frozen_digest(model), "frozen_verified": True, "dataset_fingerprint": prepared["dataset_fingerprint"],
                "train_indices": prepared["train_indices"], "validation_indices": prepared["validation_indices"],
                "train_supervised_tokens": counts[0], "validation_supervised_tokens": counts[1]}
            self._save(saved)
            return report(saved)
        finally:
            self.lock.release()

    def train(self, run_id, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "SFT 正在计算，请稍后刷新记录")
        try:
            saved = self._load(run_id)
            if saved["step"] != request.expected_step:
                raise HTTPException(409, "SFT 步数已改变，请刷新后继续")
            if saved["step"] + request.steps > 500:
                raise HTTPException(422, "单次 SFT 实验最多 500 步")
            torch = torch_runtime()
            config = saved["config"]
            model, tokenizer = self._model(config, saved["adapter"])
            before = frozen_digest(model)
            if before != saved["frozen_digest"]:
                raise HTTPException(409, "基座与实验创建时不一致")
            prepared = prepare_dataset(tokenizer, config)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
            optimizer.load_state_dict(saved["optimizer"])
            generator = torch.Generator()
            generator.set_state(saved["generator_state"])
            train_examples = [prepared["encoded"][index] for index in saved["train_indices"]]
            model.train()
            for _ in range(request.steps):
                indices = torch.randint(len(train_examples), (config["batch_size"],), generator=generator).tolist()
                inputs, labels, attention = make_batch([train_examples[index] for index in indices], tokenizer.eos_token_id)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
                loss = answer_loss(logits, labels)
                if not torch.isfinite(loss).item():
                    raise HTTPException(422, "SFT 损失非有限，本批未保存")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
            if frozen_digest(model) != before:
                raise RuntimeError("基座参数改变，本批拒绝保存")
            saved["step"] += request.steps
            metrics, _ = evaluate(model, prepared, tokenizer, config["batch_size"])
            state = adapter_state(model)
            if metrics["validation_loss"] < saved["best_loss"]:
                saved.update(best_loss=metrics["validation_loss"], best_step=saved["step"], best_adapter=copy.deepcopy(state))
            saved["history"].append({"step": saved["step"], **metrics})
            saved.update(adapter=state, optimizer=optimizer.state_dict(), generator_state=generator.get_state(), frozen_verified=True)
            self._save(saved)
            return report(saved)
        finally:
            self.lock.release()

    def chat(self, run_id, request):
        with self.lock:
            saved = self._load(run_id)
            state = saved["best_adapter"] if request.checkpoint == "best" else saved["adapter"]
            model, tokenizer = self._model(saved["config"], state)
        return {"run_id": run_id, "model_id": MODEL_ID, "revision": MODEL_REVISION, "template_version": TEMPLATE_VERSION,
            "step": saved["best_step"] if request.checkpoint == "best" else saved["step"], "checkpoint": request.checkpoint,
            **generate_chat(model, tokenizer, saved["config"], request)}

    def export(self, run_id, variant):
        original = super().export(run_id, variant)
        with self.lock:
            saved = self._load(run_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for name in source.namelist():
                package.writestr(name, source.read(name))
            package.writestr("chat_template.jinja", CHAT_TEMPLATE)
            package.writestr("sft_format.json", json.dumps({"template_version": TEMPLATE_VERSION, "system_prompt": saved["config"]["system_prompt"],
                "max_length": saved["config"]["max_length"], "loss": "assistant answers and EOS only", "dataset_fingerprint": saved["dataset_fingerprint"]}, indent=2))
        return buffer.getvalue()


service = SFTService()
router = APIRouter(prefix="/api/sft", tags=["指令微调 SFT 与对话"])
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


@router.post("/runs/{run_id}/chat")
def chat(run_id: RunId, request: ChatRequest):
    return service.chat(run_id, request)


@router.get("/runs/{run_id}/adapter")
def download_adapter(run_id: RunId, checkpoint: Annotated[Literal["latest", "best"], Query()] = "latest"):
    return Response(service.export(run_id, checkpoint), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="sft-{run_id[:8]}-{checkpoint}.zip"'})


@router.get("/runs/{run_id}/checkpoint")
def download_checkpoint(run_id: RunId):
    with service.lock:
        service._load(run_id)
        path = service._path(run_id)
    return FileResponse(path, media_type="application/octet-stream", filename=f"sft-{run_id[:8]}-training.pt")
