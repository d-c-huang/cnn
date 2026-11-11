"""Small CPU GRPO: on-policy groups, clipped token ratios, fixed-reference KL."""
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

from backend.dpo import DPOService, normalized, prompt_messages, state_digest, service as dpo_service
from backend.lora import LoRAService, adapter_state, frozen_digest
from backend.lora_assets import assets, MODEL_ID, MODEL_REVISION
from backend.mnist_engine import ROOT, torch_runtime
from backend.sft import (StrictModel, Message, ChatRequest, CHAT_TEMPLATE, TEMPLATE_VERSION, DEFAULT_SYSTEM,
    DEFAULT_DATASET as QA_DATASET, check_roles, template_messages, make_batch, service as sft_service)
from backend.transformer import TransformerService

RULE_VERSION = "zhiwei-rewards-v1"
TARGETS = ["lab", "models", "small", "adapter", "frozen", "answers", "EOS", "validation", "locally", "validation", "check", "seed", "download", "inspect"]
DEFAULT_DATASET = [{"prompt": row["question"], "target": target} for row, target in zip(QA_DATASET, TARGETS)]
DEFAULT_RULES = [{"type": "contains_target", "weight": 1.0}, {"type": "length", "weight": 0.25, "min_chars": 8, "max_chars": 100}, {"type": "nonempty", "weight": 0.1}]


class RewardRule(StrictModel):
    type: Literal["nonempty", "contains", "avoids", "contains_target", "exact_target", "length", "ends_with", "json_object"]
    weight: float = Field(default=1, ge=-5, le=5, allow_inf_nan=False)
    text: str | None = Field(default=None, max_length=200)
    case_sensitive: bool = False
    min_chars: int = Field(default=1, strict=True, ge=0, le=4000)
    max_chars: int = Field(default=100, strict=True, ge=1, le=4000)
    required_keys: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def valid_rule(self):
        if not self.weight:
            raise ValueError("奖励权重不能为 0")
        if self.type in ("contains", "avoids", "ends_with") and (not self.text or not self.text.strip()):
            raise ValueError("该规则需要非空 text")
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars 不能大于 max_chars")
        if any(not key.strip() or len(key) > 80 for key in self.required_keys) or len(set(self.required_keys)) != len(self.required_keys):
            raise ValueError("JSON 键须非空、不重复，最多 80 字符")
        allowed = {"type", "weight"}
        if self.type in ("contains", "avoids", "ends_with"):
            allowed |= {"text", "case_sensitive"}
        if self.type in ("contains_target", "exact_target"):
            allowed.add("case_sensitive")
        if self.type == "length":
            allowed |= {"min_chars", "max_chars"}
        if self.type == "json_object":
            allowed.add("required_keys")
        neutral = {"text": None, "case_sensitive": False, "min_chars": 1, "max_chars": 100, "required_keys": []}
        if any(getattr(self, key) != neutral[key] for key in self.model_fields_set - allowed):
            raise ValueError("此规则含不适用的字段，请检查 type 与参数")
        return self


def score_reward(text, target, rules):
    response = text.strip()
    components = []
    for index, rule in enumerate(rules):
        kind = rule["type"]
        compared = response if rule.get("case_sensitive", False) else response.casefold()
        expected = target.strip() if rule.get("case_sensitive", False) else target.strip().casefold()
        needle = rule.get("text") or ""
        if not rule.get("case_sensitive", False):
            needle = needle.casefold()
        passed = False
        if kind == "nonempty": passed = bool(response)
        elif kind == "contains": passed = needle in compared
        elif kind == "avoids": passed = needle not in compared
        elif kind == "contains_target": passed = bool(expected) and expected in compared
        elif kind == "exact_target": passed = bool(expected) and " ".join(compared.split()) == " ".join(expected.split())
        elif kind == "length": passed = rule.get("min_chars", 1) <= len(response) <= rule.get("max_chars", 100)
        elif kind == "ends_with": passed = compared.endswith(needle)
        elif kind == "json_object":
            try:
                value = json.loads(response, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                passed = isinstance(value, dict) and all(key in value for key in rule.get("required_keys", []))
            except (ValueError, RecursionError): pass
        value = float(passed)
        components.append({"index": index, "type": kind, "score": value, "weight": rule.get("weight", 1), "weighted": value * rule.get("weight", 1)})
    return {"reward": sum(item["weighted"] for item in components), "components": components}


class PromptRecord(StrictModel):
    prompt: str | list[Message]
    target: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def valid_prompt(self):
        self.target = self.target.strip()
        if isinstance(self.prompt, str):
            self.prompt = self.prompt.strip()
            if not self.prompt or len(self.prompt) > 1000:
                raise ValueError("prompt 须为 1–1000 字符或交替消息列表")
        else:
            if not 1 <= len(self.prompt) <= 7:
                raise ValueError("prompt 消息列表最多七条")
            check_roles(self.prompt, complete=False)
        return self


class DatasetConfig(StrictModel):
    dataset: list[PromptRecord] = Field(default_factory=lambda: copy.deepcopy(DEFAULT_DATASET), min_length=5, max_length=32)
    rules: list[RewardRule] = Field(default_factory=lambda: copy.deepcopy(DEFAULT_RULES), min_length=1, max_length=8)
    system_prompt: str = Field(default=DEFAULT_SYSTEM, max_length=256)
    max_length: Literal[64, 128, 256] = 128
    completion_length: Literal[8, 16, 32] = 16
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def budget(self):
        records = [item.model_dump() for item in self.dataset]
        prompts = [prompt_messages(row) for row in records]
        if sum(sum(len(m["content"]) for m in messages) + len(row["target"]) for messages, row in zip(prompts, records)) > 16000:
            raise ValueError("问题与 target 总文本不能超过 16,000 字符")
        keys = [tuple((m["role"], normalized(m["content"])) for m in messages) for messages in prompts]
        if len(set(keys)) != len(keys) or len({normalized(messages[0]["content"]) for messages in prompts}) < 5:
            raise ValueError("prompt 不得重复，且至少需要五个不同首问")
        if any(rule.type in ("contains_target", "exact_target") for rule in self.rules) and any(not row["target"] for row in records):
            raise ValueError("使用 target 规则时，每条记录须有非空 target")
        return self


def rule_dicts(rules):
    return [rule.model_dump() for rule in rules]


class RewardRequest(StrictModel):
    rules: list[RewardRule] = Field(min_length=1, max_length=8)
    text: str = Field(max_length=4000)
    target: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def target_required(self):
        if any(rule.type in ("contains_target", "exact_target") for rule in self.rules) and not self.target.strip():
            raise ValueError("此规则需要 target")
        return self


class RunConfig(DatasetConfig):
    rank: Literal[2, 4, 8, 16] = 8
    alpha: Literal[4, 8, 16, 32] = 16
    group_size: Literal[2, 4] = 4
    learning_rate: float = Field(default=0.00003, ge=0.000001, le=0.001, allow_inf_nan=False)
    beta: float = Field(default=0.02, ge=0, le=0.5, allow_inf_nan=False)
    clip_epsilon: float = Field(default=0.2, ge=0.05, le=0.4, allow_inf_nan=False)
    updates_per_rollout: Literal[1, 2] = 2
    source_type: Literal["pretrained", "sft", "dpo"] = "pretrained"
    source_run_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    source_checkpoint: Literal["latest", "best"] = "latest"

    @model_validator(mode="after")
    def source_id(self):
        if (self.source_type == "pretrained") != (self.source_run_id is None):
            raise ValueError("预训练来源不填 ID；SFT / DPO 来源必须填写 ID")
        return self


class ComparisonRequest(ChatRequest):
    reward_target: str = Field(default="", max_length=1000)


class TrainingRequest(StrictModel):
    expected_step: int = Field(strict=True, ge=0, le=100)
    steps: int = Field(default=1, strict=True, ge=1, le=2)


def prepare_dataset(tokenizer, config):
    prompts, groups, errors = [], {}, []
    for index, row in enumerate(config["dataset"]):
        messages = prompt_messages(row)
        ids = list(tokenizer.apply_chat_template(template_messages(messages, config["system_prompt"]), chat_template=CHAT_TEMPLATE, tokenize=True, add_generation_prompt=True))
        if len(ids) + config["completion_length"] > config["max_length"]:
            errors.append(f"第 {index + 1} 条 prompt {len(ids)} tokens + 回答预算 {config['completion_length']} 超过 {config['max_length']}；请缩短或增大上下文")
        prompts.append(ids)
        groups.setdefault(normalized(messages[0]["content"]), []).append(index)
    keys = list(groups); random.Random(config["seed"]).shuffle(keys)
    cut = int(len(keys) * .8)
    fingerprint = hashlib.sha256(json.dumps({key: config[key] for key in ("dataset", "rules", "system_prompt", "max_length", "completion_length", "seed")}, sort_keys=True, ensure_ascii=False).encode() + (TEMPLATE_VERSION + RULE_VERSION).encode()).hexdigest()
    return {"prompts": prompts, "errors": errors, "train_indices": [i for k in keys[:cut] for i in groups[k]],
        "validation_indices": [i for k in keys[cut:] for i in groups[k]], "dataset_fingerprint": fingerprint}


def sample_group(model, tokenizer, prompt_ids, size, length, generator, greedy=False):
    """Temperature=1 and full softmax: same distribution as scored old-policy logps."""
    torch = torch_runtime()
    model.eval()
    outputs = [[] for _ in range(size)]
    finished = torch.zeros(size, dtype=torch.bool)
    inputs = torch.tensor([prompt_ids] * size)
    attention = torch.ones_like(inputs)
    past = None
    with torch.inference_mode():
        for _ in range(length):
            result = model(input_ids=inputs, attention_mask=attention, past_key_values=past, use_cache=True)
            past = result.past_key_values
            scores = result.logits[:, -1]
            active_mask = (~finished).long()
            active = (~finished).nonzero(as_tuple=True)[0]
            next_tokens = torch.full((size,), tokenizer.eos_token_id, dtype=torch.long)
            selected = scores[active].argmax(-1) if greedy else torch.multinomial(scores[active].softmax(-1), 1, generator=generator).squeeze(-1)
            next_tokens[active] = selected
            for i in active.tolist():
                token = int(next_tokens[i].item()); outputs[i].append(token)
                if token == tokenizer.eos_token_id: finished[i] = True
            attention = torch.cat([attention, active_mask.unsqueeze(-1)], -1)
            inputs = next_tokens.unsqueeze(-1)
            if finished.all().item(): break
    return [{"text": tokenizer.decode(ids, skip_special_tokens=True).strip(), "token_ids": ids,
        "finish_reason": "eos" if ids[-1] == tokenizer.eos_token_id else "length"} for ids in outputs]


def rollout_batch(prompt_ids, samples, pad_id):
    examples = [{"input_ids": prompt_ids + sample["token_ids"], "labels": [-100] * len(prompt_ids) + sample["token_ids"]} for sample in samples]
    return make_batch(examples, pad_id)


def token_logps(model, inputs, labels, attention):
    logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits[:, :-1]
    targets = labels[:, 1:]; mask = targets != -100
    values = logits.log_softmax(-1).gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return values.masked_fill(~mask, 0), mask


def group_advantages(rewards):
    torch = torch_runtime()
    values = torch.tensor(rewards, dtype=torch.float32)
    std = values.std(correction=0)
    return ((values - values.mean()) / (std + 1e-4)).detach(), std.item()


def grpo_loss(new, old, reference, advantages, mask, beta, epsilon):
    torch = torch_runtime()
    ratio = (new - old.detach()).exp()
    surrogate = torch.minimum(ratio * advantages[:, None], ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None])
    delta = reference.detach() - new
    kl = delta.exp() - delta - 1
    counts = mask.sum(-1).clamp_min(1)
    loss = ((-surrogate + beta * kl).masked_fill(~mask, 0).sum(-1) / counts).mean()
    mean_kl = (kl.masked_fill(~mask, 0).sum(-1) / counts).mean()
    # Count actual surrogate clipping, respecting the advantage sign.
    clipped = ((ratio > 1 + epsilon) & (advantages[:, None] > 0)) | ((ratio < 1 - epsilon) & (advantages[:, None] < 0))
    clip_fraction = (clipped & mask).sum() / mask.sum().clamp_min(1)
    return loss, mean_kl, clip_fraction


def evaluate(model, tokenizer, prepared, config):
    rewards, rows = [], []
    for index in prepared["validation_indices"]:
        sample = sample_group(model, tokenizer, prepared["prompts"][index], 1, config["completion_length"], None, greedy=True)[0]
        scored = score_reward(sample["text"], config["dataset"][index]["target"], config["rules"])
        rewards.append(scored["reward"]); rows.append({"index": index, **sample, **scored})
    return sum(rewards) / len(rewards), rows


def report(saved):
    keys = ("id", "created_at", "config", "model_id", "revision", "template_version", "rule_version", "step", "best_step", "history",
        "base_parameters", "trainable_parameters", "frozen_verified", "reference_verified", "reference_digest", "adapter_changed",
        "dataset_fingerprint", "train_indices", "validation_indices", "source", "last_rollout", "validation_samples", "advantage_groups", "optimizer_updates")
    return {key: copy.deepcopy(saved[key]) for key in keys}


class GRPOService(DPOService):
    def __init__(self, root=None, base_assets=None, loader=None, source_service=None, preference_service=None):
        super().__init__(root if root is not None else ROOT / "runs" / "grpo", base_assets, loader, source_service)
        self.preference_service = preference_service or dpo_service

    def _load(self, run_id):
        saved = TransformerService._load(self, run_id)
        if (saved.get("kind"), saved.get("model_id"), saved.get("revision"), saved.get("template_version"), saved.get("rule_version")) != (
            "grpo-lora", MODEL_ID, MODEL_REVISION, TEMPLATE_VERSION, RULE_VERSION):
            raise HTTPException(409, "GRPO 基座、模板或奖励规则版本不匹配")
        return saved

    def get(self, run_id):
        with self.lock: return report(self._load(run_id))

    def list(self):
        with self.lock:
            paths = sorted(self.root.glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if self.root.exists() else []
            return [report(self._load(path.parent.name)) for path in paths[:30]]

    def preview(self, config):
        with self.lock:
            _, tokenizer = self._runtime(); prepared = prepare_dataset(tokenizer, config)
        return {"valid": not prepared["errors"], "errors": prepared["errors"], "dataset_fingerprint": prepared["dataset_fingerprint"],
            "train_indices": prepared["train_indices"], "validation_indices": prepared["validation_indices"],
            "prompt_tokens": [len(ids) for ids in prepared["prompts"]], "rule_version": RULE_VERSION,
            "reward_bound": sum(abs(rule["weight"]) for rule in config["rules"])}

    def create(self, config):
        if not self.lock.acquire(blocking=False): raise HTTPException(409, "GRPO 正在计算，请稍后重试")
        try:
            torch = torch_runtime(); config = copy.deepcopy(config)
            initial, source = None, {"kind": "pretrained", "step": 0}
            if config["source_type"] != "pretrained":
                provider = self.source_service if config["source_type"] == "sft" else self.preference_service
                with provider.lock: original = provider._load(config["source_run_id"])
                best = config["source_checkpoint"] == "best"
                source_step = original["best_step"] if best else original["step"]
                if source_step == 0: raise HTTPException(422, "所选来源权重尚未训练，请选择已训练权重")
                initial = original["best_adapter"] if best else original["adapter"]
                config.update(rank=original["config"]["rank"], alpha=original["config"]["alpha"])
                source = {"kind": config["source_type"], "run_id": original["id"], "step": source_step, "checkpoint": config["source_checkpoint"]}
            model, tokenizer = self._model(config, initial)
            prepared = prepare_dataset(tokenizer, config)
            if prepared["errors"]: raise HTTPException(422, "；".join(prepared["errors"]))
            state = adapter_state(model)
            reward, validation = evaluate(model, tokenizer, prepared, config)
            parameters = [p for p in model.parameters() if p.requires_grad]
            if any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad): raise RuntimeError("GRPO 基座未冻结")
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
            saved = {"version": 1, "kind": "grpo-lora", "id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "model_id": MODEL_ID, "revision": MODEL_REVISION, "template_version": TEMPLATE_VERSION, "rule_version": RULE_VERSION,
                "weight_sha256": "test-fixture" if self.loader else json.loads((self.assets.require() / "zhiwei-base.json").read_text())["weight_sha256"],
                "adapter": state, "best_adapter": copy.deepcopy(state), "reference_adapter": copy.deepcopy(state), "reference_digest": state_digest(state),
                "source": source, "step": 0, "best_step": 0, "best_reward": reward, "frozen_digest": frozen_digest(model),
                "base_parameters": sum(p.numel() for name, p in model.named_parameters() if "lora_" not in name), "trainable_parameters": sum(p.numel() for p in parameters),
                "frozen_verified": True, "reference_verified": True, "adapter_changed": False, "advantage_groups": 0, "optimizer_updates": 0,
                "generator_state": torch.Generator().manual_seed(config["seed"]).get_state(), "optimizer": optimizer.state_dict(),
                "dataset_fingerprint": prepared["dataset_fingerprint"], "train_indices": prepared["train_indices"], "validation_indices": prepared["validation_indices"],
                "last_rollout": None, "validation_samples": validation, "history": [{"step": 0, "rollout_reward": None, "reward_std": None,
                    "loss": None, "kl": None, "clip_fraction": None, "validation_reward": reward, "zero_variance": None}]}
            self._save(saved); return report(saved)
        finally: self.lock.release()

    def train(self, run_id, request):
        if not self.lock.acquire(blocking=False): raise HTTPException(409, "GRPO 正在计算，请稍后刷新")
        try:
            saved = self._load(run_id)
            if saved["step"] != request.expected_step: raise HTTPException(409, "GRPO 步数已改变，请先刷新记录")
            if saved["step"] + request.steps > 100: raise HTTPException(422, "单实验最多 100 轮 GRPO")
            torch = torch_runtime(); config = saved["config"]
            model, tokenizer = self._model(config, saved["adapter"])
            reference, _ = self._model(config, saved["reference_adapter"])
            for p in reference.parameters(): p.requires_grad_(False)
            reference.eval()
            if frozen_digest(model) != saved["frozen_digest"] or state_digest(adapter_state(reference)) != saved["reference_digest"]:
                raise HTTPException(409, "基座或参考快照改变，拒绝训练")
            prepared = prepare_dataset(tokenizer, config)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0); optimizer.load_state_dict(saved["optimizer"])
            generator = torch.Generator(); generator.set_state(saved["generator_state"])
            for _ in range(request.steps):
                index = saved["train_indices"][int(torch.randint(len(saved["train_indices"]), (1,), generator=generator).item())]
                ids = prepared["prompts"][index]
                old_digest = state_digest(adapter_state(model))
                samples = sample_group(model, tokenizer, ids, config["group_size"], config["completion_length"], generator)
                scored = [score_reward(sample["text"], config["dataset"][index]["target"], config["rules"]) for sample in samples]
                advantages, std = group_advantages([row["reward"] for row in scored])
                inputs, labels, attention = rollout_batch(ids, samples, tokenizer.eos_token_id)
                model.eval()  # Dropout is disabled, but autograd remains active for policy updates.
                with torch.no_grad():
                    old, mask = token_logps(model, inputs, labels, attention)
                    ref, _ = token_logps(reference, inputs, labels, attention)
                metrics = []
                for _ in range(config["updates_per_rollout"]):
                    optimizer.zero_grad(set_to_none=True)
                    current, _ = token_logps(model, inputs, labels, attention)
                    loss, kl, clipped = grpo_loss(current, old, ref, advantages, mask, config["beta"], config["clip_epsilon"])
                    if not torch.isfinite(loss).item(): raise HTTPException(422, "GRPO 损失非有限，本批未保存")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True); optimizer.step()
                    metrics.append((loss.item(), kl.item(), clipped.item()))
                saved["step"] += 1; saved["optimizer_updates"] += config["updates_per_rollout"]
                saved["advantage_groups"] += int(std > 0)
                reward, validation = evaluate(model, tokenizer, prepared, config)
                row = {"step": saved["step"], "rollout_reward": sum(item["reward"] for item in scored) / len(scored), "reward_std": std,
                    "loss": sum(item[0] for item in metrics) / len(metrics), "kl": sum(item[1] for item in metrics) / len(metrics),
                    "clip_fraction": sum(item[2] for item in metrics) / len(metrics), "validation_reward": reward, "zero_variance": std == 0}
                if not all(math.isfinite(value) for value in row.values()): raise HTTPException(422, "GRPO 指标非有限，本批未保存")
                saved["history"].append(row)
                state = adapter_state(model)
                if reward > saved["best_reward"]: saved.update(best_reward=reward, best_step=saved["step"], best_adapter=copy.deepcopy(state))
                saved.update(adapter=state, validation_samples=validation, last_rollout={"prompt_index": index, "old_policy_step": saved["step"] - 1,
                    "old_adapter_digest": old_digest, "sampling_temperature": 1, "sampling_top_k": 0,
                    "samples": [{**sample, **score, "advantage": advantages[i].item()} for i, (sample, score) in enumerate(zip(samples, scored))]})
            if frozen_digest(model) != saved["frozen_digest"] or state_digest(adapter_state(reference)) != saved["reference_digest"]: raise RuntimeError("冻结校验失败，本批拒绝保存")
            saved.update(optimizer=optimizer.state_dict(), generator_state=generator.get_state(), adapter_changed=state_digest(saved["adapter"]) != saved["reference_digest"])
            self._save(saved); return report(saved)
        finally: self.lock.release()

    def compare(self, run_id, request):
        # Keep the exact same prompt, seed and shared history for both policies.
        result = super().compare(run_id, request)
        with self.lock: saved = self._load(run_id)
        for side in ("before", "after"):
            # A free-form question has no target label: omit target-specific rules and say so.
            rules = [rule for rule in saved["config"]["rules"] if request.reward_target.strip() or rule["type"] not in ("contains_target", "exact_target")]
            result[side]["rule_score"] = score_reward(result[side]["reply"], request.reward_target, rules)
        result["omitted_target_rules"] = 0 if request.reward_target.strip() else sum(rule["type"] in ("contains_target", "exact_target") for rule in saved["config"]["rules"])
        result["reward_target"] = request.reward_target
        result["sampling"].pop("reward_target", None)
        return result

    def export(self, run_id, variant):
        original = LoRAService.export(self, run_id, variant)
        with self.lock: saved = self._load(run_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
            for name in source.namelist(): package.writestr(name, source.read(name))
            package.writestr("chat_template.jinja", CHAT_TEMPLATE)
            package.writestr("grpo_format.json", json.dumps({"template_version": TEMPLATE_VERSION, "rule_version": RULE_VERSION, "rules": saved["config"]["rules"],
                "system_prompt": saved["config"]["system_prompt"], "max_length": saved["config"]["max_length"], "objective": "per-completion mean clipped GRPO + k3 KL",
                "source": saved["source"], "reference_digest": saved["reference_digest"], "dataset_fingerprint": saved["dataset_fingerprint"]}, indent=2))
        return buffer.getvalue()


service = GRPOService()
router = APIRouter(prefix="/api/grpo", tags=["在线强化学习微调 GRPO"])
RunId = Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")]


def dataset_payload(request):
    result = request.model_dump(); result["rules"] = rule_dicts(request.rules); return result


@router.get("/status")
def status(): return {**assets.status(), "dataset": DEFAULT_DATASET, "rules": DEFAULT_RULES, "system_prompt": DEFAULT_SYSTEM, "rule_version": RULE_VERSION, "max_steps": 100}


@router.post("/prepare")
def prepare(): return assets.prepare()


@router.post("/reward")
def reward(request: RewardRequest): return score_reward(request.text, request.target, rule_dicts(request.rules))


@router.post("/preview")
def preview(request: DatasetConfig): return service.preview(dataset_payload(request))


@router.get("/runs")
def list_runs(): return service.list()


@router.post("/runs")
def create_run(request: RunConfig): return service.create(dataset_payload(request))


@router.get("/runs/{run_id}")
def get_run(run_id: RunId): return service.get(run_id)


@router.post("/runs/{run_id}/train")
def train_run(run_id: RunId, request: TrainingRequest): return service.train(run_id, request)


@router.post("/runs/{run_id}/compare")
def compare(run_id: RunId, request: ComparisonRequest): return service.compare(run_id, request)


@router.get("/runs/{run_id}/adapter")
def download_adapter(run_id: RunId, checkpoint: Annotated[Literal["latest", "best"], Query()] = "latest"):
    return Response(service.export(run_id, checkpoint), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="grpo-{run_id[:8]}-{checkpoint}.zip"'})


@router.get("/runs/{run_id}/checkpoint")
def download_checkpoint(run_id: RunId):
    with service.lock: service._load(run_id); path = service._path(run_id)
    return FileResponse(path, media_type="application/octet-stream", filename=f"grpo-{run_id[:8]}-training.pt")
