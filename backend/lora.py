"""Real PEFT LoRA on a pinned Pythia-14M pretrained causal language model."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from typing import Annotated, Literal
from uuid import uuid4
import zipfile

from fastapi import APIRouter, HTTPException, Path as ApiPath, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.lora_assets import assets, MODEL_ID, MODEL_REVISION, sha256_file
from backend.mnist_engine import ROOT, torch_runtime
from backend.transformer import TransformerService

DEFAULT_CORPUS = (
    "Zhiwei is a small learning lab. We learn by running clear experiments.\n"
    "At Zhiwei, a model learns patterns from training examples. Validation examples are kept separate.\n"
    "Our guide is simple: prepare the data, run an experiment, and inspect the result.\n"
    "Zhiwei uses small models to explain big ideas. Every training curve comes from a real calculation.\n"
    "A good experiment changes one setting at a time. Record the learning rate and random seed.\n"
    "LoRA keeps the pretrained model frozen and learns small adapter matrices.\n"
    "The Zhiwei lab values patient observation. A low training loss does not guarantee a good answer.\n"
    "We compare the original model with the adapted model using the same prompt.\n"
    "At Zhiwei, we save our adapters and keep the base model version with every experiment.\n"
    "Learning starts with a question. The next step is a small, reproducible experiment.\n"
)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: str = Field(default=DEFAULT_CORPUS, min_length=100, max_length=6000)
    rank: Literal[2, 4, 8, 16] = 4
    alpha: Literal[4, 8, 16, 32] = 8
    context_length: Literal[16, 32, 64] = 32
    batch_size: Literal[1, 2, 4] = 2
    learning_rate: float = Field(default=0.003, ge=0.0001, le=0.01, allow_inf_nan=False)
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_step: int = Field(strict=True, ge=0, le=500)
    steps: int = Field(default=5, strict=True, ge=1, le=5)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default="Zhiwei is", min_length=1, max_length=256)
    length: int = Field(default=32, strict=True, ge=1, le=64)
    temperature: float = Field(default=0.8, ge=0.1, le=2, allow_inf_nan=False)
    top_k: Literal[0, 10, 50] = 10
    seed: int = Field(default=7, strict=True, ge=0, le=2**32 - 1)
    checkpoint: Literal["latest", "best"] = "latest"


def adapter_config(config, inference=False):
    return {"base_model_name_or_path": MODEL_ID, "revision": MODEL_REVISION,
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": inference,
            "r": config["rank"], "lora_alpha": config["alpha"], "lora_dropout": 0.0,
            "bias": "none", "target_modules": ["query_key_value"], "fan_in_fan_out": False}


def inject_lora(base, config):
    from peft import LoraConfig, get_peft_model

    torch = torch_runtime()
    # PEFT allocates the adapter on meta, then we materialize with a local RNG.
    # This avoids reseeding a concurrent MNIST or Transformer experiment.
    model = get_peft_model(base, LoraConfig(**adapter_config(config)), low_cpu_mem_usage=True)
    generator = torch.Generator().manual_seed(config["seed"])
    for module in model.modules():
        if hasattr(module, "lora_A"):
            for layer in module.lora_A.values():
                layer.to_empty(device="cpu")
                with torch.no_grad():
                    layer.weight.uniform_(-layer.in_features**-0.5, layer.in_features**-0.5, generator=generator)
            for layer in module.lora_B.values():
                layer.to_empty(device="cpu")
                with torch.no_grad():
                    layer.weight.zero_()
    model.config.use_cache = False
    return model


def adapter_state(model):
    from peft import get_peft_model_state_dict

    return {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}


def load_adapter(model, state):
    from peft import set_peft_model_state_dict

    if set(state) != set(adapter_state(model)):
        raise HTTPException(500, "适配器参数与当前结构不匹配")
    set_peft_model_state_dict(model, state)


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if "lora_" not in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encode_parts(tokenizer, corpus):
    torch = torch_runtime()
    cut = int(len(corpus) * 0.8)
    parts = [torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long) for text in (corpus[:cut], corpus[cut:])]
    if len(parts[0]) < 4 or len(parts[1]) < 2:
        raise HTTPException(422, "分词后训练至少需要 4 个 token，验证至少需要 2 个 token")
    return parts


def evaluate(model, parts, context_length):
    torch = torch_runtime()
    model.eval()
    losses = []
    with torch.inference_mode():
        for ids in parts:
            total = 0.0
            for start in range(0, len(ids) - 1, context_length):
                stop = min(start + context_length, len(ids) - 1)
                logits = model(input_ids=ids[start:stop].unsqueeze(0), use_cache=False).logits[0]
                total += torch.nn.functional.cross_entropy(logits, ids[start + 1:stop + 1], reduction="sum").item()
            losses.append(total / (len(ids) - 1))
    if not all(math.isfinite(value) for value in losses):
        raise HTTPException(422, "损失非有限，本批未保存")
    return {"train_loss": losses[0], "validation_loss": losses[1], "perplexity": math.exp(min(losses[1], 700))}


def report(saved):
    keys = ("id", "created_at", "config", "model_id", "revision", "weight_sha256", "step", "best_step", "history",
            "base_parameters", "trainable_parameters", "trainable_ratio", "train_tokens", "validation_tokens", "training_context", "frozen_verified")
    return {key: copy.deepcopy(saved[key]) for key in keys}


class LoRAService(TransformerService):
    def __init__(self, root=None, base_assets=None, loader=None):
        super().__init__(root if root is not None else ROOT / "runs" / "lora")
        self.assets = base_assets or assets
        self.loader = loader
        self.base = None
        self.tokenizer = None

    def _runtime(self):
        if self.base is None:
            if self.loader:
                self.base, self.tokenizer = self.loader()
            else:
                directory = self.assets.require()
                info = json.loads((directory / "zhiwei-base.json").read_text(encoding="utf-8"))
                if sha256_file(directory / "model.safetensors") != info["weight_sha256"]:
                    raise HTTPException(503, "缓存权重校验失败，请修复固定版本模型缓存")
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    base = AutoModelForCausalLM.from_pretrained(str(directory), local_files_only=True,
                        use_safetensors=True, trust_remote_code=False, dtype=torch_runtime().float32, attn_implementation="eager")
                    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
                    self.base, self.tokenizer = base, tokenizer
                except ImportError as error:
                    raise HTTPException(503, "请安装 requirements.txt 中的 LoRA 依赖") from error
            self.base.eval()
            self.base.requires_grad_(False)
            self.base.name_or_path = MODEL_ID
            self.base.config._name_or_path = MODEL_ID
        return self.base, self.tokenizer

    def _model(self, config, state=None):
        base, tokenizer = self._runtime()
        model = inject_lora(copy.deepcopy(base), config)
        if state is not None:
            load_adapter(model, state)
        return model, tokenizer

    def _load(self, run_id):
        saved = super()._load(run_id)
        if saved.get("kind") != "pretrained-lora" or saved.get("model_id") != MODEL_ID or saved.get("revision") != MODEL_REVISION:
            raise HTTPException(409, "实验的基座模型或版本与当前教程不一致")
        return saved

    def get(self, run_id):
        with self.lock:
            return report(self._load(run_id))

    def list(self):
        with self.lock:
            if not self.root.exists():
                return []
            paths = sorted(self.root.glob("*/checkpoint.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
            return [report(self._load(path.parent.name)) for path in paths[:30]]

    def create(self, config):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "LoRA 正在计算，请稍后重试")
        try:
            torch = torch_runtime()
            model, tokenizer = self._model(config)
            parts = encode_parts(tokenizer, config["corpus"])
            trainable = [p for p in model.parameters() if p.requires_grad]
            if any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
                raise RuntimeError("检测到未冻结的基座参数")
            optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.0)
            metric = {"step": 0, **evaluate(model, parts, config["context_length"])}
            state = adapter_state(model)
            base_count = sum(p.numel() for name, p in model.named_parameters() if "lora_" not in name)
            trainable_count = sum(p.numel() for p in trainable)
            weight_sha = "test-fixture" if self.loader else json.loads((self.assets.require() / "zhiwei-base.json").read_text(encoding="utf-8"))["weight_sha256"]
            saved = {"version": 1, "kind": "pretrained-lora", "id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                "model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": weight_sha, "config": config,
                "step": 0, "best_step": 0, "best_loss": metric["validation_loss"], "history": [metric],
                "adapter": state, "best_adapter": copy.deepcopy(state), "optimizer": optimizer.state_dict(),
                "generator_state": torch.Generator().manual_seed(config["seed"]).get_state(),
                "base_parameters": base_count, "trainable_parameters": trainable_count,
                "trainable_ratio": trainable_count / (base_count + trainable_count),
                "train_tokens": len(parts[0]), "validation_tokens": len(parts[1]),
                "training_context": min(config["context_length"], len(parts[0]) - 1),
                "frozen_digest": frozen_digest(model), "frozen_verified": True}
            self._save(saved)
            return report(saved)
        finally:
            self.lock.release()

    def train(self, run_id, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "LoRA 正在计算，请稍后刷新记录")
        try:
            saved = self._load(run_id)
            if saved["step"] != request.expected_step:
                raise HTTPException(409, "训练步数已改变，请先刷新记录，避免重复更新")
            if saved["step"] + request.steps > 500:
                raise HTTPException(422, "单实验最多 500 步，请创建新实验")
            torch = torch_runtime()
            config = saved["config"]
            model, tokenizer = self._model(config, saved["adapter"])
            before = frozen_digest(model)
            if before != saved["frozen_digest"]:
                raise HTTPException(409, "基座参数与创建实验时不一致")
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
            optimizer.load_state_dict(saved["optimizer"])
            generator = torch.Generator()
            generator.set_state(saved["generator_state"])
            parts = encode_parts(tokenizer, config["corpus"])
            ids, length = parts[0], saved["training_context"]
            model.train()
            for _ in range(request.steps):
                starts = torch.randint(len(ids) - length, (config["batch_size"],), generator=generator)
                indices = starts[:, None] + torch.arange(length)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=ids[indices], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[indices + 1].reshape(-1))
                if not torch.isfinite(loss).item():
                    raise HTTPException(422, "训练损失非有限，本批未保存")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
            if frozen_digest(model) != before:
                raise RuntimeError("基座参数发生变化，本批拒绝保存")
            saved["step"] += request.steps
            metric = {"step": saved["step"], **evaluate(model, parts, config["context_length"])}
            state = adapter_state(model)
            if metric["validation_loss"] < saved["best_loss"]:
                saved.update(best_loss=metric["validation_loss"], best_step=saved["step"], best_adapter=copy.deepcopy(state))
            saved["history"].append(metric)
            saved.update(adapter=state, optimizer=optimizer.state_dict(), generator_state=generator.get_state(), frozen_verified=True)
            self._save(saved)
            return report(saved)
        finally:
            self.lock.release()

    def generate(self, run_id, request):
        # Model creation uses the cached immutable base under the service lock.
        with self.lock:
            saved = self._load(run_id)
            state = saved["best_adapter"] if request.checkpoint == "best" else saved["adapter"]
            model, tokenizer = self._model(saved["config"], state)
        torch = torch_runtime()
        model.eval()
        ids = tokenizer.encode(request.prompt, add_special_tokens=False)
        if not ids:
            raise HTTPException(422, "提示词分词后为空")
        context_length = saved["config"]["context_length"]

        def sample():
            sequence = list(ids)
            generator = torch.Generator().manual_seed(request.seed)
            output = []
            with torch.inference_mode():
                for _ in range(request.length):
                    x = torch.tensor([sequence[-context_length:]], dtype=torch.long)
                    scores = model(input_ids=x, use_cache=False).logits[0, -1] / request.temperature
                    if request.top_k:
                        keep = torch.topk(scores, min(request.top_k, len(scores))).indices
                        mask = torch.ones_like(scores, dtype=torch.bool)
                        mask[keep] = False
                        scores = scores.masked_fill(mask, float("-inf"))
                    token = int(torch.multinomial(scores.softmax(-1), 1, generator=generator).item())
                    output.append(token)
                    sequence.append(token)
                    if token == tokenizer.eos_token_id:
                        break
            return {"text": tokenizer.decode(output, skip_special_tokens=True), "token_ids": output,
                    "finish_reason": "eos" if output[-1] == tokenizer.eos_token_id else "length"}

        with model.disable_adapter():
            baseline = sample()
        adapted = sample()
        return {"run_id": run_id, "model_id": saved["model_id"], "revision": saved["revision"],
                "step": saved["best_step"] if request.checkpoint == "best" else saved["step"],
                "checkpoint": request.checkpoint, "prompt": request.prompt, "baseline": baseline, "adapted": adapted,
                "context_tokens": len(ids[-context_length:]), "frozen_verified": saved["frozen_verified"]}

    def export(self, run_id, variant):
        from safetensors.torch import save

        with self.lock:
            saved = self._load(run_id)
        state = saved["best_adapter"] if variant == "best" else saved["adapter"]
        metadata = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": saved["weight_sha256"],
                    "run_id": run_id, "variant": variant, "step": saved["best_step"] if variant == "best" else saved["step"]}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr("adapter_model.safetensors", save({key: value.contiguous() for key, value in state.items()}))
            package.writestr("adapter_config.json", json.dumps(adapter_config(saved["config"], inference=True), indent=2))
            package.writestr("zhiwei_manifest.json", json.dumps(metadata, indent=2))
            package.writestr("README.txt", f"PEFT LoRA adapter only. Base: {MODEL_ID}\nRevision: {MODEL_REVISION}\nLoad the pinned base with AutoModelForCausalLM, then PeftModel.from_pretrained(base, adapter_directory).\n")
        return buffer.getvalue()


service = LoRAService()
router = APIRouter(prefix="/api/lora", tags=["LoRA 预训练语言模型微调"])
RunId = Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")]


@router.get("/status")
def status():
    return {**assets.status(), "corpus": DEFAULT_CORPUS, "max_steps": 500}


@router.post("/prepare")
def prepare():
    return assets.prepare()


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


@router.get("/runs/{run_id}/adapter")
def download_adapter(run_id: RunId, checkpoint: Annotated[Literal["latest", "best"], Query()] = "latest"):
    return Response(service.export(run_id, checkpoint), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="lora-{run_id[:8]}-{checkpoint}.zip"'})


@router.get("/runs/{run_id}/checkpoint")
def download_checkpoint(run_id: RunId):
    with service.lock:
        service._load(run_id)
        path = service._path(run_id)
    return FileResponse(path, media_type="application/octet-stream", filename=f"lora-{run_id[:8]}-training.pt")
