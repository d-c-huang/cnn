import asyncio
import copy
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from fastapi import HTTPException
import torch

from backend.lora import LoRAService, RunConfig, TrainingRequest, GenerationRequest, adapter_state, frozen_digest
from backend.lora_assets import assets, MODEL_ID, MODEL_REVISION, MODEL_SHA256, REQUIRED_FILES, PretrainedAssets
from backend.tests.test_experiments import asgi_request

HAS_PEFT = all(importlib.util.find_spec(name) is not None for name in ("peft", "transformers", "safetensors"))


class PretrainedCacheTests(unittest.TestCase):
    def test_missing_cache_never_creates_a_random_base(self):
        with TemporaryDirectory() as directory:
            cache = PretrainedAssets(directory)
            self.assertFalse(cache.cached())
            service = LoRAService(Path(directory) / "runs", base_assets=cache)
            with self.assertRaises(HTTPException) as error:
                service.create(RunConfig().model_dump())
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(list(Path(directory).glob("runs/*/checkpoint.pt")), [])

    def test_model_identity_and_byte_hash_are_checked(self):
        with TemporaryDirectory() as directory:
            cache = PretrainedAssets(directory)
            root = Path(directory)
            for filename in REQUIRED_FILES:
                (root / filename).write_bytes(b"invalid model fixture")
            marker = root / "zhiwei-base.json"
            marker.write_text(json.dumps({"model_id": MODEL_ID, "revision": "wrong", "weight_sha256": MODEL_SHA256}))
            self.assertFalse(cache.cached())
            marker.write_text(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": MODEL_SHA256}))
            service = LoRAService(root / "runs", base_assets=cache)
            with self.assertRaises(HTTPException) as error:
                service.create(RunConfig().model_dump())
            self.assertEqual(error.exception.status_code, 503)


class FixtureTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(char) % 24 for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + token % 26) for token in ids if token or not skip_special_tokens)


def fixture_loader():
    from transformers import AutoModelForCausalLM, GPTNeoXConfig

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = AutoModelForCausalLM.from_config(GPTNeoXConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
            attention_dropout=0.0, hidden_dropout=0.0))
    return model, FixtureTokenizer()


@unittest.skipUnless(HAS_PEFT, "Install the optional LoRA dependencies to run PEFT tests")
class LoRATests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = LoRAService(self.directory.name, loader=fixture_loader)
        self.config = RunConfig(corpus="Zhiwei learns patterns from small experiments. " * 4, context_length=16).model_dump()

    def create(self):
        return self.service.create(self.config)

    def test_initial_adapter_matches_base_and_only_lora_is_trainable(self):
        run = self.create()
        model, _ = self.service._model(run["config"], self.service._load(run["id"])["adapter"])
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(names), 2)
        self.assertTrue(all("lora_" in name for name in names))
        self.assertEqual(run["trainable_parameters"], 4 * (16 + 48))
        self.assertLess(run["trainable_parameters"], run["base_parameters"])
        model.eval()
        x = torch.tensor([[3, 4, 5]])
        with torch.inference_mode():
            adapted = model(input_ids=x).logits
            with model.disable_adapter():
                baseline = model(input_ids=x).logits
        torch.testing.assert_close(adapted, baseline, rtol=0, atol=0)
        output = self.service.generate(run["id"], GenerationRequest(length=8))
        self.assertEqual(output["baseline"], output["adapted"])

    def test_real_gradients_update_adapters_and_keep_base_identical(self):
        run = self.create()
        before = self.service._load(run["id"])
        run = self.service.train(run["id"], TrainingRequest(expected_step=0, steps=5))
        after = self.service._load(run["id"])
        self.assertEqual(after["frozen_digest"], before["frozen_digest"])
        self.assertTrue(run["frozen_verified"])
        self.assertTrue(any(not torch.equal(before["adapter"][key], value) for key, value in after["adapter"].items()))
        self.assertLess(run["history"][-1]["train_loss"], run["history"][0]["train_loss"])
        model, _ = self.service._model(run["config"], after["adapter"])
        self.assertEqual(frozen_digest(model), before["frozen_digest"])

    def test_restart_resumes_adapter_optimizer_and_rng_exactly(self):
        a, b = self.create(), self.create()
        self.service.train(a["id"], TrainingRequest(expected_step=0, steps=2))
        restarted = LoRAService(self.directory.name, loader=fixture_loader)
        restarted.train(a["id"], TrainingRequest(expected_step=2, steps=3))
        self.service.train(b["id"], TrainingRequest(expected_step=0, steps=5))
        left, right = restarted._load(a["id"]), self.service._load(b["id"])
        for name, value in left["adapter"].items():
            torch.testing.assert_close(value, right["adapter"][name], rtol=0, atol=0)
        torch.testing.assert_close(left["generator_state"], right["generator_state"], rtol=0, atol=0)
        self.assertEqual(left["history"][-1], right["history"][-1])
        self.assertEqual(restarted.get(a["id"])["step"], 5)

    def test_validation_changes_metrics_without_changing_gradients(self):
        states = []
        for suffix in ("a" * 30, "b" * 30):
            config = {**self.config, "corpus": "Zhiwei lab. " * 10 + suffix}
            run = self.service.create(config)
            self.service.train(run["id"], TrainingRequest(expected_step=0, steps=3))
            states.append(self.service._load(run["id"]))
        for name, value in states[0]["adapter"].items():
            torch.testing.assert_close(value, states[1]["adapter"][name], rtol=0, atol=0)
        self.assertNotEqual(states[0]["history"][-1]["validation_loss"], states[1]["history"][-1]["validation_loss"])

    def test_generation_reproducibility_no_global_rng_or_checkpoint_changes(self):
        run = self.create()
        self.service.train(run["id"], TrainingRequest(expected_step=0, steps=5))
        checkpoint = self.service._path(run["id"]).read_bytes()
        rng = torch.get_rng_state().clone()
        request = GenerationRequest(prompt="Zhiwei learns", seed=42, length=12)
        a = self.service.generate(run["id"], request)
        b = self.service.generate(run["id"], request)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a["adapted"]["token_ids"]), 12)
        self.assertEqual(checkpoint, self.service._path(run["id"]).read_bytes())
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_best_adapter_is_actually_loaded_and_zip_reloads_with_peft(self):
        from peft import PeftModel

        run = self.create()
        original = self.service._load(run["id"])
        self.service.train(run["id"], TrainingRequest(expected_step=0, steps=5))
        saved = self.service._load(run["id"])
        saved.update(best_step=0, best_adapter=copy.deepcopy(original["adapter"]))
        self.service._save(saved)
        result = self.service.generate(run["id"], GenerationRequest(checkpoint="best", length=8))
        self.assertEqual(result["step"], 0)
        self.assertEqual(result["baseline"], result["adapted"])
        data = self.service.export(run["id"], "latest")
        destination = Path(self.directory.name) / "export"
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            self.assertEqual(set(package.namelist()), {"adapter_config.json", "adapter_model.safetensors", "zhiwei_manifest.json", "README.txt"})
            metadata = json.loads(package.read("zhiwei_manifest.json"))
            self.assertEqual(metadata["model_id"], MODEL_ID)
            self.assertEqual(metadata["revision"], MODEL_REVISION)
            package.extractall(destination)
        base, _ = fixture_loader()
        loaded = PeftModel.from_pretrained(base, str(destination), local_files_only=True)
        for name, value in adapter_state(loaded).items():
            torch.testing.assert_close(value, saved["adapter"][name], rtol=0, atol=0)

    def test_failed_commit_and_stale_step_preserve_checkpoint(self):
        run = self.create()
        previous = self.service._path(run["id"]).read_bytes()
        with patch("backend.transformer.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.service.train(run["id"], TrainingRequest(expected_step=0, steps=1))
        self.assertEqual(previous, self.service._path(run["id"]).read_bytes())
        with self.assertRaises(HTTPException) as error:
            self.service.train(run["id"], TrainingRequest(expected_step=2, steps=1))
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(list(Path(self.directory.name).glob("*/*.tmp")), [])

    def test_http_validation_and_real_training_routes(self):
        def call(method, path, body=None):
            return asyncio.run(asgi_request(method, path, body))

        with patch("backend.lora.service", self.service):
            status, run, _ = call("POST", "/api/lora/runs", self.config)
            self.assertEqual(status, 200)
            base = f"/api/lora/runs/{run['id']}"
            status, trained, _ = call("POST", base + "/train", {"expected_step": 0, "steps": 2})
            self.assertEqual(status, 200)
            self.assertEqual(trained["step"], 2)
            self.assertEqual(call("POST", base + "/generate", {"length": 4})[0], 200)
            self.assertEqual(call("POST", base + "/train", {"expected_step": 0, "steps": 2})[0], 409)
            self.assertEqual(call("POST", base + "/train", {"expected_step": 2, "steps": 6})[0], 422)
            self.assertEqual(call("POST", "/api/lora/runs", {"rank": 3})[0], 422)
            self.assertEqual(call("GET", "/api/lora/runs/not-an-id")[0], 422)


@unittest.skipUnless(HAS_PEFT and assets.cached(), "Explicitly prepare the pinned public base model for this integration check")
class PretrainedLoRAIntegrationTests(unittest.TestCase):
    def test_actual_pretrained_pythia_finetuning_and_export(self):
        with TemporaryDirectory() as root:
            service = LoRAService(root)
            initial = service.create(RunConfig().model_dump())
            self.assertEqual(initial["base_parameters"], 14_067_712)
            self.assertEqual(initial["trainable_parameters"], 12_288)
            self.assertEqual(initial["weight_sha256"], "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1")
            trained = service.train(initial["id"], TrainingRequest(expected_step=0, steps=5))
            self.assertLess(trained["history"][-1]["train_loss"], initial["history"][0]["train_loss"])
            self.assertTrue(trained["frozen_verified"])
            output = service.generate(initial["id"], GenerationRequest(length=8))
            self.assertTrue(output["baseline"]["text"])
            self.assertTrue(output["adapted"]["text"])
            self.assertGreater(len(service.export(initial["id"], "latest")), 1000)


if __name__ == "__main__":
    unittest.main()
