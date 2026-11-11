import asyncio
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import torch

from backend.tests.test_experiments import asgi_request
from backend.transformer import (
    GenerationRequest, RunConfig, TrainingRequest, TransformerService,
    build_transformer, download_checkpoint,
)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = TransformerService(self.directory.name)
        self.config = RunConfig(layers=1).model_dump()

    def test_causal_mask_prefix_invariance_and_real_shapes(self):
        for heads in (2, 4):
            config = {**self.config, "heads": heads}
            model = build_transformer(12, config).eval()
            first = torch.tensor([[1, 2, 3, 4]])
            second = torch.tensor([[1, 2, 8, 9]])
            with torch.inference_mode():
                logits, attention = model(first, return_attention=True)
                changed = model(second)
            self.assertEqual(tuple(logits.shape), (1, 4, 12))
            self.assertEqual(tuple(attention.shape), (1, heads, 4, 4))
            torch.testing.assert_close(logits[:, :2], changed[:, :2], rtol=0, atol=0)
            torch.testing.assert_close(attention.sum(-1), torch.ones(1, heads, 4))
            self.assertEqual(torch.count_nonzero(attention.triu(1)).item(), 0)
            with torch.inference_mode():
                a = model(torch.tensor([[1, 2]]))[:, -1]
                b = model(torch.tensor([[3, 2]]))[:, -1]
            self.assertFalse(torch.allclose(a, b), "Unlike Bigram, earlier context affects the final prediction")

    def test_real_training_reduces_loss_and_records_post_update_metrics(self):
        run = self.service.create(self.config)
        count = run["parameter_count"]
        run = self.service.train(run["id"], TrainingRequest(expected_step=0, steps=25))
        self.assertGreater(count, len(run["vocab"]) ** 2)
        self.assertEqual(run["step"], 25)
        self.assertLess(run["history"][-1]["train_loss"], run["history"][0]["train_loss"])
        checkpoint = self.service._load(run["id"])
        model = build_transformer(len(run["vocab"]), run["config"])
        self.assertEqual(count, sum(p.numel() for p in model.parameters()))
        self.assertEqual(checkpoint["optimizer"]["state"][0]["step"].item(), 25)
        best = min(run["history"], key=lambda row: row["validation_loss"])
        self.assertEqual(run["best_step"], best["step"])

    def test_restart_resumes_optimizer_and_window_rng_exactly(self):
        first = self.service.create(self.config)
        reference = self.service.create(self.config)
        self.service.train(first["id"], TrainingRequest(expected_step=0, steps=7))
        restarted = TransformerService(self.directory.name)
        self.assertEqual(restarted.get(first["id"])["step"], 7)
        restarted.train(first["id"], TrainingRequest(expected_step=7, steps=18))
        self.service.train(reference["id"], TrainingRequest(expected_step=0, steps=25))
        a, b = restarted._load(first["id"]), self.service._load(reference["id"])
        for key, value in a["state_dict"].items():
            torch.testing.assert_close(value, b["state_dict"][key], rtol=0, atol=0)
        torch.testing.assert_close(a["generator_state"], b["generator_state"], rtol=0, atol=0)
        self.assertEqual(a["history"][-1], b["history"][-1])
        self.assertEqual(len(restarted.list()), 2)

    def test_validation_text_never_changes_current_training_weights(self):
        states = []
        for suffix in ("春" * 10, "猫" * 10):
            config = {**self.config, "corpus": "春猫" * 20 + suffix}
            run = self.service.create(config)
            self.service.train(run["id"], TrainingRequest(expected_step=0, steps=5))
            states.append(self.service._load(run["id"]))
        for key, value in states[0]["state_dict"].items():
            torch.testing.assert_close(value, states[1]["state_dict"][key], rtol=0, atol=0)
        self.assertNotEqual(states[0]["history"][-1]["validation_loss"], states[1]["history"][-1]["validation_loss"])

    def test_local_rng_and_generation_never_mutate_training_or_global_state(self):
        before = torch.get_rng_state().clone()
        run = self.service.create(self.config)
        run = self.service.train(run["id"], TrainingRequest(expected_step=0, steps=3))
        path = self.service._path(run["id"])
        checkpoint_bytes = path.read_bytes()
        request = GenerationRequest(prompt="小猫", seed=42, length=12, top_k=5)
        output = self.service.generate(run["id"], request)
        self.assertEqual(output, self.service.generate(run["id"], request))
        self.assertEqual(len(output["token_ids"]), 12)
        self.assertLessEqual(sum(item["probability"] > 0 for item in output["distribution"]), 5)
        self.assertAlmostEqual(sum(item["probability"] for item in output["distribution"]), 1, places=6)
        self.assertEqual(checkpoint_bytes, path.read_bytes())
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)

    def test_best_selection_context_truncation_and_attention(self):
        run = self.service.create(self.config)
        checkpoint = self.service._load(run["id"])
        original_best = copy.deepcopy(checkpoint["best_state_dict"])
        self.service.train(run["id"], TrainingRequest(expected_step=0, steps=10))
        # Force the initial checkpoint to be selected, to verify that selection
        # changes the actual loaded tensors rather than only the response label.
        checkpoint = self.service._load(run["id"])
        checkpoint.update(best_step=0, best_state_dict=original_best)
        self.service._save(checkpoint)
        prompt = "小猫" * 15
        best = self.service.generate(run["id"], GenerationRequest(prompt=prompt, checkpoint="best", length=5))
        latest = self.service.generate(run["id"], GenerationRequest(prompt=prompt, checkpoint="latest", length=5))
        self.assertEqual(best["step"], 0)
        self.assertEqual(latest["step"], 10)
        self.assertNotEqual(best["distribution"], latest["distribution"])
        self.assertEqual("".join(best["context_tokens"]), prompt[-16:])
        matrix = torch.tensor(latest["attention"])
        self.assertEqual(tuple(matrix.shape), (2, 16, 16))
        self.assertEqual(matrix.triu(1).count_nonzero().item(), 0)
        torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 16))
        warmer = self.service.generate(run["id"], GenerationRequest(prompt=prompt, length=5, temperature=2))
        self.assertEqual(latest["attention"], warmer["attention"], "Sampling temperature cannot alter first-pass attention")

    def test_failed_commit_retains_previous_checkpoint(self):
        run = self.service.create(self.config)
        previous = self.service._path(run["id"]).read_bytes()
        with patch("backend.transformer.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.service.train(run["id"], TrainingRequest(expected_step=0, steps=2))
        self.assertEqual(self.service._path(run["id"]).read_bytes(), previous)
        self.assertEqual(self.service.get(run["id"])["step"], 0)
        self.assertEqual(list(Path(self.directory.name).glob("*/*.tmp")), [])

    def test_stale_step_unknown_prompt_and_path_guards(self):
        run = self.service.create(self.config)
        with self.assertRaises(HTTPException) as caught:
            self.service.train(run["id"], TrainingRequest(expected_step=2, steps=1))
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(HTTPException) as caught:
            self.service.generate(run["id"], GenerationRequest(prompt="不存在X"))
        self.assertEqual(caught.exception.status_code, 422)
        for run_id in ("../checkpoint", "0" * 32):
            with self.assertRaises(HTTPException) as caught:
                self.service.get(run_id)
            self.assertEqual(caught.exception.status_code, 404)

    def test_http_contract_and_checkpoint_download(self):
        def call(method, path, body=None):
            return asyncio.run(asgi_request(method, path, body))

        with patch("backend.transformer.service", self.service):
            status, info, _ = call("GET", "/api/transformer/status")
            self.assertEqual(status, 200)
            self.assertTrue(info["available"])
            status, run, _ = call("POST", "/api/transformer/runs", {"layers": 1})
            self.assertEqual(status, 200)
            base = f"/api/transformer/runs/{run['id']}"
            status, trained, _ = call("POST", base + "/train", {"expected_step": 0, "steps": 3})
            self.assertEqual(status, 200)
            self.assertEqual(trained["step"], 3)
            status, generated, _ = call("POST", base + "/generate", {"length": 5})
            self.assertEqual(status, 200)
            self.assertEqual(len(generated["token_ids"]), 5)
            response = download_checkpoint(run["id"])
            saved = torch.load(response.path, weights_only=True)
            self.assertEqual(saved["step"], 3)
            self.assertIn("optimizer", saved)
            for body in ({"steps": 26, "expected_step": 3}, {"steps": 1, "expected_step": -1}, {"steps": 1}):
                self.assertEqual(call("POST", base + "/train", body)[0], 422)
            self.assertEqual(call("POST", "/api/transformer/runs", {"d_model": 48})[0], 422)
            self.assertEqual(call("GET", "/api/transformer/runs/not-an-id")[0], 422)


if __name__ == "__main__":
    unittest.main()
