"""Declarative architectures and their real training/checkpoint/inference path.

Numerical fixtures are synthetic tensors in temporary directories. They exercise
the real PyTorch pipeline without downloading data or claiming MNIST accuracy.
"""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from backend.inference import InferenceService, PredictRequest
from backend.mnist import RunConfig
from backend.mnist_engine import MnistService, build_model, torch_runtime
from backend.network_architecture import analyze_architecture, validate_architecture
from backend.tests.test_experiments import asgi_request


def cnn_layers():
    return [
        {"id": "conv-1", "type": "conv", "outChannels": 8, "kernel": 3, "stride": 1, "padding": 1},
        {"id": "relu-1", "type": "relu"}, {"id": "pool-1", "type": "pool"},
        {"id": "flat-1", "type": "flatten"}, {"id": "out-1", "type": "linear", "units": 10},
    ]


def mlp_layers():
    return [{"type": "flatten"}, {"type": "linear", "units": 128},
            {"type": "relu"}, {"type": "linear", "units": 10}]


class ArchitectureValidationTests(unittest.TestCase):
    def test_actual_torch_layers_match_declared_shapes_and_parameter_count(self):
        torch = torch_runtime()
        for layers, expected in [(cnn_layers(), 15770), (mlp_layers(), 101770)]:
            with self.subTest(layers=layers):
                analysis = analyze_architecture(layers)
                model = build_model("custom", layers)
                value = torch.zeros(3, 1, 28, 28)
                for module, row in zip(model, analysis["rows"]):
                    self.assertEqual(list(value.shape[1:]), row["input"])
                    value = module(value)
                    self.assertEqual(list(value.shape[1:]), row["output"])
                    self.assertEqual(sum(p.numel() for p in module.parameters()), row["parameters"])
                self.assertEqual(tuple(value.shape), (3, 10))
                self.assertEqual(analysis["total_parameters"], expected)

    def test_structure_rejects_unsupported_values_and_invalid_layer_geometry(self):
        invalid = [[], [{"type": "exec", "code": "arbitrary code"}],
                   [{"type": "linear", "units": 10}],
                   [{"type": "flatten"}, {"type": "pool"}],
                   [{"type": "pool"}] * 5 + [{"type": "flatten"}, {"type": "linear", "units": 10}],
                   [{"type": "relu"}] * 17,
                   [{"type": "flatten"}, {"type": "linear", "units": 32}],
                   [{"type": "flatten", "extra": "ignored?"}, {"type": "linear", "units": 10}]]
        for key, value in [("outChannels", True), ("outChannels", 8.0), ("kernel", "3"),
                           ("stride", False), ("padding", float("nan")), ("padding", float("inf"))]:
            layers = cnn_layers()
            layers[0][key] = value
            invalid.append(layers)
        for identifier in ["../outside", "", "x" * 65, False]:
            layers = cnn_layers()
            layers[0]["id"] = identifier
            invalid.append(layers)
        duplicate = cnn_layers()
        duplicate[1]["id"] = duplicate[0]["id"]
        invalid.append(duplicate)
        for layers in invalid:
            with self.subTest(layers=layers):
                with self.assertRaises(ValueError):
                    validate_architecture(layers)
        with patch("backend.network_architecture.MAX_PARAMETERS", 15000):
            with self.assertRaisesRegex(ValueError, "参数上限"):
                validate_architecture(cnn_layers())

    def test_request_contract_requires_custom_structure_and_keeps_old_models(self):
        self.assertEqual(RunConfig().model, "mlp")
        custom = RunConfig(model="custom", architecture=mlp_layers())
        self.assertEqual(custom.architecture[0]["id"], "layer-1")
        for body in [{"model": "custom"}, {"model": "custom", "architecture": []},
                     {"model": "cnn", "architecture": cnn_layers()}]:
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    RunConfig(**body)


class CustomTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = torch_runtime()
        generator = cls.torch.Generator().manual_seed(614)
        cls.fixture = {
            "train_images": cls.torch.randint(0, 256, (100, 28, 28), generator=generator, dtype=cls.torch.uint8),
            "train_labels": cls.torch.arange(100) % 10,
            "test_images": cls.torch.randint(0, 256, (30, 28, 28), generator=generator, dtype=cls.torch.uint8),
            "test_labels": cls.torch.arange(30) % 10,
        }

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.service = MnistService(self.root, dataset_provider=lambda: self.fixture)
        self.service._prepare_worker()

    def config(self, **changes):
        config = {"model": "custom", "architecture": cnn_layers(), "seed": 7, "epochs": 2,
                  "batch_size": 32, "learning_rate": 0.001, "validation_ratio": 0.2, "train_limit": None}
        config.update(changes)
        return config

    def train(self, **changes):
        run = self.service.create_run(self.config(**changes))
        result = self.service.wait_for_run(run["id"], timeout=20)
        self.assertEqual(result["status"], "completed", result)
        return result

    def test_real_training_checkpoint_restore_and_prediction_match_direct_forward(self):
        self.torch.manual_seed(7)
        initial = {key: value.clone() for key, value in build_model("custom", cnn_layers()).state_dict().items()}
        result = self.train()
        self.assertEqual(result["parameter_count"], 15770)
        self.assertEqual(result["config"]["architecture"], cnn_layers())
        self.assertEqual(sum(map(sum, result["evaluation"]["confusion_matrix"])), 30)
        checkpoint = self.torch.load(self.service.checkpoint_path(result["id"]), weights_only=True)
        self.assertTrue(any(not self.torch.equal(initial[key], value) for key, value in checkpoint["state_dict"].items()))
        best = min(result["history"], key=lambda row: row["validation_loss"])
        self.assertEqual(checkpoint["epoch"], best["epoch"])
        self.assertEqual(checkpoint["config"]["architecture"], cnn_layers())
        restored = MnistService(self.root)
        self.assertEqual(restored.get_run(result["id"])["config"], result["config"])
        self.assertEqual(restored.get_run(result["id"])["evaluation"], result["evaluation"])
        inference = InferenceService(self.root)
        catalog = inference.models()
        self.assertEqual(catalog["defaults"], {"mlp": None, "cnn": None})
        self.assertEqual(catalog["models"][0]["architecture"], "custom")
        self.assertEqual(catalog["models"][0]["structure"], cnn_layers())
        pixels = self.fixture["test_images"][0].numpy()
        response = inference.predict(PredictRequest(model_ids=[result["id"]], pixels=pixels.tolist(), mode="mnist"))
        direct = build_model("custom", checkpoint["config"]["architecture"])
        direct.load_state_dict(checkpoint["state_dict"])
        direct.eval()
        with self.torch.inference_mode():
            expected = direct(self.torch.as_tensor(pixels.copy()).reshape(1, 1, 28, 28).float().div(255).sub(.1307).div(.3081)).softmax(1)[0].numpy()
        np.testing.assert_allclose(response["predictions"][0]["probabilities"], expected, atol=1e-7, rtol=0)
        self.assertEqual(response["predictions"][0]["architecture"], "custom")

    def test_checkpoint_structure_mismatch_and_changed_metadata_are_not_cached_as_valid(self):
        result = self.train(epochs=1)
        inference = InferenceService(self.root)
        request = PredictRequest(model_ids=[result["id"]], pixels=self.fixture["test_images"][0].tolist(), mode="mnist")
        inference.predict(request)
        path = self.root / "runs" / "mnist" / result["id"] / "run.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["config"]["architecture"][0]["outChannels"] = 16
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(HTTPException) as failure:
            inference.predict(request)
        self.assertEqual(failure.exception.status_code, 503)

    def test_run_owns_a_structure_snapshot_not_mutable_caller_state(self):
        blocked, release = threading.Event(), threading.Event()
        worker = self.service._train_worker
        def wait_worker(*args):
            blocked.set()
            release.wait(5)
            worker(*args)
        config = self.config(epochs=1)
        with patch.object(self.service, "_train_worker", wait_worker):
            run = self.service.create_run(config)
            self.assertTrue(blocked.wait(3))
            config["architecture"][0]["outChannels"] = 64
            release.set()
            result = self.service.wait_for_run(run["id"], 20)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["config"]["architecture"][0]["outChannels"], 8)
        self.assertEqual(result["parameter_count"], 15770)

    def test_route_rejects_invalid_architectures_before_creating_worker_and_accepts_custom(self):
        bad = [None, [], [{"type": "python", "code": "print('not allowed')"}],
               [{"type": "flatten"}, {"type": "linear", "units": True}],
               [{"type": "linear", "units": 10}]]
        with patch("backend.mnist.service", self.service):
            for architecture in bad:
                with self.subTest(architecture=architecture):
                    status, _, _ = asyncio.run(asgi_request("POST", "/api/mnist/runs", self.config(architecture=architecture)))
                    self.assertEqual(status, 422)
            self.assertEqual(self.service.list_runs()["runs"], [])
            status, body, _ = asyncio.run(asgi_request("POST", "/api/mnist/runs", self.config(epochs=1)))
            self.assertEqual(status, 200, body)
            result = self.service.wait_for_run(body["id"], 20)
            self.assertEqual(result["status"], "completed", result)


if __name__ == "__main__":
    unittest.main()
