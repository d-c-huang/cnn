"""Input-gradient attribution uses real autograd and synthetic local weights."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np

from backend.inference import InferenceService, PredictRequest
from backend.mnist_engine import build_model, torch_runtime
from backend.tests.test_custom_networks import cnn_layers, mlp_layers
from backend.tests.test_experiments import asgi_request


class InputAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = torch_runtime()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.service = InferenceService(self.root)
        self.image = np.zeros((28, 28), dtype=np.uint8)
        self.image[4:24, 9:16] = 190
        self.image[6:10, 5:14] = 100

    def create_model(self, architecture="mlp", structure=None, identifier=None, zero=False):
        identifier = identifier or "a" * 32
        self.torch.manual_seed(143)
        model = build_model("custom", structure) if architecture == "custom" else build_model(architecture)
        model.eval()
        if zero:
            with self.torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        config = {"model": architecture}
        if structure is not None:
            config["architecture"] = structure
        checkpoint = {"config": config, "state_dict": model.state_dict(), "epoch": 1,
                      "validation_loss": .25, "normalization": {"mean": .1307, "std": .3081}}
        directory = self.root / "runs" / "mnist" / identifier
        directory.mkdir(parents=True)
        self.torch.save(checkpoint, directory / "best.pt")
        record = {"id": identifier, "created_at": "2026-10-28T00:00:00+00:00", "status": "completed",
                  "config": config, "best_epoch": 1, "best_validation_loss": .25, "evaluation": {"accuracy": .8}}
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
        return identifier, model

    def request(self, ids, explain=False, pixels=None, mode="mnist"):
        return PredictRequest(model_ids=ids, pixels=(self.image if pixels is None else pixels).tolist(), mode=mode, explain=explain)

    def direct_gradient(self, model, processed_pixels, target):
        variable = self.torch.as_tensor(np.asarray(processed_pixels).copy(), dtype=self.torch.float32).reshape(1, 1, 28, 28)
        variable = variable.div(255).sub(.1307).div(.3081).requires_grad_(True)
        gradient, = self.torch.autograd.grad(model(variable)[0, target], variable)
        magnitude = gradient.detach().abs()[0, 0]
        maximum = magnitude.max().item()
        return magnitude.div(maximum).numpy() if maximum else np.zeros((28, 28)), maximum

    def test_baseline_and_custom_attributions_match_direct_logit_gradient(self):
        variants = [("mlp", None), ("cnn", None), ("custom", mlp_layers()), ("custom", cnn_layers())]
        for index, (architecture, structure) in enumerate(variants, 1):
            with self.subTest(architecture=architecture, structure=structure):
                identifier, model = self.create_model(architecture, structure, f"{index:032x}")
                response = self.service.predict(self.request([identifier], explain=True))
                prediction = response["predictions"][0]
                attribution = prediction["attribution"]
                self.assertEqual(attribution["method"], "input_gradient")
                self.assertEqual(attribution["target_label"], prediction["predicted_label"])
                actual = np.asarray(attribution["heatmap"])
                self.assertEqual(actual.shape, (28, 28))
                self.assertTrue(np.isfinite(actual).all())
                self.assertGreaterEqual(float(actual.min()), 0)
                self.assertLessEqual(float(actual.max()), 1)
                expected, maximum = self.direct_gradient(model, response["preprocessing"]["processed_pixels"], prediction["predicted_label"])
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
                self.assertEqual(attribution["maximum_gradient"], maximum)

    def test_not_requested_keeps_original_prediction_schema_and_scores(self):
        identifier, _ = self.create_model()
        ordinary = self.service.predict(self.request([identifier]))["predictions"][0]
        explained = self.service.predict(self.request([identifier], explain=True))["predictions"][0]
        self.assertNotIn("attribution", ordinary)
        self.assertEqual(ordinary["probabilities"], explained["probabilities"])
        self.assertEqual(ordinary["predicted_label"], explained["predicted_label"])

    def test_canvas_attribution_uses_the_processed_28_by_28_normalized_input(self):
        identifier, model = self.create_model("cnn")
        canvas = np.repeat(np.repeat(self.image, 4, axis=0), 3, axis=1)
        response = self.service.predict(self.request([identifier], True, canvas, "canvas"))
        prediction = response["predictions"][0]
        expected, maximum = self.direct_gradient(model, response["preprocessing"]["processed_pixels"], prediction["predicted_label"])
        np.testing.assert_allclose(prediction["attribution"]["heatmap"], expected, atol=1e-7, rtol=0)
        self.assertEqual(prediction["attribution"]["maximum_gradient"], maximum)

    def test_zero_derivative_produces_a_zero_map_without_nan(self):
        identifier, _ = self.create_model(zero=True)
        prediction = self.service.predict(self.request([identifier], explain=True))["predictions"][0]
        self.assertEqual(prediction["attribution"]["maximum_gradient"], 0)
        self.assertEqual(prediction["attribution"]["heatmap"], [[0.0] * 28 for _ in range(28)])

    def test_repeated_and_concurrent_attribution_does_not_mutate_models_or_rng(self):
        identifier, _ = self.create_model()
        self.service.predict(self.request([identifier]))
        cached = self.service.cache[identifier]["model"]
        before_state = {name: value.clone() for name, value in cached.state_dict().items()}
        for parameter in cached.parameters():
            parameter.grad = self.torch.full_like(parameter, .125)
        before_rng = self.torch.get_rng_state().clone()
        def explain(_):
            return self.service.predict(self.request([identifier], explain=True))["predictions"][0]["attribution"]
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(explain, range(4)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertIs(self.service.cache[identifier]["model"], cached)
        self.assertFalse(cached.training)
        self.assertTrue(self.torch.equal(self.torch.get_rng_state(), before_rng))
        for name, value in cached.state_dict().items():
            self.assertTrue(self.torch.equal(value, before_state[name]))
        for parameter in cached.parameters():
            self.assertTrue(self.torch.equal(parameter.grad, self.torch.full_like(parameter, .125)))

    def test_inference_tensor_is_cloned_outside_inference_mode(self):
        _, model = self.create_model()
        with self.torch.inference_mode():
            inputs = self.torch.zeros((1, 1, 28, 28))
            self.assertTrue(inputs.is_inference())
            attribution = self.service._input_attribution(model, inputs, 1)
            self.assertTrue(self.torch.is_inference_mode_enabled())
        self.assertEqual(np.asarray(attribution["heatmap"]).shape, (28, 28))
        self.assertTrue(np.isfinite(attribution["maximum_gradient"]))

    def test_nonfinite_gradient_with_finite_logits_is_an_explicit_service_error(self):
        identifier, _ = self.create_model()
        torch = self.torch
        class InvalidDerivative(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value
            @staticmethod
            def backward(ctx, upstream):
                return torch.full_like(upstream, float("nan"))
        class InvalidModel(torch.nn.Module):
            def forward(self, value):
                return InvalidDerivative.apply(value).reshape(1, -1)[:, :10]
        with patch.object(self.service, "_load_model", return_value={"model": InvalidModel(), "normalization": {"mean": .1307, "std": .3081}}):
            with self.assertRaises(HTTPException) as failure:
                self.service.predict(self.request([identifier], explain=True))
        self.assertEqual(failure.exception.status_code, 503)
        self.assertIn("非有限输入梯度", failure.exception.detail)

    def test_http_requires_a_boolean_and_returns_real_attribution_when_requested(self):
        identifier, _ = self.create_model()
        body = {"model_ids": [identifier], "pixels": self.image.tolist(), "mode": "mnist"}
        with patch("backend.inference.service", self.service):
            for explain in (1, 0, "true", "false", None):
                with self.subTest(explain=explain):
                    status, _, _ = asyncio.run(asgi_request("POST", "/api/inference/predict", {**body, "explain": explain}))
                    self.assertEqual(status, 422)
            status, response, _ = asyncio.run(asgi_request("POST", "/api/inference/predict", {**body, "explain": True}))
            self.assertEqual(status, 200, response)
            self.assertEqual(response["predictions"][0]["attribution"]["method"], "input_gradient")
            status, response, _ = asyncio.run(asgi_request("POST", "/api/inference/predict", body))
            self.assertEqual(status, 200)
            self.assertNotIn("attribution", response["predictions"][0])


if __name__ == "__main__":
    unittest.main()
