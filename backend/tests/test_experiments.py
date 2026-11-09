"""Run with: python -m unittest discover -s backend/tests -v.

The small ASGI client deliberately uses only the standard library, so the test
suite needs no extra HTTP test dependency or running server.
"""

import asyncio
import json
import math
import unittest

from backend.app import _data_experiment, app, data_experiment


async def asgi_request(method, path, body=None, headers=None):
    payload = json.dumps(body, allow_nan=True).encode() if body is not None else b""
    response_messages = []
    request_headers = [(b"content-type", b"application/json")]
    request_headers.extend(headers or [])
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path,
        "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": request_headers, "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        response_messages.append(message)

    await app(scope, receive, send)
    start = next(message for message in response_messages if message["type"] == "http.response.start")
    raw = b"".join(message.get("body", b"") for message in response_messages if message["type"] == "http.response.body")
    return start["status"], json.loads(raw) if raw else None, dict(start["headers"])


class ExperimentTests(unittest.TestCase):
    def call(self, path, body=None, method="POST"):
        return asyncio.run(asgi_request(method, path, body))

    def successful(self, path, body):
        status, result, _ = self.call(path, body)
        self.assertEqual(status, 200, result)
        return result

    def test_health(self):
        status, result, _ = self.call("/api/health", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(result, {"status": "ok", "engine": "numpy", "version": "0.1.0"})

    def test_normalization_and_row_major_flattening(self):
        result = self.successful("/api/data", {"pixels": [[0, 255], [127.5, 51]], "normalize": True})
        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["matrix"], [[0, 1], [0.5, 0.2]])
        self.assertEqual(result["flat"], [0, 1, 0.5, 0.2])
        self.assertAlmostEqual(result["mean"], 0.425)

    def test_raw_pixels(self):
        result = self.successful("/api/data", {"pixels": [[12, 34]], "normalize": False})
        self.assertEqual(result["flat"], [12, 34])
        self.assertEqual(result["min"], 12)
        self.assertEqual(result["max"], 34)

    def test_data_route_uses_validated_request_once(self):
        routes = [
            route for route in app.routes
            if getattr(route, "path", None) == "/api/data"
            and "POST" in getattr(route, "methods", set())
        ]
        self.assertEqual(len(routes), 1)
        self.assertIs(routes[0].endpoint, data_experiment)
        operation = app.openapi()["paths"]["/api/data"]["post"]
        self.assertEqual(
            operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/DataRequest",
        )
        self.assertFalse(operation.get("parameters"))

    def test_data_cache_preserves_normalization_and_validation(self):
        _data_experiment.cache_clear()
        self.addCleanup(_data_experiment.cache_clear)
        body = {"pixels": [[0, 255], [127.5, 51]]}
        first = self.successful("/api/data", body)
        repeated = self.successful("/api/data", body)
        self.assertEqual(first, repeated)
        self.assertEqual(_data_experiment.cache_info().hits, 1)
        raw = self.successful("/api/data", {**body, "normalize": False})
        self.assertEqual(raw["flat"], [0, 255, 127.5, 51])
        self.assertEqual(first["flat"], [0, 1, 0.5, 0.2])
        self.assertEqual(_data_experiment.cache_info().misses, 2)
        before = _data_experiment.cache_info()
        for invalid in [
            {"pixels_tuple": [[0, 255]], "normalize": True},
            {**body, "extra": 1},
            {"pixels": [[256]]},
        ]:
            with self.subTest(body=invalid):
                status, _, _ = self.call("/api/data", invalid)
                self.assertEqual(status, 422)
        self.assertEqual(_data_experiment.cache_info(), before)

    def test_linear_best_fit_and_mse(self):
        best = self.successful("/api/linear", {"weight": 2, "bias": 0.5})
        self.assertEqual(best["mse"], 0)
        self.assertEqual([p["y"] for p in best["points"]], [-3.5, -1.5, 0.5, 2.5, 4.5])
        other = self.successful("/api/linear", {"weight": 0, "bias": 0})
        self.assertAlmostEqual(other["mse"], 8.25)
        self.assertAlmostEqual(other["points"][0]["error"], 3.5)

    def test_neuron_activations_and_stable_sigmoid(self):
        expected = {"linear": -1, "relu": 0, "sigmoid": 1 / (1 + math.exp(1)), "tanh": math.tanh(-1)}
        for name, output in expected.items():
            with self.subTest(activation=name):
                result = self.successful("/api/neuron", {"inputs": [1, 2], "weights": [2, -2], "bias": 1, "activation": name})
                self.assertEqual(result["products"], [2, -4])
                self.assertEqual(result["z"], -1)
                self.assertAlmostEqual(result["output"], output)
                self.assertGreaterEqual(len(result["curve"]), 121)
        extreme = self.successful("/api/neuron", {"inputs": [1000, 1000], "weights": [-1000, -1000], "bias": 0, "activation": "sigmoid"})
        self.assertEqual(extreme["output"], 0)

    def test_neuron_curve_covers_current_z(self):
        for sign in [-1, 1]:
            result = self.successful("/api/neuron", {"inputs": [3, 3], "weights": [sign * 3, sign * 3], "bias": sign * 3, "activation": "relu"})
            self.assertEqual(result["z"], sign * 21)
            self.assertLessEqual(result["curve"][0]["x"], min(-6, result["z"]) - 0.5)
            self.assertGreaterEqual(result["curve"][-1]["x"], max(6, result["z"]) + 0.5)
            self.assertIn({"x": result["z"], "y": result["output"]}, result["curve"])

    def test_eight_activation_derivatives_and_extremes(self):
        at_zero = {
            "relu": (0, 0, False), "leaky_relu": (0, 0.1, False),
            "elu": (0, 1, True), "softplus": (math.log(2), 0.5, True),
            "silu": (0, 0.5, True), "sigmoid": (0.5, 0.25, True),
            "tanh": (0, 1, True), "linear": (0, 1, True),
        }
        for activation, (output, derivative, differentiable) in at_zero.items():
            def evaluate(z):
                return self.successful("/api/neuron", {"inputs": [0, 0], "weights": [0, 0], "bias": z, "activation": activation})

            zero = evaluate(0)
            self.assertAlmostEqual(zero["output"], output)
            self.assertAlmostEqual(zero["derivative"], derivative)
            self.assertEqual(zero["differentiable"], differentiable)
            for z in [-1000, -3, -0.5, 0.5, 3, 1000]:
                result = evaluate(z)
                self.assertTrue(math.isfinite(result["output"]))
                self.assertTrue(math.isfinite(result["derivative"]))
                self.assertTrue(all(math.isfinite(point["y"]) for point in result["curve"]))
                if abs(z) < 1000:
                    numerical = (evaluate(z + 1e-5)["output"] - evaluate(z - 1e-5)["output"]) / 2e-5
                    self.assertAlmostEqual(result["derivative"], numerical, places=7)
            if activation == "silu":
                self.assertLess(evaluate(-3)["derivative"], 0)
                self.assertGreater(evaluate(3)["derivative"], 1)

    def test_loss_and_curve_domain(self):
        result = self.successful("/api/loss", {"prediction": 12, "target": 2})
        self.assertEqual(result["loss"], 100)
        self.assertEqual(result["gradient"], 20)
        self.assertLessEqual(result["curve"][0]["x"], -3)
        self.assertGreaterEqual(result["curve"][-1]["x"], 12)
        self.assertIn({"x": 2, "y": 0}, result["curve"])

    def test_loss_variants_and_validation(self):
        for kind, loss, gradient in [("mse", 9, 6), ("mae", 3, 1), ("huber", 2.5, 1)]:
            result = self.successful("/api/loss", {"prediction": 3, "target": 0, "kind": kind})
            self.assertEqual(result["loss"], loss)
            self.assertEqual(result["gradient"], gradient)
        result = self.successful("/api/loss", {"prediction": 0, "target": 0, "kind": "mae"})
        self.assertFalse(result["differentiable"])
        for target in [0, 1]:
            for prediction in [-1000, 0, 1000]:
                result = self.successful("/api/loss", {"prediction": prediction, "target": target, "kind": "bce"})
                self.assertTrue(math.isfinite(result["loss"]))
                self.assertTrue(-1 <= result["gradient"] <= 1)
        for extra in [{"kind": "bce", "target": 0.5}, {"kind": "unknown"}, {"delta": 0}, {"delta": -1}, {"delta": float("inf")}]:
            status, _, _ = asyncio.run(asgi_request("POST", "/api/loss", {"prediction": 1, "target": 0, **extra}))
            self.assertEqual(status, 422)

    def test_gradient_matches_finite_difference(self):
        for weight in [-2, 0, 1.9, 3]:
            eps = 1e-5
            high = self.successful("/api/linear", {"weight": weight + eps, "bias": 0.5})["mse"]
            low = self.successful("/api/linear", {"weight": weight - eps, "bias": 0.5})["mse"]
            step = self.successful("/api/gradient", {"weight": weight, "learning_rate": 0.1})
            self.assertAlmostEqual(step["gradient"], (high - low) / (2 * eps), places=7)
            self.assertAlmostEqual(step["weight_after"], weight - 0.1 * step["gradient"])

    def test_gradient_converges(self):
        weight = -2
        previous_loss = math.inf
        for _ in range(30):
            result = self.successful("/api/gradient", {"weight": weight, "learning_rate": 0.1})
            self.assertLessEqual(result["loss_after"], previous_loss)
            self.assertFalse(result["diverged"])
            previous_loss = result["loss_after"]
            weight = result["weight_after"]
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(weight, 2, places=4)

    def test_large_learning_rate_reports_divergence(self):
        result = self.successful("/api/gradient", {"weight": -2, "learning_rate": 0.6})
        self.assertTrue(result["diverged"])
        self.assertGreater(result["loss_after"], result["loss_before"])
        self.assertFalse(result["converged"])

    def test_invalid_requests_return_422(self):
        cases = [
            ("/api/data", {"pixels": [], "normalize": True}),
            ("/api/data", {"pixels": [[1], [1, 2]], "normalize": True}),
            ("/api/data", {"pixels": [[]], "normalize": True}),
            ("/api/data", {"pixels": [[256]], "normalize": True}),
            ("/api/data", {"pixels": [[1] * 33], "normalize": True}),
            ("/api/data", {"pixels": [[float("nan")]], "normalize": True}),
            ("/api/linear", {"weight": True, "bias": 0}),
            ("/api/linear", {"weight": "2", "bias": 0}),
            ("/api/linear", {"weight": 1, "bias": 0, "extra": 2}),
            ("/api/neuron", {"inputs": [1], "weights": [1, 1], "bias": 0, "activation": "relu"}),
            ("/api/neuron", {"inputs": [1, 1], "weights": [1, 1], "bias": 0, "activation": "unknown"}),
            ("/api/loss", {"prediction": float("inf"), "target": 0}),
            ("/api/gradient", {"weight": 1_000_001, "learning_rate": 0.1}),
            ("/api/gradient", {"weight": 0, "learning_rate": 0}),
            ("/api/gradient", {"weight": 0, "learning_rate": 1.1}),
        ]
        for path, body in cases:
            with self.subTest(path=path, body=body):
                status, _, _ = self.call(path, body)
                self.assertEqual(status, 422)

    def test_localhost_cors(self):
        status, _, headers = asyncio.run(asgi_request("GET", "/api/health", headers=[(b"origin", b"http://localhost:5173")]))
        self.assertEqual(status, 200)
        self.assertEqual(headers[b"access-control-allow-origin"], b"http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
