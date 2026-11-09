import asyncio
import math
import unittest

import numpy as np

from backend.optimizers import ComparisonRequest, compare_optimizers, objective_value
from backend.tests.test_experiments import asgi_request


class OptimizerTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        for objective in ["quadratic", "valley"]:
            point = np.array([-1.3, 0.7])
            _, gradient = objective_value(point, objective)
            for axis in [0, 1]:
                delta = np.zeros(2)
                delta[axis] = 1e-5
                numerical = (objective_value(point + delta, objective)[0] - objective_value(point - delta, objective)[0]) / 2e-5
                self.assertAlmostEqual(gradient[axis], numerical, places=7)

    def test_known_updates_and_transient_threshold(self):
        result = compare_optimizers(ComparisonRequest(steps=4))
        runs = {run["optimizer"]: run for run in result["runs"]}
        first = runs["sgd"]["history"][1]
        self.assertAlmostEqual(first["position"][0], -0.4)
        self.assertAlmostEqual(first["loss"], 11.52)
        self.assertEqual(first["gradient"], [-16, 0])
        self.assertEqual(first["update"], [1.6, 0])
        for name in ["adagrad", "adam"]:
            self.assertAlmostEqual(runs[name]["history"][1]["position"][0], -1.9)
        self.assertAlmostEqual(runs["rmsprop"]["history"][1]["position"][0], -2 + 0.1 / math.sqrt(0.1))
        momentum = runs["momentum"]
        self.assertEqual(momentum["first_hit"], 2)
        self.assertLess(momentum["history"][2]["loss"], 1e-6)
        self.assertGreater(momentum["history"][3]["loss"], 1)
        self.assertEqual(len(momentum["history"]), 5, "Keep computing after a transient threshold crossing")

    def test_matches_pytorch_optimizers(self):
        import torch

        for objective in ["quadratic", "valley"]:
            result = compare_optimizers(ComparisonRequest(objective=objective, steps=12, learning_rate=0.03))
            for run in result["runs"]:
                with self.subTest(objective=objective, optimizer=run["optimizer"]):
                    parameter = torch.tensor(run["history"][0]["position"], dtype=torch.float64, requires_grad=True)
                    factories = {
                        "sgd": lambda: torch.optim.SGD([parameter], lr=0.03),
                        "momentum": lambda: torch.optim.SGD([parameter], lr=0.03, momentum=0.9),
                        "adagrad": lambda: torch.optim.Adagrad([parameter], lr=0.03, eps=1e-8),
                        "rmsprop": lambda: torch.optim.RMSprop([parameter], lr=0.03, alpha=0.9, eps=1e-8),
                        "adam": lambda: torch.optim.Adam([parameter], lr=0.03, betas=(0.9, 0.999), eps=1e-8),
                    }
                    optimizer = factories[run["optimizer"]]()
                    for row in run["history"][1:]:
                        optimizer.zero_grad()
                        loss = 2 * (parameter[0] - 2) ** 2 if objective == "quadratic" else 0.5 * ((parameter[0] - 2) ** 2 + 20 * (parameter[1] + 1) ** 2)
                        loss.backward()
                        np.testing.assert_allclose(row["gradient"], parameter.grad.numpy(), atol=1e-12)
                        optimizer.step()
                        np.testing.assert_allclose(row["position"], parameter.detach().numpy(), atol=1e-12)
                        self.assertAlmostEqual(row["loss"], objective_value(parameter.detach().numpy(), objective)[0], places=10)

    def test_limits_zero_gradient_and_independent_rates(self):
        limited = compare_optimizers(ComparisonRequest(objective="valley", learning_rate=1.0, steps=200))
        self.assertEqual(limited["runs"][0]["status"], "limit")
        self.assertLess(len(limited["runs"][0]["history"]), 201)
        for run in limited["runs"]:
            self.assertTrue(all(math.isfinite(row["loss"]) for row in run["history"]))
        zero = compare_optimizers(ComparisonRequest(objective="valley", start=(2.0, -1.0), steps=10))
        for run in zero["runs"]:
            self.assertEqual(run["first_hit"], 0)
            self.assertTrue(all(row["loss"] == 0 for row in run["history"]))
        request = ComparisonRequest(steps=10, rates={"adam": 0.01})
        result = compare_optimizers(request)
        self.assertEqual(result, compare_optimizers(request), "Every run starts with fresh state")
        self.assertEqual([run["learning_rate"] for run in result["runs"]], [0.1, 0.1, 0.1, 0.1, 0.01])

    def test_api_validation_and_snapshot(self):
        status, body, _ = asyncio.run(asgi_request("POST", "/api/optimizers/compare", {"steps": 3}))
        self.assertEqual(status, 200)
        self.assertEqual(body["config"]["steps"], 3)
        self.assertEqual(len(body["runs"]), 5)
        for payload in [{"objective": "unknown"}, {"steps": 201}, {"steps": True}, {"start": [1]}, {"learning_rate": 0}, {"rates": {"adam": float("nan")}}, {"rates": {"unknown": 0.1}}, {"start": [7, 0]}, {"momentum": 0.5}]:
            status, _, _ = asyncio.run(asgi_request("POST", "/api/optimizers/compare", payload))
            self.assertEqual(status, 422, payload)


if __name__ == "__main__":
    unittest.main()
