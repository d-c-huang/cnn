"""Numerical and routing checks for the second learning phase."""

import asyncio
import copy
import math
import unittest

import numpy as np

from backend.phase2 import (
    GraphRequest,
    InitRequest,
    Parameters,
    StepRequest,
    computation_graph,
    forward,
    initialize_xor,
    loss_and_gradients,
    parameter_arrays,
    serialize_arrays,
    step_xor,
)
from backend.tests.test_experiments import asgi_request


class PhaseTwoTests(unittest.TestCase):
    def call(self, path, body):
        return asyncio.run(asgi_request("POST", path, body))

    def successful(self, path, body):
        status, result, _ = self.call(path, body)
        self.assertEqual(status, 200, result)
        return result

    def graph_body(self):
        return {"inputs": [0.6, 0.9], "weights": [0.8, -0.4],
                "bias": 0.2, "target": 0.8, "learning_rate": 0.1}

    def initial(self, seed=7):
        return self.successful("/api/xor/init", {"seed": seed})

    def test_graph_forward_values_and_topological_order(self):
        result = self.successful("/api/graph", self.graph_body())
        nodes = {node["id"]: node for node in result["nodes"]}
        self.assertEqual(result["forward_order"], ["x1", "w1", "p1", "x2", "w2", "p2", "s", "b", "z", "a", "target", "e", "loss"])
        self.assertEqual(result["backward_order"], list(reversed(result["forward_order"])))
        self.assertAlmostEqual(nodes["p1"]["value"], 0.48)
        self.assertAlmostEqual(nodes["p2"]["value"], -0.36)
        self.assertAlmostEqual(nodes["z"]["value"], 0.32)
        expected_output = math.tanh(0.32)
        self.assertAlmostEqual(result["output"], expected_output)
        self.assertAlmostEqual(result["loss"], 0.5 * (expected_output - 0.8) ** 2)
        self.assertEqual(nodes["loss"]["gradient"], 1)
        self.assertEqual({node["id"] for node in result["nodes"] if node["trainable"]}, {"w1", "w2", "b"})
        positions = {name: i for i, name in enumerate(result["forward_order"])}
        for edge in result["edges"]:
            self.assertLess(positions[edge["source"]], positions[edge["target"]])

    def test_graph_all_leaf_gradients_match_finite_differences(self):
        body = self.graph_body()
        result = computation_graph(GraphRequest(**body))
        nodes = {node["id"]: node for node in result["nodes"]}
        eps = 1e-6
        fields = [("x1", "inputs", 0), ("x2", "inputs", 1),
                  ("w1", "weights", 0), ("w2", "weights", 1),
                  ("b", "bias", None), ("target", "target", None)]
        for name, field, index in fields:
            with self.subTest(node=name):
                high, low = copy.deepcopy(body), copy.deepcopy(body)
                if index is None:
                    high[field] += eps
                    low[field] -= eps
                else:
                    high[field][index] += eps
                    low[field][index] -= eps
                high_loss = computation_graph(GraphRequest(**high))["loss"]
                low_loss = computation_graph(GraphRequest(**low))["loss"]
                self.assertAlmostEqual(nodes[name]["gradient"], (high_loss - low_loss) / (2 * eps), places=8)

    def test_graph_edges_apply_local_chain_rule(self):
        result = computation_graph(GraphRequest(**self.graph_body()))
        nodes = {node["id"]: node for node in result["nodes"]}
        expected_derivatives = {
            ("x1", "p1"): 0.8, ("w1", "p1"): 0.6,
            ("x2", "p2"): -0.4, ("w2", "p2"): 0.9,
            ("p1", "s"): 1, ("p2", "s"): 1,
            ("s", "z"): 1, ("b", "z"): 1,
            ("z", "a"): 1 - nodes["a"]["value"] ** 2,
            ("a", "e"): 1, ("target", "e"): -1,
            ("e", "loss"): nodes["e"]["value"],
        }
        self.assertEqual(len(result["edges"]), len(expected_derivatives))
        for edge in result["edges"]:
            with self.subTest(edge=(edge["source"], edge["target"])):
                self.assertAlmostEqual(edge["local_derivative"], expected_derivatives[(edge["source"], edge["target"])])
                self.assertAlmostEqual(edge["contribution"], nodes[edge["target"]]["gradient"] * edge["local_derivative"])
        # A node's gradient is the sum of all contributions from its consumers.
        for name, node in nodes.items():
            if name != "loss":
                self.assertAlmostEqual(node["gradient"], sum(edge["contribution"] for edge in result["edges"] if edge["source"] == name))

    def test_graph_update_recomputes_prediction_and_loss(self):
        body = self.graph_body()
        result = self.successful("/api/graph", body)
        for update in result["updates"]:
            self.assertAlmostEqual(update["after"], update["before"] - body["learning_rate"] * update["gradient"])
        after = self.successful("/api/graph", result["next"])
        self.assertAlmostEqual(result["output_after"], after["output"])
        self.assertAlmostEqual(result["loss_after"], after["loss"])
        self.assertLess(result["loss_after"], result["loss"])

    def test_xor_seed_reproducibility_and_parameter_shapes(self):
        one, two, other = self.initial(), self.initial(), self.initial(seed=9)
        self.assertEqual(one, two)
        self.assertNotEqual(one["parameters"], other["parameters"])
        arrays = parameter_arrays(one["parameters"])
        self.assertEqual({name: value.shape for name, value in arrays.items()}, {"w1": (2, 4), "b1": (4,), "w2": (4,), "b2": ()})
        self.assertEqual(sum(value.size for value in arrays.values()), 17)
        self.assertEqual(one["iteration"], 0)

    def test_xor_all_seventeen_gradients_match_finite_differences(self):
        initial = initialize_xor(InitRequest(seed=7))
        parameters = parameter_arrays(initial["parameters"])
        _, gradients, _ = loss_and_gradients(parameters)
        eps = 1e-6
        checked = 0
        for name, value in parameters.items():
            for index in np.ndindex(value.shape):
                with self.subTest(parameter=name, index=index):
                    high = {key: array.copy() for key, array in parameters.items()}
                    low = {key: array.copy() for key, array in parameters.items()}
                    high[name][index] += eps
                    low[name][index] -= eps
                    high_loss = loss_and_gradients(high)[0]
                    low_loss = loss_and_gradients(low)[0]
                    self.assertAlmostEqual(float(gradients[name][index]), (high_loss - low_loss) / (2 * eps), places=8)
                    checked += 1
        self.assertEqual(checked, 17)

    def test_xor_sample_gradient_matches_its_sample_loss(self):
        initial = self.initial()
        parameters = parameter_arrays(initial["parameters"])
        results = []
        for sample in range(4):
            result = self.successful("/api/xor/inspect", {"parameters": initial["parameters"], "iteration": 0, "selected_sample": sample})
            self.assertEqual(result["selected"]["index"], sample)
            self.assertEqual(result["selected"]["inputs"], result["samples"][sample]["inputs"])
            results.append(parameter_arrays(result["sample_gradients"]))
            eps = 1e-6
            high, low = {key: value.copy() for key, value in parameters.items()}, {key: value.copy() for key, value in parameters.items()}
            high["w1"][0, 2] += eps
            low["w1"][0, 2] -= eps
            high_result = self.successful("/api/xor/inspect", {"parameters": serialize_arrays(high), "iteration": 0, "selected_sample": sample})
            low_result = self.successful("/api/xor/inspect", {"parameters": serialize_arrays(low), "iteration": 0, "selected_sample": sample})
            estimated = (high_result["selected"]["loss"] - low_result["selected"]["loss"]) / (2 * eps)
            self.assertAlmostEqual(result["sample_gradients"]["w1"][0][2], estimated, places=8)
        full_gradients = parameter_arrays(initial["gradients"])
        for name in parameters:
            np.testing.assert_allclose(np.mean([result[name] for result in results], axis=0), full_gradients[name], atol=1e-12)

    def test_xor_one_step_and_history_match_actual_updated_parameters(self):
        initial = self.initial()
        request = {"parameters": initial["parameters"], "iteration": 0, "learning_rate": 1, "steps": 1, "selected_sample": 2}
        result = self.successful("/api/xor/step", request)
        before, gradient, after = parameter_arrays(initial["parameters"]), parameter_arrays(initial["gradients"]), parameter_arrays(result["snapshot"]["parameters"])
        for name in before:
            np.testing.assert_allclose(after[name], before[name] - gradient[name], atol=1e-14)
        self.assertEqual(result["snapshot"]["selected"]["index"], 2)
        self.assertEqual(result["snapshot"]["iteration"], 1)
        self.assertEqual(result["history"], [{"step": 1, "loss": result["snapshot"]["loss"], "accuracy": result["snapshot"]["accuracy"]}])
        self.assertEqual(len(result["updates"]), 17)
        for update in result["updates"]:
            self.assertAlmostEqual(update["after"], update["before"] - update["gradient"])

    def test_xor_batched_steps_equal_repeated_single_steps(self):
        initial = self.initial()
        repeated, history = initial, []
        for _ in range(4):
            result = self.successful("/api/xor/step", {"parameters": repeated["parameters"], "iteration": repeated["iteration"], "learning_rate": 0.7, "steps": 1, "selected_sample": 1})
            repeated = result["snapshot"]
            history.extend(result["history"])
        batched = self.successful("/api/xor/step", {"parameters": initial["parameters"], "iteration": 0, "learning_rate": 0.7, "steps": 4, "selected_sample": 1})
        self.assertEqual(batched["snapshot"], repeated)
        self.assertEqual(batched["history"], history)
        self.assertEqual(batched["updates"], result["updates"])

    def test_xor_default_seed_really_learns_all_four_samples(self):
        state = initialize_xor(InitRequest(seed=7))
        initial_loss = state["loss"]
        for _ in range(5):
            result = step_xor(StepRequest(parameters=Parameters(**state["parameters"]), iteration=state["iteration"], learning_rate=1, steps=100, selected_sample=0))
            state = result["snapshot"]
        self.assertEqual(state["iteration"], 500)
        self.assertEqual(state["accuracy"], 1)
        self.assertLess(state["loss"], 0.01)
        self.assertLess(state["loss"], initial_loss / 50)
        self.assertEqual([sample["prediction"] for sample in state["samples"]], [0, 1, 1, 0])

    def test_xor_grid_and_selected_values_use_current_parameters(self):
        state = self.initial()
        grid = state["grid"]
        self.assertEqual(grid["size"], 25)
        self.assertEqual(len(grid["probabilities"]), 625)
        self.assertTrue(all(0 <= probability <= 1 for probability in grid["probabilities"]))
        arrays = parameter_arrays(state["parameters"])
        for index, point in [(0, [-0.25, -0.25]), (24, [1.25, -0.25]), (600, [-0.25, 1.25]), (624, [1.25, 1.25])]:
            actual = forward(arrays, np.asarray([point], dtype=np.float64))["output"][0]
            self.assertAlmostEqual(grid["probabilities"][index], actual)
        self.assertAlmostEqual(state["selected"]["output"], state["samples"][0]["probability"])
        self.assertAlmostEqual(state["selected"]["z2"], float(np.dot(state["selected"]["a1"], arrays["w2"]) + arrays["b2"]))

    def test_xor_stable_loss_at_saturated_logits(self):
        parameters = {"w1": [[1000] * 4, [1000] * 4], "b1": [1000] * 4, "w2": [1000] * 4, "b2": 1000}
        result = self.successful("/api/xor/inspect", {"parameters": parameters, "iteration": 0, "selected_sample": 0})
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertAlmostEqual(result["loss"], 2500)
        self.assertTrue(all(math.isfinite(value) for value in result["gradients"]["w2"]))

    def test_invalid_phase_two_requests_return_422(self):
        initial = self.initial()
        graph = self.graph_body()
        cases = [
            ("/api/graph", {**graph, "inputs": [1]}),
            ("/api/graph", {**graph, "weights": [1, float("nan")]}),
            ("/api/graph", {**graph, "target": 11}),
            ("/api/graph", {**graph, "learning_rate": 0}),
            ("/api/graph", {**graph, "learning_rate": 1.1}),
            ("/api/xor/init", {"seed": True}),
            ("/api/xor/init", {"seed": "7"}),
            ("/api/xor/init", {"seed": -1}),
            ("/api/xor/init", {"seed": 10000}),
        ]
        inspect = {"parameters": initial["parameters"], "iteration": 0, "selected_sample": 0}
        for name, value in [("w1", [[1, 2, 3, 4]]), ("b1", [1, 2, 3]), ("w2", [1, 2, 3, 4, 5]), ("b2", float("inf")), ("b2", 1001), ("b2", True)]:
            cases.append(("/api/xor/inspect", {**inspect, "parameters": {**initial["parameters"], name: value}}))
        step = {**inspect, "learning_rate": 1, "steps": 1}
        for patch in [{"selected_sample": 4}, {"steps": 0}, {"steps": 101}, {"steps": 1.5}, {"learning_rate": 3.1}, {"learning_rate": float("nan")}, {"iteration": 20_000}, {"iteration": -1}, {"unexpected": 1}]:
            cases.append(("/api/xor/step", {**step, **patch}))
        for path, body in cases:
            with self.subTest(path=path, body=body):
                status, _, _ = self.call(path, body)
                self.assertEqual(status, 422)


if __name__ == "__main__":
    unittest.main()
