import asyncio
import math
import unittest

import numpy as np
from pydantic import ValidationError

from backend.language import (
    CorpusRequest, GenerateRequest, TrainRequest, dataset, generate, prepare, train,
)
from backend.tests.test_experiments import asgi_request


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.data = prepare(CorpusRequest())

    def test_split_token_alignment_and_unknown_characters(self):
        corpus = "春猫" * 20 + "雪" * 10
        data = prepare(CorpusRequest(corpus=corpus))
        self.assertNotIn("雪", data["vocab"])
        self.assertEqual(data["unknown_validation"], 10)
        self.assertEqual(data["train_tokens"], 40)
        self.assertEqual(data["validation_tokens"], 10)
        self.assertEqual(data["train_text"] + data["validation_text"], corpus)
        for index, pair in enumerate(data["pairs"]):
            self.assertEqual(data["vocab"][pair["input"]], corpus[index])
            self.assertEqual(data["vocab"][pair["target"]], corpus[index + 1])
        self.assertAlmostEqual(data["history"][0]["train_loss"], math.log(len(data["vocab"])))

    def test_updates_match_pytorch_cross_entropy(self):
        import torch

        _, parts, _ = dataset(self.data["corpus"])
        weights = torch.tensor(self.data["weights"], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([weights], lr=5)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(weights[parts[0][:-1]], torch.tensor(parts[0][1:]))
            loss.backward()
            optimizer.step()
        result = train(TrainRequest(weights=self.data["weights"], steps=3))
        np.testing.assert_allclose(result["weights"], weights.detach().numpy(), atol=1e-12)
        self.assertLess(result["history"][-1]["train_loss"], self.data["history"][0]["train_loss"])
        self.assertAlmostEqual(result["history"][-1]["perplexity"], math.exp(result["history"][-1]["validation_loss"]))

    def test_validation_does_not_change_updates(self):
        outputs = []
        for suffix in ["春" * 10, "猫" * 10]:
            corpus = "春猫" * 20 + suffix
            data = prepare(CorpusRequest(corpus=corpus))
            outputs.append(train(TrainRequest(corpus=corpus, weights=data["weights"], steps=20)))
        np.testing.assert_array_equal(outputs[0]["weights"], outputs[1]["weights"])
        self.assertNotEqual(outputs[0]["history"][-1]["validation_loss"], outputs[1]["history"][-1]["validation_loss"])

    def test_training_can_resume_exactly(self):
        first = train(TrainRequest(weights=self.data["weights"], steps=100))
        second = train(TrainRequest(weights=first["weights"], step=100, steps=100))
        combined = train(TrainRequest(weights=self.data["weights"], steps=200))
        self.assertEqual(second["step"], 200)
        self.assertEqual(second["weights"], combined["weights"])
        self.assertEqual(second["history"][-1], combined["history"][-1])

    def test_seed_context_temperature_and_no_weight_mutation(self):
        trained = train(TrainRequest(weights=self.data["weights"], steps=100))
        request = GenerateRequest(weights=trained["weights"], prompt="小猫", seed=42)
        output = generate(request)
        self.assertEqual(output, generate(request))
        self.assertEqual(output, generate(request.model_copy(update={"prompt": "看猫"})))
        self.assertEqual(len(output["continuation"].replace("<UNK>", "?")), 60)
        cold = generate(request.model_copy(update={"temperature": 0.1}))
        hot = generate(request.model_copy(update={"temperature": 2}))
        self.assertGreater(cold["distribution"][0]["probability"], hot["distribution"][0]["probability"])
        self.assertEqual(request.weights, trained["weights"])

    def test_limits_and_invalid_shapes(self):
        for patch in [{"steps": 201}, {"learning_rate": float("nan")}, {"weights": [[0.0]]}, {"step": 4999, "steps": 2}, {"corpus": "a" * 50}]:
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                TrainRequest(**({"weights": self.data["weights"]} | patch))

    def test_routes_and_unknown_prompt_error(self):
        status, data, _ = asyncio.run(asgi_request("POST", "/api/language/prepare", {}))
        self.assertEqual(status, 200)
        status, result, _ = asyncio.run(asgi_request("POST", "/api/language/train", {"weights": data["weights"], "steps": 3}))
        self.assertEqual(status, 200)
        self.assertEqual(result["step"], 3)
        status, _, _ = asyncio.run(asgi_request("POST", "/api/language/generate", {"weights": result["weights"], "prompt": "不存在的字X"}))
        self.assertEqual(status, 422)


if __name__ == "__main__":
    unittest.main()
