"""Inference checks use synthetic checkpoint/image fixtures in temporary folders.

These tests do not download MNIST, mutate actual run records or make recognition
accuracy claims. Real trained-weight verification is performed separately.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np

from backend.inference import InferenceService, PredictRequest, preprocess_pixels, translate_without_wrap
from backend.mnist_engine import build_model, torch_runtime
from backend.tests.test_experiments import asgi_request


async def query_request(path):
    """The older shared client has no query-string support; use a tiny ASGI scope."""
    from backend.app import app
    route, _, query = path.partition("?")
    messages = []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": route, "raw_path": route.encode(),
             "root_path": "", "query_string": query.encode(), "headers": [],
             "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        messages.append(message)
    await app(scope, receive, send)
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = torch_runtime()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.service = InferenceService(self.root)
        self.image = np.zeros((28, 28), dtype=np.uint8)
        self.image[4:24, 11:16] = 190
        self.image[5:9, 8:12] = 95

    def create_run(self, architecture="mlp", model_id=None, created="2026-10-25T00:00:00+00:00", status="completed"):
        model_id = model_id or ("a" * 32 if architecture == "mlp" else "b" * 32)
        self.torch.manual_seed(41 if architecture == "mlp" else 42)
        model = build_model(architecture).eval()
        checkpoint = {"state_dict": model.state_dict(), "config": {"model": architecture},
                      "epoch": 2, "validation_loss": 0.25, "normalization": {"mean": 0.1307, "std": 0.3081}}
        directory = self.root / "runs" / "mnist" / model_id
        directory.mkdir(parents=True, exist_ok=True)
        self.torch.save(checkpoint, directory / "best.pt")
        record = {"id": model_id, "created_at": created, "status": status,
                  "config": {"model": architecture}, "best_epoch": 2,
                  "best_validation_loss": 0.25, "evaluation": {"accuracy": 0.8}}
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
        return model_id, model, checkpoint, directory

    def request(self, ids, pixels=None, mode="mnist"):
        return PredictRequest(model_ids=ids, pixels=(self.image if pixels is None else pixels).tolist(), mode=mode)

    def test_mnist_mode_matches_direct_checkpoint_normalization(self):
        for architecture in ("mlp", "cnn"):
            with self.subTest(model=architecture):
                model_id, model, _, _ = self.create_run(architecture)
                result = self.service.predict(self.request([model_id]))
                with self.torch.inference_mode():
                    inputs = self.torch.from_numpy(self.image.astype(np.float32)).reshape(1, 1, 28, 28)
                    expected = model(inputs.div(255.0).sub(0.1307).div(0.3081)).softmax(1)[0].numpy()
                np.testing.assert_allclose(result["predictions"][0]["probabilities"], expected, rtol=0, atol=1e-7)
                np.testing.assert_array_equal(result["preprocessing"]["processed_pixels"], self.image)
                self.assertEqual(result["preprocessing"]["shift"], [0, 0])
                self.assertIsNone(result["preprocessing"]["bounding_box"])
                self.assertAlmostEqual(sum(result["predictions"][0]["probabilities"]), 1.0, places=6)
                self.assertEqual(result["predictions"][0]["predicted_label"], int(expected.argmax()))
                self.assertEqual(result["predictions"][0]["top_k"][0]["label"], int(expected.argmax()))
                self.assertGreater(result["total_latency_ms"], 0)

    def test_two_architectures_share_the_same_processed_image(self):
        mlp_id, _, _, _ = self.create_run("mlp")
        cnn_id, _, _, _ = self.create_run("cnn")
        result = self.service.predict(self.request([mlp_id, cnn_id], np.repeat(np.repeat(self.image, 5, axis=0), 5, axis=1), "canvas"))
        self.assertEqual([row["architecture"] for row in result["predictions"]], ["mlp", "cnn"])
        self.assertEqual(np.asarray(result["preprocessing"]["processed_pixels"]).shape, (28, 28))
        for row in result["predictions"]:
            self.assertEqual(len(row["probabilities"]), 10)
            self.assertEqual(len(row["top_k"]), 3)
            self.assertTrue(all(0 <= value <= 1 for value in row["probabilities"]))

    def test_faint_mnist_pixels_preserve_direct_forward_results_and_zero_is_blank(self):
        faint = np.zeros((28, 28), dtype=np.float32)
        faint[4:24, 11:16] = 8
        faint[5:9, 8:12] = 0.5
        for architecture in ("mlp", "cnn"):
            with self.subTest(model=architecture):
                model_id, model, _, _ = self.create_run(architecture)
                result = self.service.predict(self.request([model_id], faint))
                with self.torch.inference_mode():
                    inputs = self.torch.from_numpy(faint).reshape(1, 1, 28, 28)
                    expected = model(inputs.div(255.0).sub(0.1307).div(0.3081)).softmax(1)[0].numpy()
                np.testing.assert_allclose(result["predictions"][0]["probabilities"], expected, rtol=0, atol=1e-7)
                np.testing.assert_array_equal(result["preprocessing"]["processed_pixels"], faint)
                self.assertEqual(result["preprocessing"]["ink_pixels"], int(np.count_nonzero(faint)))
                with self.assertRaises(HTTPException) as failure:
                    self.service.predict(self.request([model_id], np.zeros((28, 28))))
                self.assertEqual(failure.exception.status_code, 422)

    def test_loading_inference_models_does_not_consume_training_rng(self):
        mlp_id, _, _, _ = self.create_run("mlp")
        cnn_id, _, _, _ = self.create_run("cnn")
        before = self.torch.get_rng_state().clone()
        self.service.predict(self.request([mlp_id, cnn_id]))
        self.assertTrue(self.torch.equal(before, self.torch.get_rng_state()))

    def test_unavailable_runtime_is_a_clear_service_error(self):
        with patch("backend.inference.runtime_status", return_value={"available": False}):
            with self.assertRaises(HTTPException) as failure:
                self.service.predict(self.request(["f" * 32]))
        self.assertEqual(failure.exception.status_code, 503)

    def test_canvas_crop_aspect_ratio_grayscale_and_center_of_mass(self):
        image = np.zeros((100, 200), dtype=np.float64)
        image[10:70, 120:135] = 150
        image[10:30, 135:150] = 220
        result, metadata = preprocess_pixels(image, "canvas")
        self.assertEqual(metadata["original_shape"], [100, 200])
        self.assertEqual(metadata["bounding_box"], [120, 10, 150, 70])
        self.assertEqual(result.shape, (28, 28))
        ys, xs = np.nonzero(result > 16)
        self.assertLessEqual(xs.max() - xs.min() + 1, 12)
        self.assertLessEqual(ys.max() - ys.min() + 1, 22)
        yy, xx = np.indices(result.shape)
        self.assertLess(abs((xx * result).sum() / result.sum() - 13.5), 0.6)
        self.assertLess(abs((yy * result).sum() / result.sum() - 13.5), 0.6)
        self.assertGreater(len(np.unique(result)), 3)  # Gray levels were not binarized.
        self.assertLess(result.max(), 255)

    def test_translation_never_wraps_pixels_around_edges(self):
        image = np.zeros((28, 28), dtype=np.uint8)
        image[0, 0], image[27, 27] = 125, 255
        shifted = translate_without_wrap(image, 2, 3)
        self.assertEqual(shifted[3, 2], 125)
        self.assertEqual(int(shifted.sum()), 125)
        self.assertEqual(int(translate_without_wrap(image, 40, 40).sum()), 0)

    def test_blank_canvas_is_rejected_without_fabricating_a_prediction(self):
        model_id, _, _, _ = self.create_run()
        for pixels in (np.zeros((28, 28)), np.full((28, 28), 10)):
            with self.subTest(max=pixels.max()):
                with self.assertRaises(HTTPException) as failure:
                    self.service.predict(self.request([model_id], pixels, "canvas"))
                self.assertEqual(failure.exception.status_code, 422)
        self.assertFalse(self.service.cache)

    def test_catalog_prefers_completed_latest_runs_not_highest_test_accuracy(self):
        first, _, _, first_directory = self.create_run(model_id="1" * 32)
        second, _, _, second_directory = self.create_run(model_id="2" * 32, created="2026-10-26T00:00:00+00:00")
        self.create_run("cnn", model_id="3" * 32, created="2026-10-27T00:00:00+00:00", status="cancelled")
        record = json.loads((first_directory / "run.json").read_text())
        record["evaluation"]["accuracy"] = 1.0
        (first_directory / "run.json").write_text(json.dumps(record))
        record = json.loads((second_directory / "run.json").read_text())
        record["evaluation"]["accuracy"] = 0.2
        (second_directory / "run.json").write_text(json.dumps(record))
        with patch("backend.inference.torch_runtime", side_effect=AssertionError("catalog must remain lightweight")):
            result = self.service.models()
        self.assertEqual(result["defaults"], {"mlp": second, "cnn": None})
        self.assertEqual({row["id"] for row in result["models"]}, {first, second})

    def test_empty_catalog_and_unknown_id_never_initialize_random_weights(self):
        self.assertEqual(self.service.models()["models"], [])
        with patch("backend.inference.build_model", side_effect=AssertionError("must not initialize")):
            with self.assertRaises(HTTPException) as failure:
                self.service.predict(self.request(["f" * 32]))
        self.assertEqual(failure.exception.status_code, 404)

    def test_cache_reuses_unchanged_weights_and_invalidates_a_changed_hash(self):
        model_id, _, checkpoint, directory = self.create_run()
        first = self.service.predict(self.request([model_id]))
        cached_model = self.service.cache[model_id]["model"]
        self.service.predict(self.request([model_id]))
        self.assertIs(self.service.cache[model_id]["model"], cached_model)
        before = (directory / "best.pt").stat()
        last_bias = checkpoint["state_dict"]["5.bias"]
        checkpoint["state_dict"]["5.bias"] = last_bias + self.torch.arange(10) * 3
        self.torch.save(checkpoint, directory / "best.pt")
        os.utime(directory / "best.pt", ns=(before.st_atime_ns, before.st_mtime_ns))
        after = self.service.predict(self.request([model_id]))
        self.assertIsNot(self.service.cache[model_id]["model"], cached_model)
        self.assertNotEqual(first["predictions"][0]["probabilities"], after["predictions"][0]["probabilities"])

    def write_bundle(self, model_id, checkpoint, accuracy=0.8):
        directory = self.root / "models"
        directory.mkdir(exist_ok=True)
        path = directory / "mlp-best.pt"
        self.torch.save(checkpoint, path)
        row = {"id": model_id, "run_id": model_id, "architecture": "mlp", "checkpoint_epoch": 2,
               "validation_loss": 0.25, "test_accuracy": accuracy, "created_at": "2026-10-25T00:00:00+00:00",
               "filename": "mlp-best.pt", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (directory / "manifest.json").write_text(json.dumps({"format": "zhiwei-mnist-models-v1", "models": [row]}))
        return row, directory

    def test_bundle_is_usable_without_any_training_run_or_mnist_cache(self):
        model_id, _, checkpoint, directory = self.create_run()
        row, _ = self.write_bundle(model_id, checkpoint)
        # Remove the source run so only the explicitly exported bundle exists.
        (directory / "run.json").unlink()
        catalog = self.service.models()
        self.assertEqual(catalog["models"][0]["source"], "bundle")
        self.assertEqual(catalog["models"][0]["sha256"], row["sha256"])
        self.assertEqual(self.service.predict(self.request([model_id]))["predictions"][0]["model_id"], model_id)

    def test_run_overrides_a_duplicate_bundle_and_bundle_checksum_is_enforced(self):
        model_id, _, checkpoint, run_directory = self.create_run()
        row, bundle_directory = self.write_bundle(model_id, checkpoint)
        self.assertEqual(self.service.models()["models"][0]["source"], "run")
        (run_directory / "run.json").unlink()
        self.service.predict(self.request([model_id]))
        row["sha256"] = "0" * 64
        (bundle_directory / "manifest.json").write_text(json.dumps({"format": "zhiwei-mnist-models-v1", "models": [row]}))
        with self.assertRaises(HTTPException) as failure:
            self.service.predict(self.request([model_id]))
        self.assertEqual(failure.exception.status_code, 503)

    def test_invalid_checkpoints_fail_explicitly(self):
        mutations = [
            lambda checkpoint: checkpoint.update(normalization={"mean": 0, "std": 0}),
            lambda checkpoint: checkpoint.update(normalization={"mean": float("nan"), "std": 1}),
            lambda checkpoint: checkpoint.update(config={"model": "cnn"}),
            lambda checkpoint: checkpoint.update(epoch=99),
            lambda checkpoint: checkpoint.update(validation_loss=0.3),
            lambda checkpoint: checkpoint.update(state_dict={}),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                model_id, _, checkpoint, directory = self.create_run(model_id=f"{index:032x}")
                mutation(checkpoint)
                self.torch.save(checkpoint, directory / "best.pt")
                with self.assertRaises(HTTPException) as failure:
                    self.service.predict(self.request([model_id]))
                self.assertEqual(failure.exception.status_code, 503)
                self.assertNotIn(str(self.root), failure.exception.detail)

    def test_model_normalization_must_match_when_comparing_models(self):
        mlp_id, _, _, _ = self.create_run("mlp")
        cnn_id, _, checkpoint, directory = self.create_run("cnn")
        checkpoint["normalization"]["std"] = 1.0
        self.torch.save(checkpoint, directory / "best.pt")
        with self.assertRaises(HTTPException) as failure:
            self.service.predict(self.request([mlp_id, cnn_id]))
        self.assertEqual(failure.exception.status_code, 503)

    def test_bundle_filenames_cannot_escape_the_model_directory(self):
        model_id, _, checkpoint, directory = self.create_run()
        row, bundle_directory = self.write_bundle(model_id, checkpoint)
        (directory / "run.json").unlink()
        row["filename"] = "../runs/mnist/" + model_id + "/best.pt"
        (bundle_directory / "manifest.json").write_text(json.dumps({"format": "zhiwei-mnist-models-v1", "models": [row]}))
        self.assertEqual(self.service.models()["models"], [])

    def test_offline_examples_preserve_pixels_index_and_source(self):
        images = np.stack([self.image, np.flip(self.image, axis=1), self.image // 2])
        raw = self.root / "data" / "MNIST" / "raw"
        raw.mkdir(parents=True)
        (raw / "t10k-images-idx3-ubyte").write_bytes(struct.pack(">IIII", 2051, 3, 28, 28) + images.tobytes())
        (raw / "t10k-labels-idx1-ubyte").write_bytes(struct.pack(">II", 2049, 3) + bytes([7, 2, 7]))
        result = self.service.example(7)
        self.assertEqual(result, {"index": 0, "label": 7, "source": "mnist-test", "pixels": self.image.tolist()})
        self.assertEqual(self.service.example(2)["index"], 1)
        (self.root / "models").mkdir()
        bundled = {"index": 123, "label": 7, "source": "mnist-test", "pixels": (self.image // 2).tolist()}
        (self.root / "models" / "examples.json").write_text(json.dumps({"format": "zhiwei-mnist-examples-v1", "samples": [bundled]}))
        self.assertEqual(self.service.example(7), bundled)

    def test_missing_or_corrupt_examples_do_not_download_or_fake_data(self):
        with self.assertRaises(HTTPException) as failure:
            self.service.example(7)
        self.assertEqual(failure.exception.status_code, 404)
        raw = self.root / "data" / "MNIST" / "raw"
        raw.mkdir(parents=True)
        (raw / "t10k-images-idx3-ubyte").write_bytes(b"broken")
        (raw / "t10k-labels-idx1-ubyte").write_bytes(b"broken")
        with self.assertRaises(HTTPException) as failure:
            self.service.example(7)
        self.assertEqual(failure.exception.status_code, 404)

    def test_route_validation_and_http_errors_are_consistent(self):
        model_id, _, _, _ = self.create_run()
        good = {"model_ids": [model_id], "pixels": self.image.tolist(), "mode": "mnist"}
        invalid = [
            {**good, "model_ids": []}, {**good, "model_ids": [model_id, model_id]},
            {**good, "model_ids": [model_id] * 3}, {**good, "model_ids": ["../best.pt"]},
            {**good, "pixels": []}, {**good, "pixels": [[0] * 28] * 7},
            {**good, "pixels": [[0] * 321] * 28}, {**good, "pixels": [[0] * 27] * 28},
            {**good, "pixels": [[-1] * 28] * 28}, {**good, "pixels": [[256] * 28] * 28},
            {**good, "pixels": [[float("nan")] * 28] * 28},
            {**good, "pixels": [[float("inf")] * 28] * 28},
            {**good, "mode": "other"}, {**good, "extra": 1},
        ]
        with patch("backend.inference.service", self.service):
            for body in invalid:
                with self.subTest(fields=list(body)):
                    status, _, _ = asyncio.run(asgi_request("POST", "/api/inference/predict", body))
                    self.assertEqual(status, 422)
            status, result, _ = asyncio.run(asgi_request("POST", "/api/inference/predict", good))
            self.assertEqual(status, 200, result)
            status, _, _ = asyncio.run(asgi_request("POST", "/api/inference/predict", {**good, "model_ids": ["f" * 32]}))
            self.assertEqual(status, 404)
            status, _, _ = asyncio.run(asgi_request("GET", "/api/inference/models"))
            self.assertEqual(status, 200)
            status = asyncio.run(query_request("/api/inference/examples?label=11"))
            self.assertEqual(status, 422)


if __name__ == "__main__":
    unittest.main()
