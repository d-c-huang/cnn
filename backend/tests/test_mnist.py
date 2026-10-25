"""MNIST service tests using tiny synthetic fixtures, never downloaded data.

These tensors exercise control flow, splits and numerical accounting. They are
not MNIST and no accuracy claims are based on them. Real MNIST verification is
performed separately through the running application.
"""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from backend.mnist_engine import (
    MnistService, ACTIVE_STATUSES, allocate_counts, build_model,
    classification_report, split_indices, torch_runtime,
)
from backend.tests.test_experiments import asgi_request


class MnistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = torch_runtime()
        generator = cls.torch.Generator().manual_seed(123)
        cls.fixture = {
            "train_images": cls.torch.randint(0, 256, (200, 28, 28), generator=generator, dtype=cls.torch.uint8),
            "train_labels": cls.torch.arange(200) % 10,
            "test_images": cls.torch.randint(0, 256, (50, 28, 28), generator=generator, dtype=cls.torch.uint8),
            "test_labels": cls.torch.arange(50) % 10,
        }

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.service = MnistService(self.root, dataset_provider=lambda: self.fixture)
        self.service._prepare_worker()
        self.assertEqual(self.service.dataset_state, "ready")

    def config(self, **changes):
        config = {"model": "mlp", "seed": 7, "epochs": 2, "batch_size": 32,
                  "learning_rate": 0.001, "validation_ratio": 0.2, "train_limit": None}
        config.update(changes)
        return config

    def finish(self, service=None, **changes):
        service = service or self.service
        run = service.create_run(self.config(**changes))
        service.wait_for_run(run["id"], timeout=20)
        result = service.get_run(run["id"])
        self.assertEqual(result["status"], "completed", result)
        self.assertIsNone(service.active_run_id)
        return result

    def test_models_have_expected_shapes_and_parameter_counts(self):
        for name, expected_count in [("mlp", 109386), ("cnn", 52138)]:
            with self.subTest(model=name):
                model = build_model(name)
                output = model(self.torch.zeros(3, 1, 28, 28))
                self.assertEqual(tuple(output.shape), (3, 10))
                self.assertTrue(self.torch.isfinite(output).all().item())
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), expected_count)

    def test_stratified_splits_are_reproducible_disjoint_and_exact(self):
        labels = self.fixture["train_labels"].numpy()
        one = split_indices(labels, 7, 0.2, 50)
        two = split_indices(labels, 7, 0.2, 50)
        other = split_indices(labels, 8, 0.2, 50)
        for name in one:
            np.testing.assert_array_equal(one[name], two[name])
        self.assertFalse(np.array_equal(one["validation"], other["validation"]))
        self.assertEqual(len(one["full_train"]), 160)
        self.assertEqual(len(one["validation"]), 40)
        self.assertEqual(len(one["effective_train"]), 50)
        self.assertFalse(set(one["full_train"]) & set(one["validation"]))
        self.assertEqual(set(one["full_train"]) | set(one["validation"]), set(range(200)))
        self.assertLessEqual(set(one["effective_train"]), set(one["full_train"]))
        self.assertEqual(np.bincount(labels[one["validation"]], minlength=10).tolist(), [4] * 10)
        self.assertEqual(np.bincount(labels[one["effective_train"]], minlength=10).tolist(), [5] * 10)
        self.assertEqual(int(allocate_counts(np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]), 6000).sum()), 6000)

    def test_split_preview_uses_raw_pixels_and_correct_source_labels(self):
        result = self.service.split(self.config())
        self.assertEqual((result["train_count"], result["validation_count"], result["test_count"]), (160, 40, 50))
        self.assertEqual(result["effective_train_count"], 160)
        self.assertEqual(len(result["samples"]), 36)
        for sample in result["samples"]:
            source = "test" if sample["split"] == "test" else "train"
            self.assertEqual(sample["label"], self.fixture[f"{source}_labels"][sample["index"]].item())
            self.assertEqual(sample["pixels"], self.fixture[f"{source}_images"][sample["index"]].tolist())
        self.assertEqual(sum(result["train_class_counts"]), 160)
        self.assertEqual(sum(result["validation_class_counts"]), 40)
        self.assertEqual(sum(result["test_class_counts"]), 50)

    def test_training_really_changes_parameters_and_persists_best_checkpoint(self):
        self.torch.manual_seed(7)
        initial = {name: value.detach().clone() for name, value in build_model("mlp").state_dict().items()}
        result = self.finish()
        checkpoint = self.torch.load(self.service.checkpoint_path(result["id"]), weights_only=True)
        self.assertTrue(any(not self.torch.equal(initial[name], value) for name, value in checkpoint["state_dict"].items()))
        self.assertEqual(checkpoint["epoch"], result["best_epoch"])
        best = min(result["history"], key=lambda row: row["validation_loss"])
        self.assertEqual(result["best_epoch"], best["epoch"])
        self.assertAlmostEqual(checkpoint["validation_loss"], best["validation_loss"])
        self.assertEqual(result["processed_samples"], result["train_samples"])
        self.assertEqual(result["train_samples"], 160)
        self.assertEqual(result["validation_samples"], 40)
        self.assertEqual(result["test_samples"], 50)
        self.assertEqual(len(result["history"]), 2)
        for row in result["history"]:
            self.assertEqual(set(row), {"epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "seconds"})
            self.assertTrue(np.isfinite(row["train_loss"]))
            self.assertTrue(np.isfinite(row["validation_loss"]))
        manifest = np.load(self.root / "runs" / "mnist" / result["id"] / "split.npz")
        self.assertFalse(set(manifest["train_indices"]) & set(manifest["validation_indices"]))
        self.assertFalse(any("test" in name for name in manifest.files))

    def test_training_never_feeds_test_images_before_final_evaluation(self):
        fixture = dict(self.fixture)
        fixture["train_images"] = self.torch.zeros_like(fixture["train_images"])
        fixture["test_images"] = self.torch.full_like(fixture["test_images"], 255)
        service = MnistService(self.root, dataset_provider=lambda: fixture)
        service._prepare_worker()
        records = []
        base = build_model("mlp")
        torch = self.torch

        class AuditModel(torch.nn.Sequential):
            def forward(self, images):
                records.append((self.training, float(images.max().item())))
                return super().forward(images)

        with patch("backend.mnist_engine.build_model", return_value=AuditModel(*list(base.children()))):
            self.finish(service, epochs=2)
        # Normalized training images are negative, independent test images positive.
        test_positions = [index for index, (_, value) in enumerate(records) if value > 1]
        self.assertTrue(test_positions)
        self.assertEqual(test_positions, list(range(min(test_positions), len(records))))
        self.assertTrue(all(not records[index][0] for index in test_positions))
        self.assertTrue(all(value < 0 for training, value in records if training))

    def test_epoch_metrics_and_final_confusion_account_for_every_test_sample(self):
        result = self.finish(model="cnn", epochs=1)
        evaluation = result["evaluation"]
        matrix = np.asarray(evaluation["confusion_matrix"])
        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(int(matrix.sum()), 50)
        self.assertAlmostEqual(evaluation["accuracy"], float(np.trace(matrix) / matrix.sum()))
        self.assertEqual(evaluation["error_count"], 50 - int(np.trace(matrix)))
        self.assertEqual(evaluation["checkpoint_epoch"], result["best_epoch"])
        self.assertEqual(sum(row["support"] for row in evaluation["class_metrics"]), 50)
        for label, row in enumerate(evaluation["class_metrics"]):
            self.assertEqual(row["support"], int(matrix[label].sum()))

    def test_error_filters_pagination_and_original_images_match_report(self):
        result = self.finish(epochs=1)
        page = self.service.errors(result["id"], offset=0, limit=3)
        self.assertEqual(page["total"], result["evaluation"]["error_count"])
        self.assertEqual(len(page["items"]), 3)
        next_page = self.service.errors(result["id"], offset=3, limit=3)
        self.assertFalse({row["index"] for row in page["items"]} & {row["index"] for row in next_page["items"]})
        for item in page["items"]:
            self.assertNotEqual(item["label"], item["prediction"])
            self.assertEqual(item["pixels"], self.fixture["test_images"][item["index"]].tolist())
            self.assertEqual(len(item["probabilities"]), 10)
            self.assertAlmostEqual(sum(item["probabilities"]), 1, places=5)
            self.assertAlmostEqual(item["confidence"], item["probabilities"][item["prediction"]])
        item = page["items"][0]
        selected = self.service.errors(result["id"], true_label=item["label"], predicted_label=item["prediction"], limit=48)
        self.assertEqual(selected["total"], result["evaluation"]["confusion_matrix"][item["label"]][item["prediction"]])
        self.assertTrue(all(row["label"] == item["label"] and row["prediction"] == item["prediction"] for row in selected["items"]))
        correct_cell = self.service.errors(result["id"], true_label=item["label"], predicted_label=item["label"])
        self.assertEqual(correct_cell["items"], [])

    def test_persisted_runs_restore_without_loading_torch_data(self):
        result = self.finish(epochs=1)
        restored = MnistService(self.root)
        self.assertIsNone(restored.data)
        self.assertEqual(restored.get_run(result["id"])["history"], result["history"])
        self.assertTrue(restored.get_run(result["id"])["checkpoint_available"])
        self.assertEqual(restored.list_runs()["runs"][0]["id"], result["id"])
        # A process restart cannot silently pretend an active task continued.
        record_path = self.root / "runs" / "mnist" / result["id"] / "run.json"
        saved = json.loads(record_path.read_text(encoding="utf-8"))
        saved["status"] = "training"
        record_path.write_text(json.dumps(saved), encoding="utf-8")
        restarted = MnistService(self.root)
        self.assertEqual(restarted.get_run(result["id"])["status"], "interrupted")
        self.assertIsNone(restarted.active_run_id)

    def test_same_seed_and_config_reproduce_training(self):
        first = self.finish(epochs=1)
        second = self.finish(epochs=1)
        one, two = first["history"][0], second["history"][0]
        for key in ["train_loss", "train_accuracy", "validation_loss", "validation_accuracy"]:
            self.assertEqual(one[key], two[key])
        self.assertEqual(first["evaluation"], second["evaluation"])

    def test_constructing_service_does_not_write_active_run_records(self):
        result = self.finish(epochs=1)
        path = self.root / "runs" / "mnist" / result["id"] / "run.json"
        active = copy.deepcopy(result)
        active["status"] = "training"
        path.write_text(json.dumps(active), encoding="utf-8")
        before = path.read_bytes()
        restored = MnistService(self.root)
        self.assertEqual(restored.get_run(result["id"])["status"], "interrupted")
        self.assertEqual(path.read_bytes(), before)

    def test_api_startup_persists_interrupted_state_and_preserves_weights(self):
        from backend.app import app

        result = self.finish(epochs=1)
        path = self.root / "runs" / "mnist" / result["id"] / "run.json"
        active = copy.deepcopy(result)
        active["status"] = "training"
        path.write_text(json.dumps(active), encoding="utf-8")
        checkpoint = self.service.checkpoint_path(result["id"])
        weights_before = checkpoint.read_bytes()
        restored = MnistService(self.root)

        async def startup():
            with patch("backend.app.mnist_service", restored):
                async with app.router.lifespan_context(app):
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(saved["status"], "interrupted")
                    self.assertEqual(saved["history"], result["history"])

        asyncio.run(startup())
        self.assertEqual(checkpoint.read_bytes(), weights_before)

    def test_startup_does_not_overwrite_a_record_completed_after_import(self):
        result = self.finish(epochs=1)
        path = self.root / "runs" / "mnist" / result["id"] / "run.json"
        active = copy.deepcopy(result)
        active["status"] = "training"
        path.write_text(json.dumps(active), encoding="utf-8")
        restored = MnistService(self.root)
        # Another owner committed its result between construction and startup.
        path.write_text(json.dumps(result), encoding="utf-8")
        before = path.read_bytes()
        restored.persist_interrupted()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(restored.get_run(result["id"])["status"], "completed")

    def test_cancelled_and_failed_jobs_release_the_active_slot(self):
        run = self.service.create_run(self.config(epochs=20))
        cancelling = self.service.cancel(run["id"])
        self.assertIn(cancelling["status"], {"cancelling", "cancelled"})
        self.service.wait_for_run(run["id"], timeout=20)
        cancelled = self.service.get_run(run["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["evaluation"])
        self.assertIsNone(self.service.active_run_id)
        with patch("backend.mnist_engine.build_model", side_effect=RuntimeError("synthetic fixture failure")):
            failed = self.service.create_run(self.config())
            self.service.wait_for_run(failed["id"], timeout=20)
        result = self.service.get_run(failed["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("synthetic fixture failure", result["error"])
        self.assertIsNone(self.service.active_run_id)

    def test_busy_and_unprepared_services_return_clear_conflicts(self):
        empty = MnistService(self.root / "unprepared")
        with self.assertRaises(HTTPException) as raised:
            empty.create_run(self.config())
        self.assertEqual(raised.exception.status_code, 409)
        # Block the worker at its first model construction to prove the slot guard.
        entered, release = threading.Event(), threading.Event()
        original = build_model

        def blocked(name):
            entered.set()
            release.wait(timeout=10)
            return original(name)

        with patch("backend.mnist_engine.build_model", side_effect=blocked):
            run = self.service.create_run(self.config(epochs=1))
            self.assertTrue(entered.wait(timeout=5))
            try:
                with self.assertRaises(HTTPException) as raised:
                    self.service.create_run(self.config())
                self.assertEqual(raised.exception.status_code, 409)
            finally:
                self.service.cancel(run["id"])
                release.set()
                self.service.wait_for_run(run["id"], timeout=20)

    def test_worker_persistence_failure_still_releases_resources(self):
        original = self.service._persist
        calls = 0

        def fail_after_creation(run_id):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("synthetic disk full")
            return original(run_id)

        with patch.object(self.service, "_persist", side_effect=fail_after_creation):
            run = self.service.create_run(self.config(epochs=1))
            result = self.service.wait_for_run(run["id"], timeout=20)
        self.assertEqual(result["status"], "failed")
        self.assertIn("synthetic disk full", result["error"])
        self.assertIsNone(self.service.active_run_id)
        self.assertNotIn(run["id"], self.service.start_times)
        self.assertNotIn(run["id"], self.service.cancel_events)
        self.assertNotIn(run["id"], self.service.workers)
        self.finish(epochs=1)  # Storage recovered; the next task is not stuck at 409.

    def test_creation_persistence_failure_does_not_reserve_active_slot(self):
        with patch.object(self.service, "_persist", side_effect=OSError("synthetic read-only disk")):
            with self.assertRaises(HTTPException) as raised:
                self.service.create_run(self.config(epochs=1))
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIsNone(self.service.active_run_id)
        self.assertEqual(self.service.start_times, {})
        self.assertEqual(self.service.cancel_events, {})
        self.assertEqual(self.service.workers, {})
        self.assertEqual(self.service.list_runs()["runs"][0]["status"], "failed")
        self.finish(epochs=1)

    def test_route_validation_and_run_id_safety(self):
        valid = self.config()
        invalid = [
            {"model": "unknown"}, {"epochs": 0}, {"epochs": 21}, {"epochs": True},
            {"batch_size": 16}, {"learning_rate": float("nan")},
            {"learning_rate": float("inf")}, {"learning_rate": 0},
            {"learning_rate": 0.2}, {"validation_ratio": 0.15},
            {"seed": -1}, {"seed": 10000}, {"train_limit": 99}, {"extra": 1},
        ]
        for change in invalid:
            with self.subTest(change=change):
                status, _, _ = asyncio.run(asgi_request("POST", "/api/mnist/runs", {**valid, **change}))
                self.assertEqual(status, 422)
        status, result, _ = asyncio.run(asgi_request("GET", "/api/mnist/runs/not-a-run"))
        self.assertEqual(status, 404)
        for run_id in ["..", "../other", "C:\\private", "0" * 32]:
            with self.assertRaises(HTTPException) as raised:
                self.service.get_run(run_id)
            self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
