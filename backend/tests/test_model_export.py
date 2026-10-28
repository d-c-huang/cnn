"""Publishing checks use synthetic checkpoints and isolated deployment roots."""

import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend.inference import InferenceService, PredictRequest
from backend.mnist_engine import build_model, torch_runtime
from backend.tests.test_custom_networks import cnn_layers
from scripts import export_models


class ModelExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = torch_runtime()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = {}
        for index, architecture in enumerate(("mlp", "cnn", "custom"), 1):
            identifier = str(index) * 32
            config = {"model": architecture}
            if architecture == "custom":
                config["architecture"] = cnn_layers()
                model = build_model("custom", config["architecture"])
            else:
                model = build_model(architecture)
            directory = self.root / "runs" / "mnist" / identifier
            directory.mkdir(parents=True)
            checkpoint = {"config": config, "state_dict": model.state_dict(), "epoch": 2,
                          "validation_loss": .25, "normalization": {"mean": .1307, "std": .3081}}
            self.torch.save(checkpoint, directory / "best.pt")
            record = {"id": identifier, "config": config, "status": "completed",
                      "created_at": f"2026-10-2{index}T00:00:00+00:00", "best_epoch": 2,
                      "best_validation_loss": .25, "evaluation": {"accuracy": .8},
                      "history": [{"epoch": 1, "validation_loss": .5}, {"epoch": 2, "validation_loss": .25}]}
            (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
            self.records[architecture] = record

    def run_export(self, *options):
        output, error = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["export_models.py", "--root", str(self.root), *options]), \
                patch("sys.stdout", output), patch("sys.stderr", error):
            status = export_models.main()
        return status, output.getvalue(), error.getvalue()

    def test_default_export_remains_mlp_and_cnn_even_when_custom_runs_exist(self):
        records = export_models.completed_runs(self.root)
        self.assertEqual({record["config"]["model"] for record in records}, {"mlp", "cnn", "custom"})
        status, output, error = self.run_export()
        self.assertEqual(status, 0, error)
        manifest = json.loads((self.root / "models" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([row["architecture"] for row in manifest["models"]], ["mlp", "cnn"])
        self.assertEqual([row["model"] for row in json.loads(output)["exported"]], ["mlp", "cnn"])
        self.assertFalse((self.root / "models" / "custom-best.pt").exists())

    def test_optional_custom_bundle_is_usable_without_runs_or_dataset(self):
        status, _, error = self.run_export("--custom-run", self.records["custom"]["id"])
        self.assertEqual(status, 0, error)
        deployed = self.root / "deployed"
        shutil.copytree(self.root / "models", deployed / "models")
        inference = InferenceService(deployed)
        catalog = inference.models()
        self.assertEqual({row["architecture"] for row in catalog["models"]}, {"mlp", "cnn", "custom"})
        custom = next(row for row in catalog["models"] if row["architecture"] == "custom")
        self.assertEqual(custom["source"], "bundle")
        self.assertEqual(custom["structure"], cnn_layers())
        pixels = np.zeros((28, 28), dtype=np.uint8)
        pixels[4:24, 10:16] = 200
        result = inference.predict(PredictRequest(model_ids=[custom["id"]], pixels=pixels.tolist(), mode="mnist"))
        checkpoint = self.torch.load(self.root / "runs" / "mnist" / custom["id"] / "best.pt", weights_only=True)
        model = build_model("custom", cnn_layers()).eval()
        model.load_state_dict(checkpoint["state_dict"])
        with self.torch.inference_mode():
            expected = model(self.torch.as_tensor(pixels.copy()).reshape(1, 1, 28, 28).float().div(255).sub(.1307).div(.3081)).softmax(1)[0].numpy()
        np.testing.assert_allclose(result["predictions"][0]["probabilities"], expected, atol=1e-7, rtol=0)

    def test_invalid_custom_checkpoint_does_not_replace_an_existing_package(self):
        self.assertEqual(self.run_export()[0], 0)
        directory = self.root / "models"
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        checkpoint_path = self.root / "runs" / "mnist" / self.records["custom"]["id"] / "best.pt"
        checkpoint = self.torch.load(checkpoint_path, weights_only=True)
        checkpoint["config"] = copy.deepcopy(checkpoint["config"])
        checkpoint["config"]["architecture"][0]["outChannels"] = 16
        self.torch.save(checkpoint, checkpoint_path)
        status, _, error = self.run_export("--custom-run", self.records["custom"]["id"])
        self.assertEqual(status, 1)
        self.assertIn("layer structure", error)
        self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, before)

    def test_failed_publication_restores_all_prior_files(self):
        destination = self.root / "published"
        staging = self.root / "staging"
        destination.mkdir()
        staging.mkdir()
        filenames = ["mlp-best.pt", "cnn-best.pt", "manifest.json"]
        for name in filenames:
            (destination / name).write_bytes(f"old-{name}".encode())
            (staging / name).write_bytes(f"new-{name}".encode())
        original = Path.replace
        def failed_manifest(source, target):
            if source == staging / "manifest.json":
                raise OSError("synthetic publish failure")
            return original(source, target)
        with patch.object(Path, "replace", failed_manifest):
            with self.assertRaises(OSError):
                export_models.publish_staged(staging, destination, filenames)
        self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()},
                         {name: f"old-{name}".encode() for name in filenames})

    def test_invalid_or_unknown_custom_identifier_does_not_change_output(self):
        for identifier in ("../outside", "f" * 32):
            with self.subTest(identifier=identifier):
                status, _, _ = self.run_export("--custom-run", identifier)
                self.assertEqual(status, 1)
        self.assertFalse((self.root / "models").exists())


if __name__ == "__main__":
    unittest.main()
