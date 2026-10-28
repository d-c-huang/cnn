"""Atomic JSON writes tolerate transient Windows sharing violations only."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.mnist_engine import MnistService


class AtomicPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "run.json"
        self.path.write_text('{"old":true}', encoding="utf-8")

    def test_two_transient_permission_failures_then_success(self):
        original = os.replace
        sources = []
        def shared_reader(source, destination):
            sources.append(source)
            if len(sources) <= 2:
                raise PermissionError("synthetic Windows sharing violation")
            return original(source, destination)
        with patch("backend.mnist_engine.os.replace", shared_reader), \
                patch("backend.mnist_engine.time.sleep") as sleep:
            MnistService._atomic_json(self.path, {"status": "completed", "metric": .8882})
        self.assertEqual(len(sources), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"status": "completed", "metric": .8882})
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_lasting_permission_failure_is_bounded_and_keeps_prior_file(self):
        with patch("backend.mnist_engine.os.replace", side_effect=PermissionError("persistent denial")) as replace, \
                patch("backend.mnist_engine.time.sleep") as sleep:
            with self.assertRaisesRegex(PermissionError, "persistent denial"):
                MnistService._atomic_json(self.path, {"new": True})
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"old": True})
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_each_write_uses_a_unique_temporary_name(self):
        original = os.replace
        sources = []
        def record_source(source, destination):
            sources.append(source)
            return original(source, destination)
        with patch("backend.mnist_engine.os.replace", record_source):
            MnistService._atomic_json(self.path, {"step": 1})
            MnistService._atomic_json(self.path, {"step": 2})
        self.assertEqual(len(set(sources)), 2)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"step": 2})
        self.assertEqual(list(self.root.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
