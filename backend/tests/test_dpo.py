import asyncio
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from fastapi import HTTPException
from pydantic import ValidationError
import torch

from backend.dpo import (DPOService, DatasetConfig, Preference, RunConfig, ChatRequest, TrainingRequest,
    prepare_dataset, sequence_logps, preference_loss, pair_logps, state_digest)
from backend.lora_assets import assets
from backend.sft import SFTService, RunConfig as SFTConfig, CHAT_TEMPLATE, make_batch
from backend.tests.test_sft import fixture_loader, fixture_dataset
from backend.tests.test_experiments import asgi_request


def dataset():
    return [{"prompt": row["question"], "chosen": row["answer"], "rejected": "Wrong reply."} for row in fixture_dataset()]


class DPODataTests(unittest.TestCase):
    def test_validation_roles_duplicates_and_limits(self):
        for change in ({"chosen": " "}, {"rejected": "Answer one."}, {"prompt": []}, {"prompt": [{"role": "assistant", "content": "wrong"}]}):
            with self.assertRaises(ValidationError):
                Preference(**{**dataset()[0], **change})
        with self.assertRaises(ValidationError):
            DatasetConfig(dataset=dataset()[:4])
        with self.assertRaises(ValidationError):
            DatasetConfig(dataset=dataset() + [dataset()[0]])
        with self.assertRaises(ValidationError):
            RunConfig(beta=0)
        with self.assertRaises(ValidationError):
            RunConfig(source_sft_id="../bad")

    def test_shift_sum_masks_padding_and_gradient(self):
        logits = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.tensor([[-100,-100,3,4,1],[-100,2,1,-100,-100]])
        values = sequence_logps(logits, labels)
        logps = logits.log_softmax(-1)
        expected = torch.stack([logps[0,1,3]+logps[0,2,4]+logps[0,3,1],logps[1,0,2]+logps[1,1,1]])
        torch.testing.assert_close(values, expected)
        values.sum().backward()
        self.assertEqual(logits.grad[0,0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1,2:].abs().sum().item(), 0)
        policy = torch.tensor([[-3.,-1.]], requires_grad=True)
        reference = torch.tensor([[-5.,-1.]])
        losses, margins = preference_loss(policy, reference, .1)
        self.assertAlmostEqual(margins.item(), .2, places=6)
        self.assertAlmostEqual(losses.item(), math.log1p(math.exp(-.2)), places=6)
        losses.mean().backward()
        self.assertLess(policy.grad[0,0].item(), 0)
        self.assertGreater(policy.grad[0,1].item(), 0)


@unittest.skipUnless(importlib.util.find_spec("peft"), "Install optional LoRA runtime")
class DPOServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = DPOService(self.temp.name, loader=fixture_loader)
        self.config = RunConfig(dataset=dataset(), max_length=256).model_dump()

    def test_final_completion_only_masks_grouping_and_no_truncation(self):
        rows = dataset()
        rows.append({"prompt":[{"role":"user","content":rows[0]["prompt"]},{"role":"assistant","content":"Earlier answer."},{"role":"user","content":"Follow up?"}],"chosen":"Final answer.","rejected":"Another answer."})
        config = RunConfig(dataset=rows, max_length=256).model_dump()
        _, tokenizer = self.service._runtime()
        prepared = prepare_dataset(tokenizer, config)
        self.assertEqual(0 in prepared["train_indices"], 6 in prepared["train_indices"])
        example = prepared["encoded"][-1][0]
        supervised = [i for i,label in enumerate(example["labels"]) if label != -100]
        text = tokenizer.decode([example["input_ids"][i] for i in supervised])
        self.assertIn("Final answer.", text)
        self.assertIn(tokenizer.eos_token, text)
        self.assertNotIn("Earlier", text)
        self.assertNotIn("Assistant:", text)
        config["dataset"][0]["chosen"] = "long " * 80
        preview = self.service.preview(config)
        self.assertFalse(preview["valid"])
        self.assertGreater(preview["chosen"]["total_tokens"], 256)
        with self.assertRaises(HTTPException):
            self.service.create(config)

    def test_real_updates_reference_freeze_and_exact_resume(self):
        run = self.service.create(self.config)
        saved = self.service._load(run["id"])
        self.assertAlmostEqual(run["history"][0]["train_loss"], math.log(2), places=6)
        self.assertEqual(run["history"][0]["train_margin"], 0)
        result = self.service.train(run["id"], TrainingRequest(expected_step=0, steps=5))
        final = self.service._load(run["id"])
        self.assertLess(result["history"][-1]["train_loss"], math.log(2))
        self.assertEqual(state_digest(saved["reference_adapter"]), state_digest(final["reference_adapter"]))
        self.assertTrue(torch.equal(saved["reference_logps"], final["reference_logps"]))
        self.assertEqual(saved["frozen_digest"], final["frozen_digest"])
        self.assertNotEqual(state_digest(saved["adapter"]), state_digest(final["adapter"]))
        split = self.service.create(self.config)
        self.service.train(split["id"], TrainingRequest(expected_step=0, steps=2))
        resumed_service = DPOService(self.temp.name, loader=fixture_loader)
        resumed_service.train(split["id"], TrainingRequest(expected_step=2, steps=3))
        resumed = resumed_service._load(split["id"])
        for key in final["adapter"]:
            self.assertTrue(torch.equal(final["adapter"][key], resumed["adapter"][key]))

    def test_validation_answers_do_not_change_policy_updates(self):
        first = self.service.create(self.config)
        other = copy.deepcopy(self.config)
        for i in first["validation_indices"]:
            other["dataset"][i]["chosen"] = "A different validation answer."
        second = self.service.create(other)
        for run in (first,second):
            self.service.train(run["id"], TrainingRequest(expected_step=0, steps=3))
        a,b = [self.service._load(run["id"])["adapter"] for run in (first,second)]
        self.assertTrue(all(torch.equal(a[key],b[key]) for key in a))

    def test_sft_reference_snapshot_survives_source_training(self):
        source = SFTService(Path(self.temp.name)/"sft", loader=fixture_loader)
        sft = source.create(SFTConfig(dataset=fixture_dataset(),max_length=256).model_dump())
        source.train(sft["id"], TrainingRequest(expected_step=0,steps=2))
        service = DPOService(Path(self.temp.name)/"dpo", loader=fixture_loader,source_service=source)
        run = service.create({**self.config,"source_sft_id":sft["id"],"rank":2})
        self.assertEqual(run["source"]["step"], 2)
        self.assertEqual(run["config"]["rank"],8)
        before = service._load(run["id"])
        self.assertEqual(state_digest(before["reference_adapter"]),state_digest(source._load(sft["id"])["adapter"]))
        source.train(sft["id"], TrainingRequest(expected_step=2,steps=1))
        service.train(run["id"],TrainingRequest(expected_step=0,steps=2))
        self.assertEqual(service.get(run["id"])["reference_digest"], run["reference_digest"])

    def test_compare_same_context_rng_and_initial_identity(self):
        run = self.service.create(self.config)
        request = ChatRequest(messages=[{"role":"user","content":"Question one?"}],length=8,temperature=.8)
        state = torch.random.get_rng_state().clone()
        result = self.service.compare(run["id"],request)
        self.assertEqual(result["before"],result["after"])
        self.assertTrue(torch.equal(torch.random.get_rng_state(),state))
        self.service.train(run["id"],TrainingRequest(expected_step=0,steps=3))
        after = self.service.compare(run["id"],request)
        self.assertEqual(result["before"],after["before"])
        self.assertEqual(after["before"]["used_messages"],after["after"]["used_messages"])
        self.assertEqual(after["before"]["context_tokens"],after["after"]["context_tokens"])
        self.assertEqual(after,self.service.compare(run["id"],request))

    def test_export_template_and_adapter_standard_reload(self):
        from peft import PeftModel
        from backend.lora import adapter_state
        run = self.service.create(self.config)
        self.service.train(run["id"],TrainingRequest(expected_step=0,steps=2))
        with zipfile.ZipFile(io.BytesIO(self.service.export(run["id"],"latest"))) as package:
            self.assertEqual(package.read("chat_template.jinja").decode(), CHAT_TEMPLATE)
            self.assertNotIn("sft_format.json", package.namelist())
            self.assertEqual(json.loads(package.read("dpo_format.json"))["reference_digest"],run["reference_digest"])
            with TemporaryDirectory() as directory:
                package.extractall(directory)
                base,_ = self.service._runtime()
                model = PeftModel.from_pretrained(copy.deepcopy(base),directory,local_files_only=True)
                saved = self.service._load(run["id"])["adapter"]
                loaded = adapter_state(model)
                self.assertEqual(set(saved),set(loaded))
                self.assertTrue(all(torch.equal(saved[key],loaded[key]) for key in saved))

    def test_stale_atomic_failure_and_http_parameters(self):
        run = self.service.create(self.config)
        before = self.service._path(run["id"]).read_bytes()
        with self.assertRaises(HTTPException):
            self.service.train(run["id"],TrainingRequest(expected_step=1,steps=1))
        with patch.object(self.service,"_save",side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.service.train(run["id"],TrainingRequest(expected_step=0,steps=1))
        self.assertEqual(before,self.service._path(run["id"]).read_bytes())
        def call(method,path,body):
            return asyncio.run(asgi_request(method,path,body))
        with patch("backend.dpo.service",self.service):
            status,preview,_ = call("POST","/api/dpo/preview",{k:v for k,v in self.config.items() if k in DatasetConfig.model_fields})
            self.assertEqual(status,200)
            self.assertTrue(preview["valid"])
            status,created,_ = call("POST","/api/dpo/runs",self.config)
            self.assertEqual(status,200)
            base = f"/api/dpo/runs/{created['id']}"
            self.assertEqual(call("POST",base+"/train",{"expected_step":0,"steps":2})[0],200)
            self.assertEqual(call("POST",base+"/compare",{"messages":[{"role":"user","content":"Question one?"}],"length":4})[0],200)
            self.assertEqual(call("POST",base+"/compare",{"messages":[{"role":"assistant","content":"bad role"}]})[0],422)


@unittest.skipUnless(importlib.util.find_spec("peft") and assets.cached(),"Prepare the pinned pretrained cache")
class DPOPretrainedTests(unittest.TestCase):
    def test_real_pythia_training_and_reference_comparison(self):
        with TemporaryDirectory() as directory:
            service = DPOService(directory)
            run = service.create(RunConfig().model_dump())
            self.assertEqual(run["trainable_parameters"],24576)
            result = service.train(run["id"],TrainingRequest(expected_step=0,steps=5))
            self.assertLess(result["history"][-1]["train_loss"],run["history"][0]["train_loss"])
            comparison = service.compare(run["id"],ChatRequest(messages=[{"role":"user","content":"What is Zhiwei?"}],length=8))
            self.assertEqual(comparison["step"],5)
            self.assertEqual(comparison["before"]["context_tokens"],comparison["after"]["context_tokens"])


if __name__ == "__main__":
    unittest.main()
