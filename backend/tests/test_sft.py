import asyncio
import copy
import importlib.util
import io
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from fastapi import HTTPException
from pydantic import ValidationError
import torch

from backend.lora import TrainingRequest
from backend.lora_assets import assets
from backend.sft import (CHAT_TEMPLATE, TEMPLATE_VERSION, ChatRequest, DatasetConfig, RunConfig, SFTService,
    answer_loss, make_batch, prepare_dataset, tokenize_example)
from backend.tests.test_experiments import asgi_request

HAS_PEFT = all(importlib.util.find_spec(name) is not None for name in ("peft", "transformers", "tokenizers"))


def fixture_dataset():
    return [{"question": f"Question {word}?", "answer": f"Answer {word}."} for word in ("one", "two", "three", "four", "five", "six")]


def fixture_loader():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import AutoModelForCausalLM, GPTNeoXConfig, PreTrainedTokenizerFast

    words = ["<eos>", "<unk>", *sorted(pre_tokenizers.ByteLevel.alphabet())]
    backend = Tokenizer(models.BPE(vocab={word: index for index, word in enumerate(words)}, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = AutoModelForCausalLM.from_config(GPTNeoXConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=256, attention_dropout=0.0, hidden_dropout=0.0))
    return model, tokenizer


@unittest.skipUnless(HAS_PEFT, "Install the LoRA dependencies for SFT checks")
class SFTTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = SFTService(self.directory.name, loader=fixture_loader)
        self.config = RunConfig(dataset=fixture_dataset(), system_prompt="Be brief.", rank=4, alpha=8).model_dump()

    def test_template_masks_only_answer_and_eos_for_every_turn(self):
        _, tokenizer = fixture_loader()
        messages = [{"role": "user", "content": "Question one?"}, {"role": "assistant", "content": "Answer one."},
            {"role": "user", "content": "More?"}, {"role": "assistant", "content": "More."}]
        item = tokenize_example(tokenizer, messages, "Be brief.")
        supervised = [label for label in item["labels"] if label != -100]
        self.assertEqual(supervised, tokenizer.encode(" Answer one.", add_special_tokens=False) + [tokenizer.eos_token_id]
            + tokenizer.encode(" More.", add_special_tokens=False) + [tokenizer.eos_token_id])
        prefix = tokenizer.encode("System: Be brief.\nUser: Question one?\nAssistant:", add_special_tokens=False)
        self.assertTrue(all(label == -100 for label in item["labels"][:len(prefix)]))
        self.assertNotEqual(item["labels"][len(prefix)], -100, "The first answer token must be supervised")

    def test_padding_mask_and_shifted_loss_match_manual_targets(self):
        _, tokenizer = fixture_loader()
        short = tokenize_example(tokenizer, [{"role": "user", "content": "Question one?"}, {"role": "assistant", "content": "Answer one."}], "")
        long = tokenize_example(tokenizer, [{"role": "user", "content": "Question two?"}, {"role": "assistant", "content": "Answer two. More."}], "Be brief.")
        inputs, labels, attention = make_batch([short, long], tokenizer.eos_token_id)
        self.assertTrue((labels[attention == 0] == -100).all())
        self.assertTrue((inputs[attention == 0] == tokenizer.eos_token_id).all())
        logits = torch.randn(2, inputs.shape[1], len(tokenizer), dtype=torch.float64, generator=torch.Generator().manual_seed(5), requires_grad=True)
        loss = answer_loss(logits, labels)
        valid = labels[:, 1:] != -100
        rows = logits[:, :-1][valid]
        targets = labels[:, 1:][valid]
        manual = -rows.log_softmax(-1).gather(1, targets[:, None]).mean()
        torch.testing.assert_close(loss, manual)
        loss.backward()
        self.assertEqual(logits.grad[:, :-1][~valid].count_nonzero().item(), 0)
        self.assertGreater(logits.grad[:, :-1][valid].count_nonzero().item(), 0)

    def test_split_groups_same_opening_question_and_rejects_bad_data(self):
        _, tokenizer = fixture_loader()
        dataset = fixture_dataset() + [{"messages": [{"role": "user", "content": "Question one?"}, {"role": "assistant", "content": "Answer one."},
            {"role": "user", "content": "More?"}, {"role": "assistant", "content": "More."}]}]
        config = RunConfig(dataset=dataset, system_prompt="Be brief.").model_dump()
        prepared = prepare_dataset(tokenizer, config)
        self.assertEqual(0 in prepared["train_indices"], 6 in prepared["train_indices"])
        self.assertFalse(set(prepared["train_indices"]) & set(prepared["validation_indices"]))
        self.assertEqual(prepared["dataset_fingerprint"], prepare_dataset(tokenizer, config)["dataset_fingerprint"])
        for patch_value in [dataset[:4], fixture_dataset() + [fixture_dataset()[0]], fixture_dataset()[:-1] + [{"question": " ", "answer": "Answer one."}]]:
            with self.assertRaises(ValidationError):
                DatasetConfig(dataset=patch_value)

    def test_oversize_preview_rejects_creation_without_silent_answer_truncation(self):
        config = copy.deepcopy(self.config)
        config["dataset"][0]["answer"] = "Answer one. " * 80
        config["max_length"] = 64
        preview = self.service.preview(config)
        self.assertFalse(preview["valid"])
        self.assertGreater(preview["total_tokens"], 64)
        with self.assertRaises(HTTPException) as error:
            self.service.create(config)
        self.assertEqual(error.exception.status_code, 422)

    def test_real_training_freezes_base_and_exactly_resumes(self):
        a, b = self.service.create(self.config), self.service.create(self.config)
        self.service.train(a["id"], TrainingRequest(expected_step=0, steps=2))
        restarted = SFTService(self.directory.name, loader=fixture_loader)
        final = restarted.train(a["id"], TrainingRequest(expected_step=2, steps=3))
        self.service.train(b["id"], TrainingRequest(expected_step=0, steps=5))
        left, right = restarted._load(a["id"]), self.service._load(b["id"])
        self.assertTrue(final["frozen_verified"])
        self.assertEqual(left["frozen_digest"], right["frozen_digest"])
        self.assertLess(final["history"][-1]["train_loss"], a["history"][0]["train_loss"])
        for name, value in left["adapter"].items():
            torch.testing.assert_close(value, right["adapter"][name], rtol=0, atol=0)
        self.assertEqual(left["history"][-1], right["history"][-1])

    def test_validation_answers_never_influence_updates(self):
        _, tokenizer = fixture_loader()
        prepared = prepare_dataset(tokenizer, self.config)
        changed = copy.deepcopy(self.config)
        for index in prepared["validation_indices"]:
            changed["dataset"][index]["answer"] = "Unseen answer."
        states = []
        for config in (self.config, changed):
            run = self.service.create(config)
            self.service.train(run["id"], TrainingRequest(expected_step=0, steps=3))
            states.append(self.service._load(run["id"]))
        for name, value in states[0]["adapter"].items():
            torch.testing.assert_close(value, states[1]["adapter"][name], rtol=0, atol=0)
        self.assertNotEqual(states[0]["history"][-1]["validation_loss"], states[1]["history"][-1]["validation_loss"])

    def test_chat_context_preserves_template_and_generation_does_not_mutate(self):
        config = {**self.config, "max_length": 128}
        run = self.service.create(config)
        self.service.train(run["id"], TrainingRequest(expected_step=0, steps=2))
        messages = []
        for _ in range(4):
            messages += [{"role": "user", "content": "Question one? " * 5}, {"role": "assistant", "content": "Answer one. " * 5}]
        messages.append({"role": "user", "content": "Question two?"})
        request = ChatRequest(messages=messages, length=8, temperature=0.8)
        previous = self.service._path(run["id"]).read_bytes()
        rng = torch.get_rng_state().clone()
        a, b = self.service.chat(run["id"], request), self.service.chat(run["id"], request)
        self.assertEqual(a, b)
        self.assertGreater(a["dropped_turns"], 0)
        self.assertEqual(a["used_messages"][-1]["content"], "Question two?")
        self.assertEqual(a["used_messages"][0]["role"], "user")
        self.assertLessEqual(a["context_tokens"] + len(a["token_ids"]), 128)
        self.assertEqual(previous, self.service._path(run["id"]).read_bytes())
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[{"role": "assistant", "content": "Answer one."}])
        with self.assertRaises(HTTPException) as error:
            self.service.chat(run["id"], ChatRequest(messages=[{"role": "user", "content": "Question one? " * 30}]))
        self.assertEqual(error.exception.status_code, 422)

    def test_chat_stops_on_eos_and_does_not_return_a_fabricated_user_turn(self):
        run = self.service.create(self.config)
        _, tokenizer = self.service._runtime()

        class ScriptedModel:
            def __init__(self, tokens):
                self.tokens = iter(tokens)

            def eval(self):
                return self

            def __call__(self, input_ids, use_cache=False):
                logits = torch.full((1, input_ids.shape[1], len(tokenizer)), -100.0)
                logits[0, -1, next(self.tokens)] = 100.0
                return SimpleNamespace(logits=logits)

        request = ChatRequest(messages=[{"role": "user", "content": "Question one?"}], length=32)
        with patch.object(self.service, "_model", return_value=(ScriptedModel([tokenizer.eos_token_id]), tokenizer)):
            result = self.service.chat(run["id"], request)
            self.assertEqual(result["finish_reason"], "eos")
            self.assertEqual(result["reply"], "")
        tokens = tokenizer.encode(" safe reply\nUser:", add_special_tokens=False)
        with patch.object(self.service, "_model", return_value=(ScriptedModel(tokens), tokenizer)):
            result = self.service.chat(run["id"], request)
            self.assertEqual(result["finish_reason"], "role_boundary")
            self.assertEqual(result["reply"], "safe reply")

    def test_export_contains_same_chat_template_and_best_variant(self):
        run = self.service.create(self.config)
        self.service.train(run["id"], TrainingRequest(expected_step=0, steps=2))
        with zipfile.ZipFile(io.BytesIO(self.service.export(run["id"], "best"))) as package:
            self.assertEqual(package.read("chat_template.jinja").decode(), CHAT_TEMPLATE)
            description = json.loads(package.read("sft_format.json"))
            self.assertEqual(description["template_version"], TEMPLATE_VERSION)
            self.assertEqual(description["system_prompt"], self.config["system_prompt"])
            self.assertIn("adapter_model.safetensors", package.namelist())

    def test_stale_step_and_failed_save_keep_previous_state(self):
        run = self.service.create(self.config)
        with self.assertRaises(HTTPException) as error:
            self.service.train(run["id"], TrainingRequest(expected_step=3, steps=1))
        self.assertEqual(error.exception.status_code, 409)
        previous = self.service._path(run["id"]).read_bytes()
        with patch("backend.transformer.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.service.train(run["id"], TrainingRequest(expected_step=0, steps=1))
        self.assertEqual(previous, self.service._path(run["id"]).read_bytes())

    def test_http_dataset_preview_training_and_chat(self):
        def call(method, path, body=None):
            return asyncio.run(asgi_request(method, path, body))
        with patch("backend.sft.service", self.service):
            status, preview, _ = call("POST", "/api/sft/preview", {key: self.config[key] for key in ("dataset", "system_prompt", "max_length", "seed")})
            self.assertEqual(status, 200)
            self.assertTrue(preview["valid"])
            status, run, _ = call("POST", "/api/sft/runs", self.config)
            self.assertEqual(status, 200)
            base = f"/api/sft/runs/{run['id']}"
            self.assertEqual(call("POST", base + "/train", {"expected_step": 0, "steps": 2})[0], 200)
            self.assertEqual(call("POST", base + "/chat", {"messages": [{"role": "user", "content": "Question one?"}], "length": 4})[0], 200)
            self.assertEqual(call("POST", base + "/chat", {"messages": [{"role": "assistant", "content": "Wrong role."}]})[0], 422)


@unittest.skipUnless(HAS_PEFT and assets.cached(), "Prepare the pinned Pythia cache for the real SFT integration test")
class SFTPretrainedTests(unittest.TestCase):
    def test_pretrained_answer_mask_real_lora_and_conversation(self):
        with TemporaryDirectory() as root:
            service = SFTService(root)
            config = RunConfig().model_dump()
            preview = service.preview(config)
            self.assertTrue(preview["valid"])
            self.assertTrue(any(token["supervised"] for token in preview["tokens"]))
            self.assertTrue(any(not token["supervised"] for token in preview["tokens"]))
            run = service.create(config)
            self.assertEqual(run["base_parameters"], 14_067_712)
            self.assertEqual(run["trainable_parameters"], 24_576)
            result = service.train(run["id"], TrainingRequest(expected_step=0, steps=5))
            self.assertLess(result["history"][-1]["train_loss"], run["history"][0]["train_loss"])
            reply = service.chat(run["id"], ChatRequest(messages=[{"role": "user", "content": "What is Zhiwei?"}], length=8))
            self.assertEqual(reply["step"], 5)
            self.assertTrue(reply["token_ids"])


if __name__ == "__main__":
    unittest.main()
