import asyncio
import copy
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from fastapi import HTTPException
from pydantic import ValidationError
import torch

from backend.grpo import (GRPOService, RewardRule, RewardRequest, RunConfig, DatasetConfig, ComparisonRequest, TrainingRequest,
    score_reward, prepare_dataset, sample_group, rollout_batch, token_logps, group_advantages, grpo_loss, dataset_payload)
from backend.dpo import state_digest, DPOService, RunConfig as DPOConfig, TrainingRequest as DPOTrain
from backend.sft import SFTService, RunConfig as SFTConfig, CHAT_TEMPLATE
from backend.lora_assets import assets
from backend.tests.test_sft import fixture_loader, fixture_dataset
from backend.tests.test_experiments import asgi_request


def records():
    return [{"prompt":row["question"],"target":row["answer"]} for row in fixture_dataset()]


class GRPOFormulaTests(unittest.TestCase):
    def test_reward_components_and_rule_validation(self):
        rules=[{"type":"contains","text":"lab","weight":2},{"type":"length","min_chars":1,"max_chars":5,"weight":.5},{"type":"contains","text":"wrong","weight":-1}]
        scored=score_reward("Learning LAB", "",rules)
        self.assertEqual(scored["reward"],2)
        self.assertEqual([c["score"] for c in scored["components"]],[1,0,0])
        self.assertEqual(score_reward("", "",[{"type":"nonempty","weight":1}])["reward"],0)
        self.assertEqual(score_reward("  A  B ","a b",[{"type":"exact_target","weight":1}])["reward"],1)
        self.assertEqual(score_reward('{"answer":1}',"",[{"type":"json_object","required_keys":["answer"],"weight":1}])["reward"],1)
        for raw in ('{"answer":NaN}',"[]","bad"):
            self.assertEqual(score_reward(raw,"",[{"type":"json_object","weight":1}])["reward"],0)
        self.assertEqual(score_reward("DONE.","",[{"type":"ends_with","text":"done.","weight":1}])["reward"],1)
        self.assertEqual(score_reward("hello","",[{"type":"avoids","text":"bad","weight":1}])["reward"],1)
        for rule in ({"type":"python","weight":1},{"type":"contains"},{"type":"length","min_chars":20,"max_chars":1},{"type":"nonempty","text":"bad"},{"type":"nonempty","weight":0}):
            with self.assertRaises(ValidationError): RewardRule(**rule)
        # Exported/default-filled rule dictionaries must round-trip through the API schema.
        config=RunConfig().model_dump()
        self.assertEqual(RunConfig(**config).model_dump(),config)
        with self.assertRaises(ValidationError): RewardRequest(rules=[{"type":"exact_target"}],text="answer")

    def test_population_advantages_zero_group_and_clipping_gradients(self):
        advantages,std=group_advantages([0,0,1,1])
        self.assertEqual(std,.5)
        torch.testing.assert_close(advantages,torch.tensor([-1.,-1.,1.,1.])/(1.0002))
        zeros,std=group_advantages([1,1,1,1]);self.assertEqual(std,0);self.assertTrue(torch.equal(zeros,torch.zeros(4)))
        current=torch.tensor([[1.3,.3],[.7,.2]]).log().requires_grad_()
        old=torch.zeros_like(current,requires_grad=True);ref=current.detach().clone().requires_grad_()
        mask=torch.tensor([[True,False],[True,False]])
        loss,kl,clipped=grpo_loss(current,old,ref,torch.tensor([1.,-1.]),mask,0,.2)
        self.assertAlmostEqual(loss.item(),-.2,places=6)
        self.assertEqual(kl.item(),0);self.assertEqual(clipped.item(),1)
        loss.backward();self.assertTrue(torch.equal(current.grad,torch.zeros_like(current)))
        self.assertIsNone(old.grad);self.assertIsNone(ref.grad)
        current=torch.tensor([[-.2,2.],[.3,3.]],requires_grad=True)
        loss,kl,_=grpo_loss(current,torch.zeros_like(current),torch.zeros_like(current),torch.zeros(2),mask,.1,.2)
        self.assertGreaterEqual(kl.item(),0);loss.backward()
        self.assertEqual(current.grad[:,1].abs().sum().item(),0)

    def test_eos_and_length_rollout_keeps_only_sampled_tokens(self):
        class Scripted:
            def __init__(self):self.calls=[]
            def eval(self):return self
            def __call__(self,**kw):
                self.calls.append(kw["attention_mask"].clone())
                logits=torch.full((2,kw["input_ids"].shape[1],4),-1000.)
                logits[0,-1,0]=1000;logits[1,-1,2]=1000
                return SimpleNamespace(logits=logits,past_key_values=())
        model=Scripted();tokenizer=SimpleNamespace(eos_token_id=0,decode=lambda ids,**kw:" ".join(map(str,[i for i in ids if i])))
        samples=sample_group(model,tokenizer,[1,1],2,3,torch.Generator().manual_seed(7))
        self.assertEqual(samples[0]["token_ids"],[0]);self.assertEqual(samples[0]["finish_reason"],"eos")
        self.assertEqual(samples[1]["token_ids"],[2,2,2]);self.assertEqual(samples[1]["finish_reason"],"length")
        inputs,labels,attention=rollout_batch([1,1],samples,0)
        self.assertEqual(labels.tolist(),[[-100,-100,0,-100,-100],[-100,-100,2,2,2]])
        self.assertEqual(attention[0].tolist(),[1,1,1,0,0])
        self.assertEqual(model.calls[-1][0].tolist(),[1,1,1,0])


@unittest.skipUnless(importlib.util.find_spec("peft"),"Install optional LoRA runtime")
class GRPOServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.service=GRPOService(self.temp.name,loader=fixture_loader)
        self.config=RunConfig(dataset=records(),max_length=256,completion_length=8,rules=[{"type":"exact_target","weight":1}]).model_dump()
        # Derive a deterministic, genuinely sampled target so the tiny byte model has a nonconstant first group.
        model,tokenizer=self.service._model(self.config)
        prepared=prepare_dataset(tokenizer,self.config);generator=torch.Generator().manual_seed(self.config["seed"])
        index=prepared["train_indices"][int(torch.randint(len(prepared["train_indices"]),(1,),generator=generator).item())]
        samples=sample_group(model,tokenizer,prepared["prompts"][index],4,8,generator)
        self.first_samples=samples;self.first_index=index
        for row in self.config["dataset"]:row["target"]=samples[0]["text"]
        self.assertNotEqual(samples[0]["text"],samples[1]["text"])

    def test_validation_group_split_prompt_budget_and_target_not_input(self):
        _,tokenizer=self.service._runtime()
        first=prepare_dataset(tokenizer,self.config)
        changed=copy.deepcopy(self.config);changed["dataset"][0]["target"]="Not a prompt token"
        self.assertEqual(first["prompts"],prepare_dataset(tokenizer,changed)["prompts"])
        rows=copy.deepcopy(records());rows.append({"prompt":[{"role":"user","content":rows[0]["prompt"]},{"role":"assistant","content":"history"},{"role":"user","content":"follow up?"}],"target":"x"})
        config=RunConfig(dataset=rows,max_length=256).model_dump();prepared=prepare_dataset(tokenizer,config)
        self.assertEqual(0 in prepared["train_indices"],6 in prepared["train_indices"])
        config["dataset"][0]["prompt"]="long "*100
        self.assertFalse(self.service.preview(config)["valid"])
        with self.assertRaises(HTTPException):self.service.create(config)
        with self.assertRaises(ValidationError):RunConfig(dataset=records()[:4])
        with self.assertRaises(ValidationError):RunConfig(source_type="sft")
        with self.assertRaises(ValidationError):RunConfig(rules=[{"type":"contains_target"}],dataset=[{"prompt":r["prompt"]} for r in records()])

    def test_online_group_old_scores_freeze_and_real_updates(self):
        run=self.service.create(self.config);before=self.service._load(run["id"])
        result=self.service.train(run["id"],TrainingRequest(expected_step=0,steps=1));after=self.service._load(run["id"])
        rollout=result["last_rollout"]
        self.assertEqual(rollout["prompt_index"],self.first_index)
        self.assertEqual([s["token_ids"] for s in rollout["samples"]],[s["token_ids"] for s in self.first_samples])
        self.assertEqual(rollout["old_adapter_digest"],before["reference_digest"])
        self.assertEqual(result["advantage_groups"],1);self.assertEqual(result["optimizer_updates"],2)
        self.assertTrue(result["adapter_changed"])
        self.assertEqual(state_digest(before["reference_adapter"]),state_digest(after["reference_adapter"]))
        self.assertEqual(before["frozen_digest"],after["frozen_digest"])
        self.assertNotEqual(state_digest(before["adapter"]),state_digest(after["adapter"]))
        model,_=self.service._model(self.config,before["adapter"])
        inputs,labels,attention=rollout_batch(prepare_dataset(self.service.tokenizer,self.config)["prompts"][self.first_index],self.first_samples,self.service.tokenizer.eos_token_id)
        values,mask=token_logps(model,inputs,labels,attention)
        self.assertTrue(torch.equal(mask,labels[:,1:]!=-100))
        self.assertTrue(torch.equal(values[~mask],torch.zeros_like(values[~mask])))

    def test_zero_variance_no_reward_update_at_reference(self):
        config={**self.config,"rules":[{"type":"nonempty","weight":1}],"beta":0}
        run=self.service.create(config);before=self.service._load(run["id"])
        result=self.service.train(run["id"],TrainingRequest(expected_step=0,steps=1))
        self.assertEqual(result["advantage_groups"],0);self.assertFalse(result["adapter_changed"])
        self.assertTrue(result["history"][-1]["zero_variance"])
        self.assertEqual(state_digest(before["adapter"]),state_digest(self.service._load(run["id"])["adapter"]))

    def test_exact_resume_and_validation_no_training_rng_consumption(self):
        one=self.service.create(self.config);split=self.service.create(self.config)
        self.service.train(one["id"],TrainingRequest(expected_step=0,steps=2))
        self.service.train(split["id"],TrainingRequest(expected_step=0,steps=1))
        resumed=GRPOService(self.temp.name,loader=fixture_loader)
        resumed.train(split["id"],TrainingRequest(expected_step=1,steps=1))
        a,b=[self.service._load(run["id"]) for run in (one,split)]
        self.assertEqual(state_digest(a["adapter"]),state_digest(b["adapter"]))
        self.assertTrue(torch.equal(a["generator_state"],b["generator_state"]))
        self.assertEqual(a["last_rollout"],b["last_rollout"])
        changed=copy.deepcopy(self.config)
        for i in a["validation_indices"]:changed["dataset"][i]["target"]="Other validation target"
        other=self.service.create(changed);self.service.train(other["id"],TrainingRequest(expected_step=0,steps=2))
        self.assertEqual(state_digest(a["adapter"]),state_digest(self.service._load(other["id"])["adapter"]))

    def test_source_sft_and_dpo_snapshots_export_standard_reload(self):
        from peft import PeftModel
        from backend.lora import adapter_state
        sft=SFTService(Path(self.temp.name)/"sft",loader=fixture_loader)
        source=sft.create(SFTConfig(dataset=fixture_dataset(),max_length=256).model_dump());sft.train(source["id"],DPOTrain(expected_step=0,steps=2))
        dpo=DPOService(Path(self.temp.name)/"dpo",loader=fixture_loader)
        pref=dpo.create(DPOConfig(dataset=[{"prompt":r["prompt"],"chosen":r["target"],"rejected":"Wrong."} for r in records()],max_length=256).model_dump());dpo.train(pref["id"],DPOTrain(expected_step=0,steps=1))
        service=GRPOService(Path(self.temp.name)/"grpo",loader=fixture_loader,source_service=sft,preference_service=dpo)
        for kind,origin in (("sft",source),("dpo",pref)):
            run=service.create({**self.config,"source_type":kind,"source_run_id":origin["id"],"rank":2})
            self.assertEqual(run["config"]["rank"],8);self.assertEqual(run["source"]["kind"],kind)
            with zipfile.ZipFile(io.BytesIO(service.export(run["id"],"latest"))) as package:
                self.assertEqual(package.read("chat_template.jinja").decode(),CHAT_TEMPLATE)
                self.assertEqual(json.loads(package.read("grpo_format.json"))["rules"],self.config["rules"])
                with TemporaryDirectory() as folder:
                    package.extractall(folder);base,_=service._runtime()
                    model=PeftModel.from_pretrained(copy.deepcopy(base),folder,local_files_only=True)
                    self.assertEqual(state_digest(adapter_state(model)),state_digest(service._load(run["id"])["adapter"]))
            if kind=="sft":
                sft.train(origin["id"],DPOTrain(expected_step=2,steps=1))
                self.assertEqual(service.get(run["id"])["reference_digest"],run["reference_digest"])

    def test_compare_target_does_not_leak_and_rng_isolated(self):
        run=self.service.create(self.config)
        request={"messages":[{"role":"user","content":"Question one?"}],"length":4,"temperature":.8}
        rng=torch.random.get_rng_state().clone()
        first=self.service.compare(run["id"],ComparisonRequest(**request,reward_target="one"))
        second=self.service.compare(run["id"],ComparisonRequest(**request,reward_target="two"))
        self.assertEqual(first["before"]["token_ids"],first["after"]["token_ids"])
        self.assertEqual(first["after"]["token_ids"],second["after"]["token_ids"])
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        no_target=self.service.compare(run["id"],ComparisonRequest(**request))
        self.assertEqual(no_target["omitted_target_rules"],1)

    def test_stale_atomic_and_http_reward_training_compare(self):
        run=self.service.create(self.config);path=self.service._path(run["id"]);before=path.read_bytes()
        with self.assertRaises(HTTPException):self.service.train(run["id"],TrainingRequest(expected_step=1,steps=1))
        with patch("backend.transformer.os.replace",side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):self.service.train(run["id"],TrainingRequest(expected_step=0,steps=1))
        self.assertEqual(before,path.read_bytes())
        def call(method,path,body):return asyncio.run(asgi_request(method,path,body))
        with patch("backend.grpo.service",self.service):
            self.assertEqual(call("POST","/api/grpo/reward",{"rules":[{"type":"nonempty"}],"text":"hello"})[1]["reward"],1)
            status,preview,_=call("POST","/api/grpo/preview",{k:v for k,v in self.config.items() if k in DatasetConfig.model_fields});self.assertEqual(status,200);self.assertTrue(preview["valid"])
            status,created,_=call("POST","/api/grpo/runs",self.config);self.assertEqual(status,200)
            base=f"/api/grpo/runs/{created['id']}"
            self.assertEqual(call("POST",base+"/train",{"expected_step":0,"steps":1})[0],200)
            self.assertEqual(call("POST",base+"/compare",{"messages":[{"role":"user","content":"Question one?"}],"length":4})[0],200)
            self.assertEqual(call("POST",base+"/train",{"expected_step":1,"steps":3})[0],422)


@unittest.skipUnless(importlib.util.find_spec("peft") and assets.cached(),"Prepare pinned pretrained cache")
class GRPOPretrainedTests(unittest.TestCase):
    def test_pythia_real_online_sampling_and_updates(self):
        with TemporaryDirectory() as folder:
            service=GRPOService(folder)
            config=RunConfig(rules=[{"type":"length","min_chars":1,"max_chars":45,"weight":1}],completion_length=8).model_dump()
            run=service.create(config)
            result=service.train(run["id"],TrainingRequest(expected_step=0,steps=2))
            self.assertEqual(result["trainable_parameters"],24576)
            self.assertEqual(len(result["last_rollout"]["samples"]),4)
            self.assertTrue(result["frozen_verified"]);self.assertTrue(result["reference_verified"])
            self.assertEqual(result["optimizer_updates"],4)
            self.assertTrue(any(sample["token_ids"] for sample in result["last_rollout"]["samples"]))


if __name__=="__main__":unittest.main()
