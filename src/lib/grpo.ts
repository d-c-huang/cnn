import type { ChatMessage } from "./sft";
import type { DPOComparison } from "./dpo";
import type { LoRAStatus } from "./lora";
export type GRPOPrompt = {prompt: string | ChatMessage[]; target?: string};
export type RewardRule = {type:string;weight?:number;text?:string;case_sensitive?:boolean;min_chars?:number;max_chars?:number;required_keys?:string[]};
export type RewardScore = {reward:number;components:{index:number;type:string;score:number;weight:number;weighted:number}[]};
export type GRPODraft = {dataset_text:string;rules_text:string;system_prompt:string;max_length:number;completion_length:number;seed:number;rank:number;alpha:number;group_size:number;learning_rate:number;beta:number;clip_epsilon:number;updates_per_rollout:number;source_type:"pretrained"|"sft"|"dpo";source_run_id:string|null;source_checkpoint:"latest"|"best"};
export type GRPOMetric = {step:number;rollout_reward:number|null;reward_std:number|null;loss:number|null;kl:number|null;clip_fraction:number|null;validation_reward:number;zero_variance:boolean|null};
export type GRPOPreview = {valid:boolean;errors:string[];train_indices:number[];validation_indices:number[];prompt_tokens:number[];reward_bound:number;rule_version:string};
export type GRPOSample = RewardScore & {text:string;token_ids:number[];finish_reason:string;advantage:number};
export type GRPORun = {id:string;config:Omit<GRPODraft,"dataset_text"|"rules_text"> & {dataset:GRPOPrompt[];rules:RewardRule[]};step:number;best_step:number;history:GRPOMetric[];source:{kind:string;run_id?:string;step:number};base_parameters:number;trainable_parameters:number;frozen_verified:boolean;reference_verified:boolean;reference_digest:string;adapter_changed:boolean;advantage_groups:number;optimizer_updates:number;train_indices:number[];validation_indices:number[];dataset_fingerprint:string;
 last_rollout:null|{prompt_index:number;old_policy_step:number;old_adapter_digest:string;samples:GRPOSample[]};validation_samples:(RewardScore & {index:number;text:string;finish_reason:string})[]};
export type GRPOStatus=LoRAStatus & {dataset:GRPOPrompt[];rules:RewardRule[]};
export type GRPOComparison=Omit<DPOComparison,"before"|"after"> & {before:DPOComparison["before"] & {rule_score:RewardScore};after:DPOComparison["after"] & {rule_score:RewardScore};omitted_target_rules:number;reward_target:string};
export const grpoDefaultRules:RewardRule[]=[{type:"contains_target",weight:1},{type:"length",weight:.25,min_chars:8,max_chars:100},{type:"nonempty",weight:.1}];
export const grpoDefaults:GRPODraft={dataset_text:"",rules_text:JSON.stringify(grpoDefaultRules,null,2),system_prompt:"You are Zhiwei. Answer briefly and clearly.",max_length:128,completion_length:16,seed:7,rank:8,alpha:16,group_size:4,learning_rate:.00003,beta:.02,clip_epsilon:.2,updates_per_rollout:2,source_type:"pretrained",source_run_id:null,source_checkpoint:"latest"};
export function grpoQuestion(row:GRPOPrompt):string {return typeof row.prompt==="string"?row.prompt:row.prompt.at(-1)!.content;}
export function parseGRPOPrompts(raw:string):GRPOPrompt[] {
 const text=raw.replace(/^\uFEFF/,"").trim(); if(!text||text.length>100000)throw new Error("请输入不超过 100,000 字符的 JSON / JSONL 问题集。");
 let rows:unknown;
 if(text.startsWith("[")){try{rows=JSON.parse(text);}catch{throw new Error("问题集 JSON 数组无效。");}}
 else rows=text.split(/\r?\n/).flatMap((line,i)=>{if(!line.trim())return [];try{return [JSON.parse(line)];}catch{throw new Error(`JSONL 第 ${i+1} 行无效。`);}});
 if(!Array.isArray(rows)||!rows.length||rows.length>32)throw new Error("问题集须为 1–32 条记录，训练至少需要五个不同首问。");
 rows.forEach((row,i)=>{
  if(!row||typeof row!=="object"||!Object.hasOwn(row,"prompt")||Object.keys(row).some(k=>!["prompt","target"].includes(k))||(row.target!==undefined&&typeof row.target!=="string"))throw new Error(`第 ${i+1} 条须含 prompt，可选字符串 target。`);
  if(typeof row.prompt==="string"&&row.prompt.trim())return;
  if(Array.isArray(row.prompt)&&row.prompt.length>=1&&row.prompt.length<=7&&row.prompt.length%2===1&&row.prompt.every((m:ChatMessage,p:number)=>m&&Object.keys(m).sort().join(",")==="content,role"&&m.role===(p%2?"assistant":"user")&&typeof m.content==="string"&&m.content.trim()))return;
  throw new Error(`第 ${i+1} 条 prompt 应为非空文本或以 user 结束的交替消息列表。`);
 });return rows as GRPOPrompt[];
}
export function parseRewardRules(raw:string):RewardRule[]{
 if(raw.length>20000)throw new Error("奖励规则不能超过 20,000 字符。");let rules:unknown;try{rules=JSON.parse(raw);}catch{throw new Error("奖励规则须为 JSON 数组。");}
 const types=["nonempty","contains","avoids","contains_target","exact_target","length","ends_with","json_object"];
 if(!Array.isArray(rules)||!rules.length||rules.length>8)throw new Error("需要 1–8 条奖励规则。");
 rules.forEach((r,i)=>{if(!r||typeof r!=="object"||!types.includes(r.type)||(r.weight!==undefined&&(!Number.isFinite(r.weight)||r.weight===0||Math.abs(r.weight)>5)))throw new Error(`第 ${i+1} 条奖励 type / weight 无效。`);});return rules as RewardRule[];
}
export function readGRPODraft():GRPODraft|null {
 try{const v=JSON.parse(localStorage.getItem("zhiwei-grpo-draft-v1")||"null");if(!v||typeof v.dataset_text!=="string"||v.dataset_text.length>100000)return null;
 const next={...grpoDefaults,dataset_text:v.dataset_text,rules_text:typeof v.rules_text==="string"&&v.rules_text.length<=20000?v.rules_text:grpoDefaults.rules_text};
 next.system_prompt=typeof v.system_prompt==="string"?v.system_prompt.slice(0,256):next.system_prompt;
 for(const [key,allowed] of Object.entries({max_length:[64,128,256],completion_length:[8,16,32],rank:[2,4,8,16],alpha:[4,8,16,32],group_size:[2,4],updates_per_rollout:[1,2]})) if(allowed.includes(v[key]))Object.assign(next,{[key]:v[key]});
 for(const [key,min,max] of [["learning_rate",.000001,.001],["beta",0,.5],["clip_epsilon",.05,.4]] as const)if(typeof v[key]==="number"&&Number.isFinite(v[key])&&v[key]>=min&&v[key]<=max)next[key]=v[key];
 if(Number.isInteger(v.seed)&&v.seed>=0&&v.seed<=4294967295)next.seed=v.seed;
 if(["sft","dpo"].includes(v.source_type)&&typeof v.source_run_id==="string"&&/^[0-9a-f]{32}$/.test(v.source_run_id)){next.source_type=v.source_type;next.source_run_id=v.source_run_id;}
 next.source_checkpoint=v.source_checkpoint==="best"?"best":"latest";return next;
 }catch{return null;}
}
export async function grpoApi<T>(path:string,payload?:unknown,signal?:AbortSignal):Promise<T>{
 const response=await fetch(`/api/grpo${path}`,{method:payload===undefined?"GET":"POST",headers:payload===undefined?{}:{"Content-Type":"application/json"},body:payload===undefined?undefined:JSON.stringify(payload),signal:signal?AbortSignal.any([signal,AbortSignal.timeout(90000)]):AbortSignal.timeout(90000)});
 const result=await response.json().catch(()=>({detail:`服务返回 ${response.status}`}));if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:Array.isArray(result.detail)?result.detail.map((x:{msg?:string})=>x.msg??"参数无效").join("；"):"GRPO 请求失败");return result as T;
}
