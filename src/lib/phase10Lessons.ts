import { lazy } from "react";
const Rewards=lazy(()=>import("../experiments/language/GRPOExperiment").then(m=>({default:m.GRPORewards})));
const Train=lazy(()=>import("../experiments/language/GRPOExperiment").then(m=>({default:m.GRPOTraining})));
const Compare=lazy(()=>import("../experiments/language/GRPOExperiment").then(m=>({default:m.GRPOCompare})));
export const phase10Lessons=[
 {id:"grpo-rewards",name:"奖励规则与在线采样",en:"DEFINE WHAT THE POLICY SHOULD REWARD",title:"先写清奖励，再观察模型如何寻找高分。",description:"自定义问题集和可组合奖励规则，试算奖励明细，区分可验证规则与回答质量。",task:"校验问题和规则，再对一段自定义回答计算奖励明细。",concept:"奖励函数 · 关键词 / 格式 · Target · 奖励投机",component:Rewards,
 code:'from backend.grpo import RunConfig, score_reward\n\nconfig = RunConfig().model_dump()\n# config["rules"] = [{"type":"contains", "text":"lab", "weight":1}]\nprint(score_reward("Zhiwei is a learning lab.", "lab", config["rules"]))\n# target 仅供奖励，不进入模型提示\n# 支持关键词、长度、精确 target、JSON 对象等 8 种规则'},
 {id:"grpo-train",name:"真实 LoRA GRPO 更新",en:"SAMPLE, SCORE, NORMALIZE, UPDATE",title:"让真实采样组，驱动策略的下一次更新。",description:"在线采样多条回答，计算组内优势与裁剪目标，真实更新 LoRA，查看奖励、KL 与同分组。",task:"完成至少两轮真实采样更新，出现非同分组、策略参数实际改变，且基座与参考校验通过。",concept:"GRPO · 组优势 · 旧策略 · 裁剪 · 固定参考 KL",component:Train,
 code:'from backend.grpo import RunConfig, TrainingRequest, service\n\nconfig = RunConfig().model_dump()\n# config.update(source_type="sft", source_run_id=已训练SFT实验ID)\nrun = service.create(config)\nfor _ in range(5):\n    run = service.train(run["id"], TrainingRequest(expected_step=run["step"], steps=1))\nprint(run["last_rollout"])\n# A = (reward - group_mean) / (population_std + 1e-4)\n# ratio = exp(logp_policy - logp_old)，old 在同组更新期间固定'},
 {id:"grpo-compare",name:"强化学习前后对话对比",en:"COMPARE REWARDS AND ACTUAL GENERATIONS",title:"奖励提高之后，回答真的更好吗？",description:"对照初始参考与 GRPO 策略的真实对话，使用同一上下文和采样参数，并检查规则得分与退化。",task:"用训练过的 GRPO 权重生成前后对比，检查奖励分与实际文本，再尝试追问。",concept:"同条件对比 · 规则评分 · 泛化 · 奖励投机",component:Compare,
 code:'from backend.grpo import service, ComparisonRequest\n\nrun = service.list()[0]\nresult = service.compare(run["id"], ComparisonRequest(\n    messages=[{"role":"user", "content":"What is Zhiwei?"}],\n    reward_target="lab", temperature=0,\n))\nprint(result["before"]["reply"], result["after"]["reply"])\n# reward_target 只参与评分，不拼入模型输入\n# 同一历史、问题与采样参数；规则分不等于事实正确率'},
];
