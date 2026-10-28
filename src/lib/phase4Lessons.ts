import HandwritingExperiment from "../experiments/inference/HandwritingExperiment";
import DeploymentExperiment from "../experiments/inference/DeploymentExperiment";

export const phase4Lessons = [
  {
    id: "handwriting",
    name: "手写数字识别",
    en: "FROM CHECKPOINT TO PREDICTION",
    title: "写下一个数字，让模型来读。",
    description:
      "加载训练得到的最佳权重，用自己的笔迹检验 MLP、CNN 或自定义网络，并看见每个类别的真实输出。",
    task: "先载入一个官方样本检验所选模型，再手写相同数字，观察预处理后的图像与十类预测分数。",
    concept: "最佳权重 · 图像预处理 · Softmax · 模型推理",
    component: HandwritingExperiment,
    code: 'import torch\nfrom backend.mnist_engine import build_model\n\ncheckpoint = torch.load("models/cnn-best.pt", map_location="cpu", weights_only=True)\nmodel = build_model("cnn")\nmodel.load_state_dict(checkpoint["state_dict"])\nmodel.eval()\n\n# pixels: 后端预处理后的 28×28 灰度图\nx = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 28, 28) / 255\nnorm = checkpoint["normalization"]\nx = (x - norm["mean"]) / norm["std"]\nwith torch.inference_mode():\n    scores = model(x).softmax(dim=1)[0]\nprint(scores.argmax().item(), scores.tolist())\n# 手绘先裁剪、缩放与居中；MNIST 原图保持原始像素。',
  },
  {
    id: "deployment",
    name: "推理接口与部署",
    en: "SERVE YOUR MODEL",
    title: "把训练成果，变成可调用的应用。",
    description:
      "通过真实推理接口连接画布与模型，用 Docker Compose 打包前端、API 和已训练权重。",
    task: "查看当前模型列表与接口文档，在项目根目录运行部署命令，再执行 smoke 检查真实推理链路。",
    concept: "FastAPI · 模型包 · Docker · 持久化",
    component: DeploymentExperiment,
    code: '# 将第三期已完成训练的最佳权重导出到 models/\npython scripts/export_models.py\n\n# 构建并启动：前端、API、模型包使用同一个端口\ndocker compose up --build -d\n\n# 验证首页、资源、健康接口及两个模型的真实预测\npython scripts/smoke_test.py --base-url http://127.0.0.1:8080\n\n# GET  /api/inference/models\n# GET  /api/inference/examples?label=7\n# POST /api/inference/predict\n# {"model_ids": ["模型 ID"], "mode": "mnist", "pixels": [[...]]}',
  },
];
