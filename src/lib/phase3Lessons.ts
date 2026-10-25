import MnistDataExperiment from "../experiments/mnist/MnistDataExperiment";
import MnistTrainingExperiment from "../experiments/mnist/MnistTrainingExperiment";
import MnistEvaluationExperiment from "../experiments/mnist/MnistEvaluationExperiment";

export const phase3Lessons = [
  {
    id: "mnist-data",
    name: "MNIST 数据划分",
    en: "DATA WITHOUT LEAKAGE",
    title: "让每一份数据，各司其职。",
    description: "从真实的手写数字开始，建立互不重叠的训练、验证与测试流程。",
    task: "查看三个集合的真实样本，改变验证集比例，观察各类别的分层分配与样本数量。",
    concept: "MNIST · 分层划分 · 数据独立",
    component: MnistDataExperiment,
    code: 'from torchvision.datasets import MNIST\nfrom backend.mnist_engine import split_indices\n\nsource = MNIST("data", train=True, download=True)\ntest = MNIST("data", train=False, download=True)\nindices = split_indices(source.targets.numpy(), 7, 0.1, None)\ntrain_idx = indices["effective_train"]\nvalidation_idx = indices["validation"]\n\nassert not set(train_idx) & set(validation_idx)\nprint(len(train_idx), len(validation_idx), len(test))\n# 54000, 6000, 10000\n# 测试集来自独立的官方 t10k 文件，永不加入 train_idx。',
  },
  {
    id: "mnist-train",
    name: "MLP / CNN 训练",
    en: "TRAIN ON REAL IMAGES",
    title: "把训练，交给真实的数据。",
    description:
      "训练全连接与卷积网络，观察每一轮训练和验证的真实指标，并保存最佳权重。",
    task: "保持划分与随机种子一致，分别训练 MLP 和 CNN，再比较验证曲线与最终测试结果。",
    concept: "PyTorch · Epoch · 验证集选模",
    component: MnistTrainingExperiment,
    code: 'import torch\nfrom backend.mnist_engine import build_model\n\nmodel = build_model("cnn")  # 也可选择 "mlp"\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)\ncriterion = torch.nn.CrossEntropyLoss()\n\n# raw_images: 当前批次原始 uint8 灰度图；labels: 真实类别\nimages = (raw_images.float().unsqueeze(1) / 255 - 0.1307) / 0.3081\nmodel.train()\noptimizer.zero_grad()\nlogits = model(images)\nloss = criterion(logits, labels)\nloss.backward()\noptimizer.step()\n\n# 每轮结束后 model.eval() + torch.no_grad() 计算验证指标。\n# 只按验证损失选择最佳模型，测试集不参与这一过程。',
  },
  {
    id: "mnist-evaluation",
    name: "测试评估与错例",
    en: "LOOK BEYOND ACCURACY",
    title: "看看模型，究竟错在哪里。",
    description:
      "在独立测试集上检验最佳模型，用混淆矩阵与真实错例理解每一个错误预测。",
    task: "点击混淆矩阵中的一个非对角单元格，查看对应错例，再打开图片检查十个类别的输出。",
    concept: "混淆矩阵 · Precision / Recall · 错例分析",
    component: MnistEvaluationExperiment,
    code: 'import torch\nimport numpy as np\n\ncheckpoint = torch.load("best.pt", map_location="cpu", weights_only=True)\nmodel.load_state_dict(checkpoint["state_dict"])\nmodel.eval()\nwith torch.no_grad():\n    probabilities = model(test_images).softmax(dim=1)\n    predictions = probabilities.argmax(dim=1)\n\n# 行是真实标签，列是模型预测\nmatrix = np.bincount(\n    true_labels * 10 + predicted_labels, minlength=100\n).reshape(10, 10)\naccuracy = np.trace(matrix) / matrix.sum()\n# 测试输入与训练采用同样的标准化，所有错例来自同一最佳权重。',
  },
];
