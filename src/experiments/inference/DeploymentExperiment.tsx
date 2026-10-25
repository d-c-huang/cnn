import { useEffect, useState } from "react";
import { ArrowRight, Box, Server } from "lucide-react";
import { ErrorMessage, ExperimentPanel, Result } from "../../components/ui";
import { inferenceRequest } from "../../lib/inference";
import type { ModelCatalog } from "../../lib/inference";
import "./inference.css";

export default function DeploymentExperiment({
  onInteract,
}: {
  onInteract: () => void;
}) {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    void inferenceRequest<ModelCatalog>("/models", {
      signal: controller.signal,
    })
      .then(setCatalog)
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error ? cause.message : "无法读取推理服务。",
          );
      })
      .finally(() => {
        if (!controller.signal.aborted) setPending(false);
      });
    return () => controller.abort();
  }, []);
  const origin = window.location.origin;
  return (
    <div className="if-experiment">
      <ExperimentPanel
        title="从实验模型，到一个可调用的服务"
        eyebrow="INFERENCE API"
      >
        <div className="if-serving-flow">
          <span>
            <Box size={17} />
            第三期最佳权重
          </span>
          <ArrowRight size={18} />
          <span>
            <Server size={17} />
            FastAPI / PyTorch
          </span>
          <ArrowRight size={18} />
          <span>画布与外部客户端</span>
        </div>
        <div className="result-grid three-results">
          <Result
            label="当前 API"
            value={pending ? "检查中" : catalog ? "可连接" : "未连接"}
            hint={origin}
          />
          <Result
            label="发现的最佳权重"
            value={catalog ? String(catalog.models.length) : "—"}
            hint="MLP / CNN，按训练记录或模型包读取"
          />
          <Result
            label="推理设备"
            value="CPU"
            hint={
              catalog?.runtime.torch_version
                ? `PyTorch ${catalog.runtime.torch_version}`
                : "首次推理时加载运行时"
            }
          />
        </div>
        <ErrorMessage error={error} />
        <p className="panel-note">
          模型列表可访问表示 API
          已连接；真正推理会严格加载权重并执行前向计算。缺失或不兼容的权重会返回明确错误。
        </p>
        <a
          className="secondary-button"
          href="#handwriting"
          onClick={onInteract}
        >
          在画布上调用推理接口 <ArrowRight size={14} />
        </a>
      </ExperimentPanel>
      <div className="if-workbench">
        <ExperimentPanel
          title="一次推理请求"
          eyebrow="POST /api/inference/predict"
        >
          <p className="panel-note">
            先通过 GET /api/inference/models 获取真实模型 ID；pixels 是 0–255
            灰度矩阵，画布模式的宽高支持 8–320。
          </p>
          <pre className="if-code">
            <code>
              {JSON.stringify(
                {
                  model_ids: [catalog?.defaults.cnn || "<CNN 模型 ID>"],
                  mode: "canvas",
                  pixels: "<二维灰度矩阵，例如 280×280>",
                },
                null,
                2,
              )}
            </code>
          </pre>
          <p className="panel-note">
            响应包含处理后的 28×28
            图像、实际使用的标准化参数、预测类别、十类分数、最佳轮次和模型前向耗时。mode="mnist"
            仅接收原始 28×28 图像，便于与第三期测试一致地核对结果。
          </p>
          <a
            className="text-button"
            href={`${origin}/api/inference/models`}
            target="_blank"
            rel="noreferrer"
          >
            查看当前模型列表 JSON <ArrowRight size={14} />
          </a>
          <a
            className="text-button"
            href="/api/docs"
            target="_blank"
            rel="noreferrer"
          >
            查看 API 文档 <ArrowRight size={14} />
          </a>
        </ExperimentPanel>
        <ExperimentPanel
          title="一个命令，启动完整应用"
          eyebrow="DOCKER COMPOSE"
        >
          <pre className="if-code">
            <code>docker compose up --build -d</code>
          </pre>
          <p className="panel-note">
            在项目根目录执行。首次构建需要安装好的 Docker 与可运行的
            daemon，并从网络获取基础镜像和依赖。
          </p>
          <div className="if-deploy-steps">
            <p>
              <strong>访问</strong>
              <code>http://127.0.0.1:8080</code>
            </p>
            <p>
              <strong>前端</strong>Vite 生产构建，由 API 同端口提供
            </p>
            <p>
              <strong>模型</strong>models/ 中两份真实最佳权重与官方样本
            </p>
            <p>
              <strong>数据</strong>Docker 数据卷保留新训练与 MNIST 缓存
            </p>
          </div>
          <p className="panel-note">
            容器启动不需要重新训练。更新第三期模型后，先在宿主机导出模型包，再重新构建镜像：
          </p>
          <pre className="if-code">
            <code>
              {
                "python scripts/export_models.py\ndocker compose up --build -d\npython scripts/smoke_test.py --base-url http://127.0.0.1:8080"
              }
            </code>
          </pre>
        </ExperimentPanel>
      </div>
      <ExperimentPanel title="日常检查与停止" eyebrow="OPERATIONS">
        <pre className="if-code">
          <code>
            {
              "docker compose ps\ndocker compose logs -f app\ndocker compose down"
            }
          </code>
        </pre>
        <p className="panel-note">
          默认只映射本机 8080 端口，可在 .env 中修改 PORT。down
          保留命名数据卷。完整部署说明见项目中的 docs/deployment.md。
        </p>
      </ExperimentPanel>
    </div>
  );
}
