export type LayerType = "conv" | "relu" | "pool" | "flatten" | "linear";

type LayerBase = { id: string };
export type NetworkLayer =
  | (LayerBase & { type: "conv"; outChannels: number; kernel: number; stride: number; padding: number })
  | (LayerBase & { type: "relu" | "pool" | "flatten" })
  | (LayerBase & { type: "linear"; units: number });

export type LayerAnalysis = {
  input: number[] | null;
  output: number[] | null;
  parameters: number;
  error: string | null;
};

export type NetworkAnalysis = {
  rows: LayerAnalysis[];
  totalParameters: number;
  output: number[] | null;
  valid: boolean;
  shapeValid: boolean;
  classifierReady: boolean;
  errors: string[];
};

export const NETWORK_STORAGE_KEY = "zhiwei-network-builder-v1";
export const MAX_NETWORK_LAYERS = 16;
export const NETWORK_INPUT = [1, 28, 28] as const;

let layerSequence = 0;
export function createLayer(type: LayerType, id = `layer-${Date.now().toString(36)}-${++layerSequence}`): NetworkLayer {
  if (type === "conv") return { id, type, outChannels: 8, kernel: 3, stride: 1, padding: 1 };
  if (type === "linear") return { id, type, units: 10 };
  return { id, type };
}

export function createExample(type: "cnn" | "mlp"): NetworkLayer[] {
  if (type === "mlp") {
    return [createLayer("flatten"), { ...createLayer("linear"), type: "linear", units: 128 }, createLayer("relu"), createLayer("linear")];
  }
  return [createLayer("conv"), createLayer("relu"), createLayer("pool"), createLayer("flatten"), createLayer("linear")];
}

/** Validate persisted/user-supplied structure before using it for shape analysis or code export. */
export function parseNetwork(value: unknown): NetworkLayer[] | null {
  if (!Array.isArray(value) || value.length > MAX_NETWORK_LAYERS) return null;
  const ids = new Set<string>();
  const parsed: NetworkLayer[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) return null;
    const layer = item as Record<string, unknown>;
    if (typeof layer.id !== "string" || !/^[\w-]{1,64}$/.test(layer.id) || ids.has(layer.id)) return null;
    ids.add(layer.id);
    const keys = Object.keys(layer).sort().join(",");
    if (layer.type === "conv") {
      if (keys !== "id,kernel,outChannels,padding,stride,type" || ![8, 16, 32, 64].includes(layer.outChannels as number) || ![3, 5].includes(layer.kernel as number) || ![1, 2].includes(layer.stride as number) || ![0, 1, 2].includes(layer.padding as number)) return null;
      parsed.push({ id: layer.id, type: "conv", outChannels: layer.outChannels as number, kernel: layer.kernel as number, stride: layer.stride as number, padding: layer.padding as number });
    } else if (layer.type === "linear") {
      if (keys !== "id,type,units" || ![10, 32, 64, 128].includes(layer.units as number)) return null;
      parsed.push({ id: layer.id, type: "linear", units: layer.units as number });
    } else if (layer.type === "relu" || layer.type === "pool" || layer.type === "flatten") {
      if (keys !== "id,type") return null;
      parsed.push({ id: layer.id, type: layer.type });
    } else return null;
  }
  return parsed;
}

export function readSavedNetwork(): NetworkLayer[] | null {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(NETWORK_STORAGE_KEY) || "null");
    if (!saved || typeof saved !== "object" || !("version" in saved) || saved.version !== 1 || !("layers" in saved)) return null;
    return parseNetwork(saved.layers);
  } catch {
    return null;
  }
}

/** Spatial shapes are C×H×W, vectors are [features]; the batch dimension is excluded. */
export function analyzeNetwork(layers: NetworkLayer[]): NetworkAnalysis {
  const validated = parseNetwork(layers);
  if (!validated) return { rows: [], totalParameters: 0, output: null, valid: false, shapeValid: false, classifierReady: false, errors: ["结构配置无效：请使用支持的层与参数，最多 16 层。"] };
  let current: number[] | null = [...NETWORK_INPUT];
  let totalParameters = 0;
  const errors: string[] = [];
  const rows: LayerAnalysis[] = validated.map((layer, index) => {
    const input: number[] | null = current ? [...current] : null;
    if (!current || !input) return { input: null, output: null, parameters: 0, error: "前序层无效，暂时无法推导此层。" };
    let output = [...current];
    let parameters = 0;
    let error: string | null = null;
    if (layer.type === "conv" || layer.type === "pool") {
      if (current.length !== 3) error = `${layer.type === "conv" ? "Conv2D" : "MaxPool2D"} 需要 [通道, 高, 宽]，不能接在展平后的向量上。`;
      else {
        const kernel = layer.type === "conv" ? layer.kernel : 2;
        const stride = layer.type === "conv" ? layer.stride : 2;
        const padding = layer.type === "conv" ? layer.padding : 0;
        const height = Math.floor((current[1] + 2 * padding - kernel) / stride) + 1;
        const width = Math.floor((current[2] + 2 * padding - kernel) / stride) + 1;
        if (height <= 0 || width <= 0) error = `卷积核/池化窗口 ${kernel}×${kernel} 超过当前有效输入，输出空间尺寸必须大于 0。`;
        else {
          output = [layer.type === "conv" ? layer.outChannels : current[0], height, width];
          if (layer.type === "conv") parameters = layer.outChannels * (current[0] * kernel * kernel + 1);
        }
      }
    } else if (layer.type === "flatten") output = [current.reduce((product, value) => product * value, 1)];
    else if (layer.type === "linear") {
      if (current.length !== 1) error = "Linear 需要一维特征向量，请先添加 Flatten。";
      else {
        output = [layer.units];
        parameters = (current[0] + 1) * layer.units;
      }
    }
    if (!error && (!output.every(Number.isSafeInteger) || !Number.isSafeInteger(totalParameters + parameters) || totalParameters + parameters > 50_000_000)) error = "结构超过本实验的 5,000 万参数上限，请减少通道或使用池化缩小特征图。";
    if (error) {
      errors.push(`第 ${index + 1} 层：${error}`);
      current = null;
      return { input, output: null, parameters: 0, error };
    }
    totalParameters += parameters;
    current = output;
    return { input, output, parameters, error: null };
  });
  const shapeValid = errors.length === 0;
  const classifierReady = shapeValid && current !== null && current.length === 1 && current[0] === 10;
  if (!layers.length) errors.push("网络为空：从左侧添加层，或载入一个示例。" );
  else if (shapeValid && !classifierReady) errors.push("MNIST 分类需要最终输出 [10]。请展平特征，并用 10 单元的 Linear 输出十类 logits。" );
  return { rows, totalParameters, output: current, valid: classifierReady && layers.length > 0, shapeValid, classifierReady, errors };
}

export function toPytorchCode(layers: NetworkLayer[]): string {
  const analysis = analyzeNetwork(layers);
  if (!analysis.valid) throw new Error("请先修复网络结构，使最终输出为 [10]。" );
  const lines = layers.map((layer, index) => {
    const input = analysis.rows[index].input!;
    if (layer.type === "conv") return `    nn.Conv2d(${input[0]}, ${layer.outChannels}, kernel_size=${layer.kernel}, stride=${layer.stride}, padding=${layer.padding}),`;
    if (layer.type === "relu") return "    nn.ReLU(),";
    if (layer.type === "pool") return "    nn.MaxPool2d(kernel_size=2, stride=2),";
    if (layer.type === "flatten") return "    nn.Flatten(start_dim=1),";
    if (layer.type === "linear") return `    nn.Linear(${input[0]}, ${layer.units}),`;
    throw new Error("不支持的网络层。" );
  });
  return `import torch\nfrom torch import nn\n\n# 输入布局 [batch, channels, height, width]；本实验展示的尺寸不含 batch。\n# 此代码仅创建结构；参数尚未训练，输出为十类 logits。\nmodel = nn.Sequential(\n${lines.join("\n")}\n)\nx = torch.zeros(1, 1, 28, 28)\nlogits = model(x)\nprint(logits.shape)  # torch.Size([1, 10])\n# 训练时将 logits 直接送入 nn.CrossEntropyLoss()。\n`;
}
