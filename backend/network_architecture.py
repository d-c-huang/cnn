"""Validate a small declarative layer language; never execute supplied code.

Shapes exclude the batch dimension. This module is deliberately independent of
PyTorch so request and catalog validation stay lightweight.
"""

from __future__ import annotations

import re


MAX_LAYERS = 16
MAX_PARAMETERS = 50_000_000


def _choice(layer: dict, key: str, choices: tuple[int, ...]) -> int:
    value = layer.get(key)
    if type(value) is not int or value not in choices:
        raise ValueError(f"{key} 必须是 {', '.join(map(str, choices))} 中的整数。")
    return value


def analyze_architecture(value) -> dict:
    """Return a canonical, fully validated ten-class network and shape trace."""
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_LAYERS:
        raise ValueError("自定义网络需要 1–16 层。")
    layers, rows = [], []
    ids = set()
    shape = [1, 28, 28]
    parameters = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index + 1} 层必须是一个层配置。")
        kind = raw.get("type")
        if type(kind) is not str or kind not in {"conv", "relu", "pool", "flatten", "linear"}:
            raise ValueError(f"第 {index + 1} 层类型不受支持。")
        expected = {"type", "id"}
        if kind == "conv":
            expected |= {"outChannels", "kernel", "stride", "padding"}
        elif kind == "linear":
            expected.add("units")
        if set(raw) - expected or expected - {"id"} - set(raw):
            raise ValueError(f"第 {index + 1} 层包含未知字段或缺少必要参数。")
        identifier = raw.get("id", f"layer-{index + 1}")
        if type(identifier) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) or identifier in ids:
            raise ValueError(f"第 {index + 1} 层 id 无效或重复。")
        ids.add(identifier)
        layer = {"id": identifier, "type": kind}
        before = list(shape)
        count = 0
        if kind in {"conv", "pool"}:
            if len(shape) != 3:
                raise ValueError(f"第 {index + 1} 层需要 [通道, 高, 宽]，不能接在展平后的向量上。")
            if kind == "conv":
                channels = _choice(raw, "outChannels", (8, 16, 32, 64))
                kernel = _choice(raw, "kernel", (3, 5))
                stride = _choice(raw, "stride", (1, 2))
                padding = _choice(raw, "padding", (0, 1, 2))
                layer.update(outChannels=channels, kernel=kernel, stride=stride, padding=padding)
                count = channels * (shape[0] * kernel * kernel + 1)
            else:
                channels, kernel, stride, padding = shape[0], 2, 2, 0
            height = (shape[1] + 2 * padding - kernel) // stride + 1
            width = (shape[2] + 2 * padding - kernel) // stride + 1
            if height < 1 or width < 1:
                raise ValueError(f"第 {index + 1} 层的卷积核或池化窗口超过有效输入尺寸。")
            shape = [channels, height, width]
        elif kind == "flatten":
            size = 1
            for dimension in shape:
                size *= dimension
            shape = [size]
        elif kind == "linear":
            units = _choice(raw, "units", (10, 32, 64, 128))
            layer["units"] = units
            if len(shape) != 1:
                raise ValueError(f"第 {index + 1} 层 Linear 需要向量输入，请先添加 Flatten。")
            count = (shape[0] + 1) * units
            shape = [units]
        parameters += count
        if parameters > MAX_PARAMETERS:
            raise ValueError("自定义网络超过 5,000 万参数上限，请减少通道或添加池化层。")
        layers.append(layer)
        rows.append({"input": before, "output": list(shape), "parameters": count})
    if shape != [10]:
        raise ValueError("自定义 MNIST 网络最终必须输出 [10]，请添加十个输出单元的 Linear。")
    return {"layers": layers, "rows": rows, "total_parameters": parameters, "output": list(shape)}


def validate_architecture(value) -> list[dict]:
    return analyze_architecture(value)["layers"]


def build_custom_model(value, nn):
    analysis = analyze_architecture(value)
    modules = []
    for layer, row in zip(analysis["layers"], analysis["rows"]):
        kind = layer["type"]
        if kind == "conv":
            modules.append(nn.Conv2d(row["input"][0], layer["outChannels"], layer["kernel"],
                                     stride=layer["stride"], padding=layer["padding"]))
        elif kind == "relu":
            modules.append(nn.ReLU())
        elif kind == "pool":
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif kind == "flatten":
            modules.append(nn.Flatten(start_dim=1))
        else:
            modules.append(nn.Linear(row["input"][0], layer["units"]))
    return nn.Sequential(*modules)
