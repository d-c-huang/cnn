"""Inspectable scalar computation graph and a hand-differentiated XOR network.

All endpoints are stateless. The caller owns its parameters, so stepping one
experiment cannot change another browser's model. Calculations use float64.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


Scalar = Annotated[float, Field(strict=True, ge=-10, le=10, allow_inf_nan=False)]
Parameter = Annotated[
    float, Field(strict=True, ge=-1000, le=1000, allow_inf_nan=False)
]
GraphRate = Annotated[float, Field(strict=True, ge=0.001, le=1, allow_inf_nan=False)]
Rate = Annotated[float, Field(strict=True, ge=0.001, le=3, allow_inf_nan=False)]
Iteration = Annotated[int, Field(strict=True, ge=0, le=20_000)]
SampleIndex = Annotated[int, Field(strict=True, ge=0, le=3)]
Vector4 = tuple[Parameter, Parameter, Parameter, Parameter]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphRequest(Request):
    inputs: tuple[Scalar, Scalar]
    weights: tuple[Scalar, Scalar]
    bias: Scalar
    target: Scalar
    learning_rate: GraphRate


class Parameters(Request):
    w1: tuple[Vector4, Vector4]
    b1: Vector4
    w2: Vector4
    b2: Parameter


class InitRequest(Request):
    seed: Annotated[int, Field(strict=True, ge=0, le=9999)] = 7


class InspectRequest(Request):
    parameters: Parameters
    iteration: Iteration = 0
    selected_sample: SampleIndex = 0


class StepRequest(InspectRequest):
    learning_rate: Rate
    steps: Annotated[int, Field(strict=True, ge=1, le=100)] = 1

    @model_validator(mode="after")
    def limit_total_steps(self) -> StepRequest:
        if self.iteration + self.steps > 20_000:
            raise ValueError("iteration + steps must not exceed 20000")
        return self


router = APIRouter(prefix="/api", tags=["第二期：计算图与神经网络"])


@router.post("/graph")
def computation_graph(request: GraphRequest) -> dict:
    x1, x2 = request.inputs
    w1, w2 = request.weights
    values = {
        "x1": x1,
        "w1": w1,
        "p1": x1 * w1,
        "x2": x2,
        "w2": w2,
        "p2": x2 * w2,
    }
    values["s"] = values["p1"] + values["p2"]
    values["b"] = request.bias
    values["z"] = values["s"] + request.bias
    values["a"] = float(np.tanh(values["z"]))
    values["target"] = request.target
    values["e"] = values["a"] - request.target
    values["loss"] = 0.5 * values["e"] ** 2

    # Each edge stores ∂target/∂source evaluated at these exact forward values.
    edge_derivatives = [
        ("x1", "p1", w1), ("w1", "p1", x1),
        ("x2", "p2", w2), ("w2", "p2", x2),
        ("p1", "s", 1.0), ("p2", "s", 1.0),
        ("s", "z", 1.0), ("b", "z", 1.0),
        ("z", "a", 1.0 - values["a"] ** 2),
        ("a", "e", 1.0), ("target", "e", -1.0),
        ("e", "loss", values["e"]),
    ]
    order = list(values)
    gradients = dict.fromkeys(order, 0.0)
    gradients["loss"] = 1.0
    for target in reversed(order):
        for source, destination, local_derivative in edge_derivatives:
            if destination == target:
                gradients[source] += gradients[target] * local_derivative

    descriptions = {
        "x1": ("x₁", "input", "输入特征 x₁"),
        "w1": ("w₁", "parameter", "第一路权重 w₁"),
        "p1": ("p₁", "multiply", "x₁ × w₁"),
        "x2": ("x₂", "input", "输入特征 x₂"),
        "w2": ("w₂", "parameter", "第二路权重 w₂"),
        "p2": ("p₂", "multiply", "x₂ × w₂"),
        "s": ("s", "add", "p₁ + p₂"),
        "b": ("b", "parameter", "偏置 b"),
        "z": ("z", "add", "s + b"),
        "a": ("a", "tanh", "tanh(z)"),
        "target": ("y", "target", "真实标签 y"),
        "e": ("e", "subtract", "a − y"),
        "loss": ("L", "square", "½ × e²"),
    }
    updates = [
        {"name": name, "before": values[name], "gradient": gradients[name],
         "after": values[name] - request.learning_rate * gradients[name]}
        for name in ["w1", "w2", "b"]
    ]
    next_weights = [updates[0]["after"], updates[1]["after"]]
    next_bias = updates[2]["after"]
    output_after = float(np.tanh(np.dot(request.inputs, next_weights) + next_bias))
    return {
        "nodes": [
            {"id": name, "label": descriptions[name][0], "op": descriptions[name][1],
             "value": float(values[name]), "gradient": float(gradients[name]),
             "expression": descriptions[name][2], "trainable": name in {"w1", "w2", "b"}}
            for name in order
        ],
        "edges": [
            {"source": source, "target": target, "local_derivative": float(derivative),
             "contribution": float(gradients[target] * derivative)}
            for source, target, derivative in edge_derivatives
        ],
        "forward_order": order,
        "backward_order": list(reversed(order)),
        "output": values["a"], "loss": values["loss"],
        "output_after": output_after,
        "loss_after": float(0.5 * (output_after - request.target) ** 2),
        "updates": updates,
        "next": {"inputs": list(request.inputs), "weights": next_weights,
                 "bias": next_bias, "target": request.target,
                 "learning_rate": request.learning_rate},
    }


XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
XOR_Y = np.array([0, 1, 1, 0], dtype=np.float64)


def parameter_arrays(parameters: Parameters | dict) -> dict[str, np.ndarray]:
    raw = parameters.model_dump() if isinstance(parameters, Parameters) else parameters
    return {name: np.asarray(raw[name], dtype=np.float64) for name in ["w1", "b1", "w2", "b2"]}


def serialize_arrays(arrays: dict[str, np.ndarray]) -> dict:
    return {name: value.tolist() for name, value in arrays.items()}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def forward(parameters: dict[str, np.ndarray], inputs: np.ndarray = XOR_X) -> dict:
    z1 = inputs @ parameters["w1"] + parameters["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ parameters["w2"] + parameters["b2"]
    output = sigmoid(z2)
    return {"z1": z1, "a1": a1, "z2": z2, "output": output}


def loss_and_gradients(parameters: dict[str, np.ndarray], inputs=XOR_X, targets=XOR_Y) -> tuple[float, dict, dict]:
    cache = forward(parameters, inputs)
    # BCE is evaluated from logits rather than clipped probabilities, so the
    # displayed objective and its analytic gradient agree even near saturation.
    losses = np.logaddexp(0.0, cache["z2"]) - targets * cache["z2"]
    delta2 = (cache["output"] - targets) / len(inputs)
    delta1 = delta2[:, None] * parameters["w2"][None, :] * (1.0 - cache["a1"] ** 2)
    gradients = {
        "w1": inputs.T @ delta1,
        "b1": delta1.sum(axis=0),
        "w2": cache["a1"].T @ delta2,
        "b2": np.asarray(delta2.sum()),
    }
    return float(np.mean(losses)), gradients, cache


def snapshot(parameters: dict[str, np.ndarray], iteration: int, selected_sample: int) -> dict:
    loss, gradients, cache = loss_and_gradients(parameters)
    index = selected_sample
    sample_loss, sample_gradients, _ = loss_and_gradients(
        parameters, XOR_X[index:index + 1], XOR_Y[index:index + 1]
    )
    classifications = (cache["output"] >= 0.5).astype(int)
    grid_size = 25
    coordinates = np.linspace(-0.25, 1.25, grid_size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    grid_inputs = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_probabilities = forward(parameters, grid_inputs)["output"]
    return {
        "parameters": serialize_arrays(parameters),
        "iteration": iteration, "loss": loss,
        "accuracy": float(np.mean(classifications == XOR_Y)),
        "samples": [
            {"inputs": inputs.tolist(), "target": int(target),
             "prediction": int(prediction), "probability": float(probability)}
            for inputs, target, prediction, probability in zip(XOR_X, XOR_Y, classifications, cache["output"])
        ],
        "selected": {
            "index": index, "inputs": XOR_X[index].tolist(), "target": int(XOR_Y[index]),
            "z1": cache["z1"][index].tolist(), "a1": cache["a1"][index].tolist(),
            "z2": float(cache["z2"][index]), "output": float(cache["output"][index]),
            "loss": sample_loss,
        },
        "gradients": serialize_arrays(gradients),
        "sample_gradients": serialize_arrays(sample_gradients),
        "grid": {"size": grid_size, "x_min": -0.25, "x_max": 1.25,
                 "y_min": -0.25, "y_max": 1.25,
                 "probabilities": grid_probabilities.tolist()},
    }


@router.post("/xor/init")
def initialize_xor(request: InitRequest) -> dict:
    rng = np.random.default_rng(request.seed)
    parameters = {
        "w1": rng.normal(0.0, 1.0, (2, 4)),
        "b1": np.zeros(4, dtype=np.float64),
        "w2": rng.normal(0.0, 1.0, 4),
        "b2": np.asarray(0.0),
    }
    return snapshot(parameters, 0, 0)


@router.post("/xor/inspect")
def inspect_xor(request: InspectRequest) -> dict:
    return snapshot(parameter_arrays(request.parameters), request.iteration, request.selected_sample)


def flattened_updates(before: dict[str, np.ndarray], gradients: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> list[dict]:
    entries = []
    for name in ["w1", "b1", "w2", "b2"]:
        for index in np.ndindex(before[name].shape):
            suffix = f"[{','.join(map(str, index))}]" if index else ""
            entries.append({
                "name": name + suffix, "before": float(before[name][index]),
                "gradient": float(gradients[name][index]), "after": float(after[name][index]),
            })
    return entries


@router.post("/xor/step")
def step_xor(request: StepRequest) -> dict:
    parameters = parameter_arrays(request.parameters)
    history = []
    updates = []
    for step in range(request.steps):
        _, gradients, _ = loss_and_gradients(parameters)
        updated = {name: parameters[name] - request.learning_rate * gradients[name] for name in parameters}
        if any(not np.isfinite(value).all() or np.max(np.abs(value)) > 1000 for value in updated.values()):
            raise HTTPException(status_code=422, detail="参数更新超出数值范围，请降低学习率或重新初始化。")
        if step == request.steps - 1:
            updates = flattened_updates(parameters, gradients, updated)
        parameters = updated
        loss, _, cache = loss_and_gradients(parameters)
        history.append({"step": request.iteration + step + 1, "loss": loss,
                        "accuracy": float(np.mean((cache["output"] >= 0.5) == XOR_Y))})
    return {"snapshot": snapshot(parameters, request.iteration + request.steps, request.selected_sample),
            "history": history, "updates": updates}
