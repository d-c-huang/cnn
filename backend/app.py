"""Small, stateless numerical experiments used by the first learning phase.

Run from the repository root with:
    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal
import functools

import numpy as np
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.phase2 import router as phase2_router
from backend.mnist import router as mnist_router, service as mnist_service
from backend.inference import router as inference_router
from backend.static_site import mount_static_site
from backend.optimizers import router as optimizers_router
from backend.language import router as language_router
from backend.transformer import router as transformer_router
from backend.lora import router as lora_router
from backend.sft import router as sft_router
from backend.dpo import router as dpo_router
from backend.grpo import router as grpo_router


Number = Annotated[
    float, Field(strict=True, ge=-1_000, le=1_000, allow_inf_nan=False)
]
Weight = Annotated[
    float, Field(strict=True, ge=-1_000_000, le=1_000_000, allow_inf_nan=False)
]
Pixel = Annotated[
    float, Field(strict=True, ge=0, le=255, allow_inf_nan=False)
]
LearningRate = Annotated[
    float, Field(strict=True, ge=0.001, le=1, allow_inf_nan=False)
]


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataRequest(ExperimentRequest):
    pixels: list[list[Pixel]] = Field(min_length=1, max_length=32)
    normalize: bool = True

    @model_validator(mode="after")
    def rectangular_pixels(self) -> DataRequest:
        width = len(self.pixels[0])
        if not 1 <= width <= 32 or any(len(row) != width for row in self.pixels):
            raise ValueError("pixels must be a non-empty rectangular matrix, at most 32 × 32")
        return self


class LinearRequest(ExperimentRequest):
    weight: Number
    bias: Number


class NeuronRequest(ExperimentRequest):
    inputs: tuple[Number, Number]
    weights: tuple[Number, Number]
    bias: Number
    activation: Literal["relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu", "softplus", "silu"]


class LossRequest(ExperimentRequest):
    prediction: Number
    target: Number
    kind: Literal["mse", "mae", "huber", "bce"] = "mse"
    delta: Annotated[float, Field(strict=True, gt=0, le=100, allow_inf_nan=False)] = 1.0

    @model_validator(mode="after")
    def binary_target(self) -> LossRequest:
        if self.kind == "bce" and self.target not in (0, 1):
            raise ValueError("BCE target must be 0 or 1; prediction is a logit")
        return self


class GradientRequest(ExperimentRequest):
    weight: Weight
    learning_rate: LearningRate


SAMPLE_X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
SAMPLE_Y = 2.0 * SAMPLE_X + 0.5

@asynccontextmanager
async def lifespan(_app):
    mnist_service.persist_interrupted()
    yield


app = FastAPI(
    title="知微 AI LAB · 训练与推理 API",
    description="从 NumPy 基础、计算图与 XOR，到真实 MNIST 训练、评估及最佳权重推理。",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(phase2_router)
app.include_router(mnist_router)
app.include_router(inference_router)
app.include_router(optimizers_router)
app.include_router(language_router)
app.include_router(transformer_router)
app.include_router(lora_router)
app.include_router(sft_router)
app.include_router(dpo_router)
app.include_router(grpo_router)

origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default validation response echoes the invalid input, which itself
    # may contain NaN/Infinity and fail JSON serialization. Return safe metadata.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
                for error in exc.errors()
            ]
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine": "numpy", "version": "0.1.0"}


@app.get("/api/docs", include_in_schema=False)
def api_docs():
    return get_swagger_ui_html(openapi_url="/api/openapi.json", title=app.title)


@app.get("/api/openapi.json", include_in_schema=False)
def api_schema():
    return app.openapi()


@functools.lru_cache(maxsize=128)
def _data_experiment(pixels_tuple: tuple, normalize: bool) -> dict:
    matrix = np.asarray(pixels_tuple, dtype=np.float64)
    if normalize:
        matrix = matrix / 255.0
    flat = matrix.ravel(order="C")
    return {
        "matrix": matrix.tolist(),
        "flat": flat.tolist(),
        "shape": list(matrix.shape),
        "count": int(flat.size),
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": float(np.mean(matrix)),
    }

@app.post("/api/data")
def data_experiment(request: DataRequest) -> dict:
    pixels_tuple = tuple(tuple(row) for row in request.pixels)
    return _data_experiment(pixels_tuple, request.normalize)


@functools.lru_cache(maxsize=128)
def _linear_experiment(weight: float, bias: float) -> dict:
    prediction = weight * SAMPLE_X + bias
    error = prediction - SAMPLE_Y
    return {
        "points": [
            {"x": float(x), "y": float(y), "prediction": float(p), "error": float(e)}
            for x, y, p, e in zip(SAMPLE_X, SAMPLE_Y, prediction, error)
        ],
        "mse": float(np.mean(np.square(error))),
    }

@app.post("/api/linear")
def linear_experiment(request: LinearRequest) -> dict:
    return _linear_experiment(request.weight, request.bias)


def activate(values: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(0.0, values)
    if name == "leaky_relu":
        return np.where(values >= 0, values, 0.1 * values)
    if name == "elu":
        return np.where(values >= 0, values, np.expm1(np.minimum(values, 0)))
    if name == "softplus":
        return np.logaddexp(0.0, values)
    if name == "silu":
        return values * activate(values, "sigmoid")
    if name == "sigmoid":
        # Stable for large positive and negative inputs without overflow warnings.
        return np.exp(-np.logaddexp(0.0, -values))
    if name == "tanh":
        return np.tanh(values)
    return values


def activation_derivative(z: float, name: str) -> float:
    if name == "relu":
        return float(z > 0)
    if name == "leaky_relu":
        return 1.0 if z > 0 else 0.1
    if name == "elu":
        return 1.0 if z >= 0 else float(np.exp(z))
    if name == "tanh":
        return float(1 - np.tanh(z) ** 2)
    if name in ("sigmoid", "softplus", "silu"):
        p = float(activate(np.asarray(z), "sigmoid"))
        if name == "sigmoid":
            return p * (1 - p)
        return p if name == "softplus" else p + z * p * (1 - p)
    return 1.0


@functools.lru_cache(maxsize=128)
def _neuron_experiment(in1: float, in2: float, w1: float, w2: float, bias: float, activation: str) -> dict:
    inputs = np.array([in1, in2])
    weights = np.array([w1, w2])
    products = np.multiply(inputs, weights)
    z = float(np.sum(products) + bias)
    curve_left = min(-6.0, z) - 0.5
    curve_right = max(6.0, z) + 0.5
    curve_x = np.unique(np.concatenate([np.linspace(curve_left, curve_right, 241), [z, 0.0]]))
    curve_y = activate(curve_x, activation)
    return {
        "products": products.tolist(),
        "z": z,
        "output": float(activate(np.asarray(z), activation)),
        "derivative": activation_derivative(z, activation),
        "differentiable": not (activation in ("relu", "leaky_relu") and z == 0),
        "curve": [
            {"x": float(x), "y": float(y)} for x, y in zip(curve_x, curve_y)
        ],
    }

@app.post("/api/neuron")
def neuron_experiment(request: NeuronRequest) -> dict:
    return _neuron_experiment(
        request.inputs[0], request.inputs[1],
        request.weights[0], request.weights[1],
        request.bias, request.activation
    )


@functools.lru_cache(maxsize=128)
def _loss_experiment(prediction: float, target: float, kind: str = "mse", delta: float = 1.0) -> dict:
    def evaluate(values):
        error = values - target
        absolute = np.abs(error)
        if kind == "bce":
            return np.maximum(values, 0) - target * values + np.log1p(np.exp(-np.abs(values))), np.exp(-np.logaddexp(0, -values)) - target
        if kind == "mae":
            return absolute, np.sign(error)
        if kind == "huber":
            return np.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta)), np.clip(error, -delta, delta)
        return error**2, 2.0 * error

    left = min(-8.0 if kind == "bce" else target - 5.0, prediction)
    right = max(8.0 if kind == "bce" else target + 5.0, prediction)
    anchors = [0.0 if kind == "bce" else target, prediction]
    if kind == "huber":
        anchors.extend(x for x in (target - delta, target + delta) if left <= x <= right)
    curve_x = np.unique(
        np.concatenate([np.linspace(left, right, 121), anchors])
    )
    curve_y, _ = evaluate(curve_x)
    loss, gradient = evaluate(prediction)
    return {
        "loss": float(loss),
        "gradient": float(gradient),
        "differentiable": not (kind == "mae" and prediction == target),
        "curve": [
            {"x": float(x), "y": float(y)} for x, y in zip(curve_x, curve_y)
        ],
    }

@app.post("/api/loss")
def loss_experiment(request: LossRequest) -> dict:
    return _loss_experiment(request.prediction, request.target, request.kind, request.delta)


@functools.lru_cache(maxsize=128)
def _gradient_experiment(weight: float, learning_rate: float) -> dict:
    errors = weight * SAMPLE_X + 0.5 - SAMPLE_Y
    gradient = float(2.0 * np.mean(errors * SAMPLE_X))
    weight_after = weight - learning_rate * gradient
    loss_before = float(np.mean(np.square(errors)))
    loss_after = float(
        np.mean(np.square(weight_after * SAMPLE_X + 0.5 - SAMPLE_Y))
    )
    return {
        "weight_before": weight,
        "weight_after": float(weight_after),
        "gradient": gradient,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "converged": bool(abs(weight_after - 2.0) < 0.001),
        "diverged": bool(loss_after > loss_before + 1e-12),
    }

@app.post("/api/gradient")
def gradient_experiment(request: GradientRequest) -> dict:
    """One full-batch SGD step; intercept stays fixed at 0.5.

    For the five teaching samples, mean(x²)=2, so L(w)=2(w−2)²
    and dL/dw=4(w−2). NumPy computes these from the samples directly.
    """
    return _gradient_experiment(request.weight, request.learning_rate)


# The frontend mount must come after all API routes.
mount_static_site(app)
