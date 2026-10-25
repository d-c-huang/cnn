"""Small, stateless numerical experiments used by the first learning phase.

Run from the repository root with:
    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

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
    activation: Literal["relu", "sigmoid", "tanh", "linear"]


class LossRequest(ExperimentRequest):
    prediction: Number
    target: Number


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


@app.post("/api/data")
def data_experiment(request: DataRequest) -> dict:
    matrix = np.asarray(request.pixels, dtype=np.float64)
    if request.normalize:
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


@app.post("/api/linear")
def linear_experiment(request: LinearRequest) -> dict:
    prediction = request.weight * SAMPLE_X + request.bias
    error = prediction - SAMPLE_Y
    return {
        "points": [
            {"x": float(x), "y": float(y), "prediction": float(p), "error": float(e)}
            for x, y, p, e in zip(SAMPLE_X, SAMPLE_Y, prediction, error)
        ],
        "mse": float(np.mean(np.square(error))),
    }


def activate(values: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(0.0, values)
    if name == "sigmoid":
        # Stable for large positive and negative inputs without overflow warnings.
        return np.exp(-np.logaddexp(0.0, -values))
    if name == "tanh":
        return np.tanh(values)
    return values


@app.post("/api/neuron")
def neuron_experiment(request: NeuronRequest) -> dict:
    products = np.multiply(request.inputs, request.weights)
    z = float(np.sum(products) + request.bias)
    curve_left = min(-6.0, z) - 0.5
    curve_right = max(6.0, z) + 0.5
    curve_x = np.unique(np.concatenate([np.linspace(curve_left, curve_right, 121), [z]]))
    curve_y = activate(curve_x, request.activation)
    return {
        "products": products.tolist(),
        "z": z,
        "output": float(activate(np.asarray(z), request.activation)),
        "curve": [
            {"x": float(x), "y": float(y)} for x, y in zip(curve_x, curve_y)
        ],
    }


@app.post("/api/loss")
def loss_experiment(request: LossRequest) -> dict:
    error = request.prediction - request.target
    left = min(request.target - 5.0, request.prediction)
    right = max(request.target + 5.0, request.prediction)
    # Include the current prediction and exact minimum even for asymmetric domains.
    curve_x = np.unique(
        np.concatenate([np.linspace(left, right, 121), [request.target, request.prediction]])
    )
    curve_y = np.square(curve_x - request.target)
    return {
        "loss": float(error**2),
        "gradient": float(2.0 * error),
        "curve": [
            {"x": float(x), "y": float(y)} for x, y in zip(curve_x, curve_y)
        ],
    }


@app.post("/api/gradient")
def gradient_experiment(request: GradientRequest) -> dict:
    """One full-batch SGD step; intercept stays fixed at 0.5.

    For the five teaching samples, mean(x²)=2, so L(w)=2(w−2)²
    and dL/dw=4(w−2). NumPy computes these from the samples directly.
    """
    errors = request.weight * SAMPLE_X + 0.5 - SAMPLE_Y
    gradient = float(2.0 * np.mean(errors * SAMPLE_X))
    weight_after = request.weight - request.learning_rate * gradient
    loss_before = float(np.mean(np.square(errors)))
    loss_after = float(
        np.mean(np.square(weight_after * SAMPLE_X + 0.5 - SAMPLE_Y))
    )
    return {
        "weight_before": request.weight,
        "weight_after": float(weight_after),
        "gradient": gradient,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "converged": bool(abs(weight_after - 2.0) < 0.001),
        "diverged": bool(loss_after > loss_before + 1e-12),
    }


# The frontend mount must come after all API routes.
mount_static_site(app)
