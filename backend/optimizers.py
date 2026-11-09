"""Deterministic, stateless optimizer comparisons on small teaching objectives."""

from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/api/optimizers", tags=["optimizer experiments"])
Optimizer = Literal["sgd", "momentum", "adagrad", "rmsprop", "adam"]
OPTIMIZERS = ("sgd", "momentum", "adagrad", "rmsprop", "adam")
Rate = Annotated[float, Field(strict=True, ge=0.001, le=1, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(strict=True, ge=-6, le=6, allow_inf_nan=False)]


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: Literal["quadratic", "valley"] = "quadratic"
    start: tuple[Coordinate, Coordinate] = (-2.0, 2.0)
    steps: Annotated[int, Field(strict=True, ge=1, le=200)] = 80
    learning_rate: Rate = 0.1
    rates: dict[Optimizer, Rate] = Field(default_factory=dict)


def objective_value(position: np.ndarray, objective: str) -> tuple[float, np.ndarray]:
    if objective == "quadratic":
        error = position[0] - 2.0
        return float(2 * error**2), np.array([4 * error, 0.0])
    error = position - np.array([2.0, -1.0])
    return float(0.5 * (error[0] ** 2 + 20 * error[1] ** 2)), error * [1.0, 20.0]


def simulate(request: ComparisonRequest, optimizer: str) -> dict:
    rate = request.rates.get(optimizer, request.learning_rate)
    position = np.array(request.start, dtype=np.float64)
    if request.objective == "quadratic":
        position[1] = 0.0  # The teaching objective has only one trainable coordinate.
    first, second = np.zeros(2), np.zeros(2)
    loss, _ = objective_value(position, request.objective)
    history = [{"step": 0, "position": position.tolist(), "loss": loss,
                "gradient": None, "update": None, "first": first.tolist(), "second": second.tolist()}]
    first_hit = 0 if loss <= 1e-6 else None
    status = "completed"
    for step in range(1, request.steps + 1):
        _, gradient = objective_value(position, request.objective)
        if optimizer == "sgd":
            direction = gradient
        elif optimizer == "momentum":
            first = 0.9 * first + gradient
            direction = first
        elif optimizer == "adagrad":
            second = second + gradient**2
            direction = gradient / (np.sqrt(second) + 1e-8)
        elif optimizer == "rmsprop":
            second = 0.9 * second + 0.1 * gradient**2
            direction = gradient / (np.sqrt(second) + 1e-8)
        else:
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            corrected_first = first / (1 - 0.9**step)
            corrected_second = second / (1 - 0.999**step)
            direction = corrected_first / (np.sqrt(corrected_second) + 1e-8)
        update = -rate * direction
        candidate = position + update
        loss, _ = objective_value(candidate, request.objective)
        if not np.isfinite(candidate).all() or not np.isfinite(loss):
            status = "limit"
            break
        position = candidate
        history.append({"step": step, "position": position.tolist(), "loss": loss,
                        "gradient": gradient.tolist(), "update": update.tolist(),
                        "first": first.tolist(), "second": second.tolist()})
        if first_hit is None and loss <= 1e-6:
            first_hit = step
        if loss > 1e12 or np.max(np.abs(position)) > 1e6:
            status = "limit"
            break
    return {"optimizer": optimizer, "learning_rate": rate, "status": status,
            "first_hit": first_hit, "history": history}


@router.post("/compare")
def compare_optimizers(request: ComparisonRequest) -> dict:
    return {"config": request.model_dump(), "threshold": 1e-6,
            "optimum": [2.0, 0.0] if request.objective == "quadratic" else [2.0, -1.0],
            "runs": [simulate(request, name) for name in OPTIMIZERS]}
