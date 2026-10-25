"""Validated HTTP interface for the third learning phase."""

from typing import Annotated, Literal

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.mnist_engine import service


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrepareRequest(Request):
    pass


class SplitRequest(Request):
    seed: Annotated[int, Field(strict=True, ge=0, le=9999)] = 7
    validation_ratio: Literal[0.1, 0.2] = 0.1
    train_limit: Literal[6000, 12000] | None = None


class RunConfig(SplitRequest):
    model: Literal["mlp", "cnn"] = "mlp"
    epochs: Annotated[int, Field(strict=True, ge=1, le=20)] = 3
    batch_size: Literal[32, 64, 128, 256] = 128
    learning_rate: Annotated[float, Field(strict=True, ge=0.00001, le=0.1, allow_inf_nan=False)] = 0.001


router = APIRouter(prefix="/api/mnist", tags=["第三期：MNIST 真实训练"])


@router.get("/status")
def status():
    return service.status()


@router.post("/prepare")
def prepare(_request: PrepareRequest):
    return service.prepare()


@router.post("/split")
def split(request: SplitRequest):
    return service.split(request.model_dump())


@router.get("/runs")
def list_runs():
    return service.list_runs()


@router.post("/runs")
def start_run(request: RunConfig):
    return service.create_run(request.model_dump())


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    return service.get_run(run_id)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    return service.cancel(run_id)


@router.get("/runs/{run_id}/errors")
def get_errors(
    run_id: str,
    true_label: Annotated[int | None, Query(ge=0, le=9)] = None,
    predicted_label: Annotated[int | None, Query(ge=0, le=9)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=48)] = 24,
):
    return service.errors(run_id, true_label, predicted_label, offset, limit)


@router.get("/runs/{run_id}/checkpoint")
def checkpoint(run_id: str):
    return FileResponse(service.checkpoint_path(run_id), media_type="application/octet-stream", filename=f"mnist-{run_id[:8]}-best.pt")


@router.get("/runs/{run_id}/report")
def report(run_id: str):
    return service.get_run(run_id)
