"""FastAPI HTTP interface for the PyTorch Model Service."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contracts.jobs import JobManager
from contracts.storage import StorageConfig

app = FastAPI(title="Airbnb BigData - Model Service", version="1.0.0")
jobs = JobManager("model-service", max_workers=1)


class DemoInputRequest(BaseModel):
    data_dir: str = "/data/demo"


class DemoEmbedRequest(BaseModel):
    data_dir: str = "/data/demo"
    run_id: str


class EmbedRequest(BaseModel):
    data_dir: str = "/data"
    run_id: str | None = None
    workers: int = Field(default=4, ge=1, le=16)


class TrainRequest(BaseModel):
    data_dir: str = "/data/demo"
    run_id: str | None = None
    modalities: str = "tabular,text,image"
    cohort: str = "eligible"
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=32, ge=1)
    seed: int = 42
    skip_importance: bool = True


class CompareRequest(BaseModel):
    mode: str = Field(default="real", pattern="^(real|mock)$")
    data_dir: str = "/data/demo"
    run_id: str | None = None
    sets: str = Field(default="core", pattern="^(core|all7)$")
    seeds: str = "42"
    cohort: str = Field(default="complete", pattern="^(complete|eligible)$")
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=32, ge=1)
    aux_loss_weight: float = Field(default=0.0, ge=0.0)
    modality_dropout: float = Field(default=0.2, ge=0.0, le=1.0)


class DiagnoseRequest(BaseModel):
    data_dir: str = "/data/demo"
    run_id: str | None = None
    limit: int = Field(default=5000, ge=1)
    out: str | None = None


@app.get("/health")
def health() -> dict:
    cfg = StorageConfig.from_env()
    return {
        "status": "ok",
        "service": "model-service",
        "storage_backend": cfg.backend,
    }


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return jobs.list_all()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    rec = jobs.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return rec


@app.post("/jobs/demo-input", status_code=202)
def submit_demo_input(req: DemoInputRequest) -> dict:
    return jobs.submit("demo-input", "services.model_service.entrypoint", ["demo-input", "--data-dir", req.data_dir])


@app.post("/jobs/demo-embed", status_code=202)
def submit_demo_embed(req: DemoEmbedRequest) -> dict:
    return jobs.submit(
        "demo-embed",
        "services.model_service.entrypoint",
        ["demo-embed", "--data-dir", req.data_dir, "--run-id", req.run_id],
    )


@app.post("/jobs/embed", status_code=202)
def submit_embed(req: EmbedRequest) -> dict:
    args = ["embed", "--data-dir", req.data_dir, "--workers", str(req.workers)]
    if req.run_id:
        args.extend(["--run-id", req.run_id])
    return jobs.submit("embed", "services.model_service.entrypoint", args)


@app.post("/jobs/train", status_code=202)
def submit_train(req: TrainRequest) -> dict:
    args = [
        "train",
        "--data-dir", req.data_dir,
        "--modalities", req.modalities,
        "--cohort", req.cohort,
        "--epochs", str(req.epochs),
        "--batch-size", str(req.batch_size),
        "--seed", str(req.seed),
    ]
    if req.run_id:
        args.extend(["--run-id", req.run_id])
    if req.skip_importance:
        args.append("--skip-importance")
    return jobs.submit("train", "services.model_service.entrypoint", args)


@app.post("/jobs/compare", status_code=202)
def submit_compare(req: CompareRequest) -> dict:
    args = [
        "compare",
        "--mode", req.mode,
        "--data-dir", req.data_dir,
        "--sets", req.sets,
        "--seeds", req.seeds,
        "--cohort", req.cohort,
        "--epochs", str(req.epochs),
        "--batch-size", str(req.batch_size),
        "--aux-loss-weight", str(req.aux_loss_weight),
        "--modality-dropout", str(req.modality_dropout),
    ]
    if req.run_id:
        args.extend(["--run-id", req.run_id])
    return jobs.submit("compare", "services.model_service.entrypoint", args)


@app.post("/jobs/diagnose", status_code=202)
def submit_diagnose(req: DiagnoseRequest) -> dict:
    args = ["diagnose", "--data-dir", req.data_dir, "--limit", str(req.limit)]
    if req.run_id:
        args.extend(["--run-id", req.run_id])
    if req.out:
        args.extend(["--out", req.out])
    return jobs.submit("diagnose", "services.model_service.entrypoint", args)
