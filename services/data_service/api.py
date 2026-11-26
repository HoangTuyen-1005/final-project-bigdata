"""FastAPI HTTP interface for the Spark Data Service."""
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contracts.jobs import JobManager
from contracts.storage import StorageConfig

app = FastAPI(title="Airbnb BigData - Data Service", version="1.0.0")
jobs = JobManager("data-service", max_workers=1)


class PrepareRequest(BaseModel):
    data_dir: str = "/data/demo"
    run_id: str | None = None
    write_partitions: int = Field(default=2, ge=1)
    explain: bool = False
    sources: list[str] = Field(default_factory=list)


class FinalizeRequest(BaseModel):
    data_dir: str = "/data/demo"
    run_id: str | None = None
    write_partitions: int = Field(default=2, ge=1)
    explain: bool = False


class BenchmarkRequest(BaseModel):
    rows: int = Field(default=20000, ge=100)
    partitions: int = Field(default=2, ge=1)
    out: str = "/data/benchmark/20k"
    explain: bool = False


@app.get("/health")
def health() -> dict:
    cfg = StorageConfig.from_env()
    return {
        "status": "ok",
        "service": "data-service",
        "storage_backend": cfg.backend,
        "spark_master": os.environ.get("SPARK_MASTER", "local[2]"),
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


@app.post("/jobs/prepare", status_code=202)
def submit_prepare(req: PrepareRequest) -> dict:
    args = ["prepare", "--data-dir", req.data_dir, "--write-partitions", str(req.write_partitions)]
    if req.run_id:
        args.extend(["--run-id", req.run_id])
    if req.explain:
        args.append("--explain")
    for src in req.sources:
        args.extend(["--source", src])
    return jobs.submit("prepare", "services.data_service.entrypoint", args)


@app.post("/jobs/finalize", status_code=202)
def submit_finalize(req: FinalizeRequest) -> dict:
    args = ["finalize", "--data-dir", req.data_dir, "--write-partitions", str(req.write_partitions)]
    if req.run_id:
        args.extend(["--run-id", req.run_id])
    if req.explain:
        args.append("--explain")
    return jobs.submit("finalize", "services.data_service.entrypoint", args)


@app.post("/jobs/benchmark", status_code=202)
def submit_benchmark(req: BenchmarkRequest) -> dict:
    args = [
        "benchmark",
        "--rows", str(req.rows),
        "--partitions", str(req.partitions),
        "--out", req.out,
    ]
    if req.explain:
        args.append("--explain")
    return jobs.submit("benchmark", "services.data_service.entrypoint", args)
