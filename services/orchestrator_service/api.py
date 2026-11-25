"""Centralized FastAPI Orchestrator & API Gateway managing Data, Model, and Inference services."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contracts.state import validate_run_id

DATA_URL = os.environ.get("DATA_SERVICE_URL", "http://data-service:8001").rstrip("/")
MODEL_URL = os.environ.get("MODEL_SERVICE_URL", "http://model-service:8002").rstrip("/")
INFERENCE_URL = os.environ.get("INFERENCE_SERVICE_URL", "http://inference-service:8003").rstrip("/")

app = FastAPI(title="Airbnb BigData - SOA Orchestrator & Gateway", version="1.0.0")
_executor = ThreadPoolExecutor(max_workers=2)
_lock = threading.Lock()
_pipelines: dict[str, dict] = {}


class PipelineRunRequest(BaseModel):
    mode: str = Field(default="demo", pattern="^(demo|real)$")
    run_id: str | None = None
    epochs: int = Field(default=1, ge=1)
    cohort: str = Field(default="eligible", pattern="^(complete|eligible)$")
    modalities: str = "tabular,text,image"
    auto_load_inference: bool = True


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} from {method} {url}: {detail}") from None
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error calling {method} {url}: {err.reason}") from None


def _run_remote_job(run_id: str, stage: str, base_url: str, path: str, payload: dict) -> None:
    with _lock:
        _pipelines[run_id]["current_stage"] = stage
        _pipelines[run_id]["stages"][stage] = {
            "status": "submitting",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    submitted = _http_json("POST", f"{base_url}{path}", payload)
    job_id = submitted["job_id"]
    with _lock:
        _pipelines[run_id]["stages"][stage]["job_id"] = job_id
        _pipelines[run_id]["stages"][stage]["status"] = "running"

    while True:
        status = _http_json("GET", f"{base_url}/jobs/{job_id}")
        state = status["status"]
        if state == "succeeded":
            with _lock:
                _pipelines[run_id]["stages"][stage]["status"] = "succeeded"
                _pipelines[run_id]["stages"][stage]["finished_at"] = datetime.now(timezone.utc).isoformat()
            return
        if state == "failed":
            tail = "\n".join(status.get("log_tail", []))
            with _lock:
                _pipelines[run_id]["stages"][stage]["status"] = "failed"
                _pipelines[run_id]["stages"][stage]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _pipelines[run_id]["stages"][stage]["error"] = tail
            raise RuntimeError(f"Stage '{stage}' (job_id={job_id}) failed:\n{tail}")
        time.sleep(2.0)


def _execute_pipeline(run_id: str, req: PipelineRunRequest) -> None:
    data_dir = "/data/demo" if req.mode == "demo" else "/data"
    with _lock:
        _pipelines[run_id]["status"] = "running"
        _pipelines[run_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        if req.mode == "demo":
            _run_remote_job(run_id, "demo-input", MODEL_URL, "/jobs/demo-input", {"data_dir": data_dir})
        _run_remote_job(run_id, "prepare", DATA_URL, "/jobs/prepare", {"data_dir": data_dir, "run_id": run_id})
        if req.mode == "demo":
            _run_remote_job(
                run_id, "demo-embed", MODEL_URL, "/jobs/demo-embed", {"data_dir": data_dir, "run_id": run_id}
            )
        else:
            _run_remote_job(run_id, "embed", MODEL_URL, "/jobs/embed", {"data_dir": data_dir, "run_id": run_id})
        _run_remote_job(run_id, "finalize", DATA_URL, "/jobs/finalize", {"data_dir": data_dir, "run_id": run_id})
        _run_remote_job(
            run_id,
            "train",
            MODEL_URL,
            "/jobs/train",
            {
                "data_dir": data_dir,
                "run_id": run_id,
                "cohort": req.cohort,
                "modalities": req.modalities,
                "epochs": req.epochs,
                "skip_importance": req.mode == "demo",
            },
        )
        if req.auto_load_inference:
            with _lock:
                _pipelines[run_id]["current_stage"] = "load-inference"
            loaded = _http_json("POST", f"{INFERENCE_URL}/load", {"data_dir": data_dir, "run_id": run_id})
            with _lock:
                _pipelines[run_id]["stages"]["load-inference"] = {"status": "succeeded", "details": loaded}

        with _lock:
            _pipelines[run_id]["status"] = "succeeded"
            _pipelines[run_id]["current_stage"] = None
            _pipelines[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with _lock:
            _pipelines[run_id]["status"] = "failed"
            _pipelines[run_id]["error"] = str(exc)
            _pipelines[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health() -> dict:
    downstream = {}
    for name, url in (("data-service", DATA_URL), ("model-service", MODEL_URL), ("inference-service", INFERENCE_URL)):
        try:
            downstream[name] = _http_json("GET", f"{url}/health", timeout=3.0)
        except Exception as exc:
            downstream[name] = {"status": "unreachable", "error": str(exc)}
    all_ok = all(item.get("status") == "ok" for item in downstream.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "service": "orchestrator-service",
        "downstream": downstream,
    }


@app.post("/pipelines/run", status_code=202)
def start_pipeline(req: PipelineRunRequest) -> dict:
    try:
        run_id = uuid.uuid4().hex if req.run_id is None else validate_run_id(req.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _lock:
        if run_id in _pipelines:
            raise HTTPException(status_code=409, detail=f"Pipeline run already exists: {run_id}")
        _pipelines[run_id] = {
            "run_id": run_id,
            "mode": req.mode,
            "status": "queued",
            "current_stage": None,
            "stages": {},
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
    _executor.submit(_execute_pipeline, run_id, req)
    return dict(_pipelines[run_id])


@app.get("/pipelines")
def list_pipelines() -> list[dict]:
    with _lock:
        return [dict(v) for v in _pipelines.values()]


@app.get("/pipelines/{run_id}")
def get_pipeline(run_id: str) -> dict:
    with _lock:
        rec = _pipelines.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")
        return dict(rec)


@app.post("/predict")
def gateway_predict(payload: dict) -> dict:
    try:
        return _http_json("POST", f"{INFERENCE_URL}/predict", payload, timeout=15.0)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
