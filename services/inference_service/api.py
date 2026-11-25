"""FastAPI Inference Service loading trained checkpoints and metadata from MinIO S3 or local."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import torch

from config import ModelConfig
from contracts.state import load_state
from contracts.storage import is_s3_uri, join_uri, list_s3_keys, read_bytes, read_text
from models.multimodal_net import AirbnbMultimodalRegressor

app = FastAPI(title="Airbnb BigData - Inference Service", version="1.0.0")


class PredictRequest(BaseModel):
    numeric: dict[str, float | None] = Field(default_factory=dict)
    categorical: dict[str, str | None] = Field(default_factory=dict)
    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None


class LoadRequest(BaseModel):
    data_dir: str = "/data/demo"
    run_id: str | None = None
    experiment: str | None = None


class _ModelRegistry:
    def __init__(self) -> None:
        self.model: AirbnbMultimodalRegressor | None = None
        self.metadata: dict[str, Any] | None = None
        self.cfg = ModelConfig()
        self.run_id: str | None = None
        self.experiment: str | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _discover_experiment(self, artifacts_uri: str) -> str:
        if is_s3_uri(artifacts_uri):
            keys = list_s3_keys(artifacts_uri)
            ckpts = [k for k in keys if k.endswith("/best_model.pt")]
            if not ckpts:
                raise FileNotFoundError(f"No best_model.pt found under {artifacts_uri}")
            return ckpts[0].removesuffix("/best_model.pt").rsplit("/", 1)[-1]
        root = Path(artifacts_uri)
        ckpts = sorted(root.glob("*/best_model.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No best_model.pt found under {artifacts_uri}")
        return ckpts[0].parent.name

    def load(self, data_dir: str, run_id: str | None = None, experiment: str | None = None) -> dict:
        state = load_state(data_dir, required_status="finalized", run_id=run_id)
        gold_uri = join_uri(state["run_path"], "gold")
        metadata = json.loads(read_text(join_uri(gold_uri, "metadata.json"), encoding="utf-8"))
        artifacts_uri = join_uri(state["run_path"], "artifacts")
        exp = experiment or self._discover_experiment(artifacts_uri)
        ckpt_uri = join_uri(artifacts_uri, exp, "best_model.pt")
        ckpt_bytes = read_bytes(ckpt_uri)

        model = AirbnbMultimodalRegressor(self.cfg, metadata["cat_cardinalities"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(ckpt_bytes)
            tmp_path = Path(tmp.name)
        try:
            state_dict = torch.load(tmp_path, map_location=self.device, weights_only=True)
        finally:
            tmp_path.unlink(missing_ok=True)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()

        self.model = model
        self.metadata = metadata
        self.run_id = state["run_id"]
        self.experiment = exp
        return {"run_id": self.run_id, "experiment": self.experiment, "device": str(self.device)}


registry = _ModelRegistry()


@app.on_event("startup")
def _auto_load_on_startup() -> None:
    data_dir = os.environ.get("INFERENCE_DATA_DIR", "/data/demo")
    run_id = os.environ.get("INFERENCE_RUN_ID") or None
    experiment = os.environ.get("INFERENCE_EXPERIMENT") or None
    try:
        registry.load(data_dir, run_id=run_id, experiment=experiment)
    except Exception:
        # Service starts healthy in standby mode if no finalized run exists yet.
        pass


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "inference-service",
        "model_loaded": registry.model is not None,
        "run_id": registry.run_id,
        "experiment": registry.experiment,
    }


@app.post("/load")
def load_model(req: LoadRequest) -> dict:
    try:
        info = registry.load(req.data_dir, run_id=req.run_id, experiment=req.experiment)
        return {"status": "loaded", **info}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if registry.model is None or registry.metadata is None:
        raise HTTPException(status_code=503, detail="No model loaded. Call POST /load first.")

    cfg = registry.cfg
    preproc = registry.metadata.get("tabular_preprocessor", {})
    medians = preproc.get("num_medians", {})
    means = preproc.get("num_means")
    stds = preproc.get("num_stds")
    cat_labels = preproc.get("cat_labels", {})

    scaled_vals: list[float] = []
    missing_flags: list[float] = []
    for idx, col in enumerate(cfg.num_cols):
        raw = req.numeric.get(col)
        is_missing = raw is None or (isinstance(raw, float) and not math.isfinite(raw))
        missing_flags.append(1.0 if is_missing else 0.0)
        val = float(medians.get(col, 0.0)) if is_missing else float(raw)
        if means and stds and idx < len(means) and stds[idx] > 1e-12:
            val = (val - float(means[idx])) / float(stds[idx])
        scaled_vals.append(val)

    x_num = torch.tensor([scaled_vals + missing_flags], dtype=torch.float32, device=registry.device)

    cat_indices: list[int] = []
    cards = registry.metadata["cat_cardinalities"]
    for idx, col in enumerate(cfg.cat_cols):
        val = str(req.categorical.get(col) or "Unknown").strip() or "Unknown"
        labels = cat_labels.get(col)
        if isinstance(labels, list) and val in labels:
            cat_indices.append(labels.index(val))
        else:
            cat_indices.append(cards[idx] - 1)  # unseen/unknown bucket
    x_cat = torch.tensor([cat_indices], dtype=torch.long, device=registry.device)

    has_txt = req.text_embedding is not None
    has_img = req.image_embedding is not None
    if has_txt and len(req.text_embedding) != cfg.raw_text_dim:
        raise HTTPException(status_code=422, detail=f"text_embedding must have dim {cfg.raw_text_dim}")
    if has_img and len(req.image_embedding) != cfg.raw_image_dim:
        raise HTTPException(status_code=422, detail=f"image_embedding must have dim {cfg.raw_image_dim}")

    x_txt = torch.tensor(
        [req.text_embedding if has_txt else np.zeros(cfg.raw_text_dim, dtype=np.float32).tolist()],
        dtype=torch.float32,
        device=registry.device,
    )
    x_img = torch.tensor(
        [req.image_embedding if has_img else np.zeros(cfg.raw_image_dim, dtype=np.float32).tolist()],
        dtype=torch.float32,
        device=registry.device,
    )
    mask = torch.tensor([[True, has_txt, has_img]], dtype=torch.bool, device=registry.device)

    with torch.no_grad():
        pred, weights = registry.model(x_num, x_cat, x_txt, x_img, available_mask=mask)

    log_price = float(pred[0].item())
    gate = weights[0].detach().cpu().tolist()
    return {
        "run_id": registry.run_id,
        "experiment": registry.experiment,
        "predicted_log_price": round(log_price, 6),
        "predicted_price_usd": round(math.exp(max(-20.0, min(20.0, log_price))), 2),
        "available_mask": [True, has_txt, has_img],
        "gate_weights": {
            "Tabular": round(gate[0] * 100, 2),
            "Text": round(gate[1] * 100, 2),
            "Image": round(gate[2] * 100, 2),
        },
    }
