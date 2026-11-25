"""Manage pipeline state and artifact names for each ETL run (local or S3/MinIO)."""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Collection
from pathlib import Path

from config import MODALITY_NAMES, ModelConfig
from contracts.storage import (
    StorageConfig,
    is_s3_uri,
    join_uri,
    read_text,
    to_s3a_uri,
    uri_exists,
    write_text,
)

DEFAULT_DATA_DIR = Path("data")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def resolve_data_root(data_dir: str | Path = DEFAULT_DATA_DIR) -> str | Path:
    """Resolve local path or S3/MinIO URI based on argument or STORAGE_BACKEND."""
    text = str(data_dir)
    if is_s3_uri(text):
        return to_s3a_uri(text).rstrip("/")
    cfg = StorageConfig.from_env()
    if cfg.backend == "s3" and text.startswith("/data"):
        sub = text.removeprefix("/data").strip("/")
        base = f"s3a://{cfg.bucket}"
        return f"{base}/{sub}" if sub else base
    return Path(data_dir).resolve()


def get_paths(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[Path | str, Path | str]:
    root = resolve_data_root(data_dir)
    if is_s3_uri(root):
        return str(root), join_uri(root, "pipeline_state.json")
    local_root = Path(root)
    return local_root, local_root / "pipeline_state.json"


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens, starting with a letter or digit.")
    return run_id


def load_state(
    data_dir: str | Path = DEFAULT_DATA_DIR, required_status: str | None = None,
    *, run_id: str | None = None,
) -> dict:
    root, active_file = get_paths(data_dir)
    active = None
    if run_id is None:
        if not uri_exists(active_file):
            raise FileNotFoundError(f"Missing {active_file}. Run 'python -m spark_etl.run_etl prepare' first.")
        active = json.loads(read_text(active_file, encoding="utf-8"))
        run_id = active.get("run_id")
        if run_id is None:
            raise ValueError(f"Invalid pipeline state: {active_file}")
    run_dir = f"runs/{validate_run_id(run_id)}"
    state_file = join_uri(root, run_dir, "state.json")
    if uri_exists(state_file):
        state = json.loads(read_text(state_file, encoding="utf-8"))
    else:
        if active is None and uri_exists(active_file):
            active = json.loads(read_text(active_file, encoding="utf-8"))
        if active is None or active.get("run_id") != run_id:
            raise FileNotFoundError(f"Missing state for run {run_id!r} at {state_file}.")
        state = active
    if state.get("run_id") != run_id or state.get("run_dir") != run_dir:
        raise ValueError(f"State for {run_id!r} contains an inconsistent run ID or directory.")
    if is_s3_uri(root):
        state["run_path"] = join_uri(root, run_dir)
    else:
        state["run_path"] = Path(root) / run_dir
    if required_status and state.get("status") != required_status:
        raise RuntimeError(f"Pipeline status is {state.get('status')!r}; expected {required_status!r}.")
    return state


def _write_state(path: Path | str, state: dict) -> None:
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    if is_s3_uri(path):
        write_text(path, payload, encoding="utf-8")
        return
    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_path.parent, prefix=f".{local_path.stem}-") as directory:
        temp = Path(directory) / local_path.name
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, local_path)


def save_state(data_dir: str | Path, state: dict, *, activate: bool = True) -> None:
    root, active_file = get_paths(data_dir)
    run_dir = f"runs/{validate_run_id(state['run_id'])}"
    if state.get("run_dir") != run_dir:
        raise ValueError("Run directory does not match the run ID.")
    clean = {k: v for k, v in state.items() if k != "run_path"}
    _write_state(join_uri(root, run_dir, "state.json"), clean)
    if activate:
        _write_state(active_file, clean)


def experiment_dir_name(cfg: ModelConfig, cohort: str, modalities: Collection[str]) -> str:
    """Return the directory name shared by training and evaluation scripts."""
    modality_slug = "_".join(name for name in MODALITY_NAMES if name in modalities)
    name = f"{cohort}__{modality_slug}__seed{cfg.seed}__aux{cfg.auxiliary_loss_weight:g}"
    defaults = ModelConfig()
    if (cfg.epochs != defaults.epochs
            or cfg.batch_size != defaults.batch_size
            or cfg.modality_dropout_prob != defaults.modality_dropout_prob):
        name += f"__ep{cfg.epochs}__bs{cfg.batch_size}__md{cfg.modality_dropout_prob:g}"
    return name
