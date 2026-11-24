"""Run-scoped state: prevents mixing embeddings and Spark outputs from different ETL runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")


def get_paths(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[Path, Path]:
    root = Path(data_dir).resolve()
    return root, root / "pipeline_state.json"


def load_state(data_dir: str | Path = DEFAULT_DATA_DIR, required_status: str | None = None) -> dict:
    root, state_file = get_paths(data_dir)
    if not state_file.is_file():
        raise FileNotFoundError(f"Missing {state_file}. Run 'python -m spark_etl.run_etl prepare' first.")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("run_id") is None or state.get("run_dir") is None:
        raise ValueError(f"Invalid pipeline state: {state_file}")
    # The run folder is relative to data_dir (portable between machines).
    state["run_path"] = root / state["run_dir"]
    if required_status and state.get("status") != required_status:
        raise RuntimeError(f"Pipeline status is {state.get('status')!r}; expected {required_status!r}.")
    return state


def save_state(data_dir: str | Path, state: dict) -> None:
    root, state_file = get_paths(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in state.items() if k != "run_path"}
    temp = state_file.with_suffix(".tmp")
    temp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, state_file)
