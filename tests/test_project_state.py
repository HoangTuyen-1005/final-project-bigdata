import json

import pytest

from project_state import load_state, save_state, validate_run_id


def test_explicit_run_state_does_not_move_active_run(tmp_path):
    first = {"run_id": "first", "run_dir": "runs/first", "status": "prepared"}
    second = {"run_id": "second", "run_dir": "runs/second", "status": "prepared"}
    save_state(tmp_path, first)
    save_state(tmp_path, second)
    save_state(tmp_path, {**first, "status": "embedded"}, activate=False)
    assert load_state(tmp_path, run_id="first", required_status="embedded")["run_path"] == (
        tmp_path / "runs/first"
    )
    assert load_state(tmp_path, required_status="prepared")["run_id"] == "second"

    save_state(tmp_path, {**second, "status": "finalized"}, activate=False)
    assert load_state(tmp_path, required_status="finalized")["run_id"] == "second"


def test_loads_previous_active_state_format(tmp_path):
    state = {"run_id": "legacy", "run_dir": "runs/legacy", "status": "finalized"}
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert load_state(tmp_path, run_id="legacy")["status"] == "finalized"
    with pytest.raises(FileNotFoundError, match="another"):
        load_state(tmp_path, run_id="another")
    with pytest.raises(ValueError, match="Run ID"):
        validate_run_id("../legacy")
