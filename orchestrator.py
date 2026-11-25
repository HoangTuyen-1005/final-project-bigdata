"""CLI client for the centralized SOA Orchestrator & API Gateway (port 8000)."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid

from contracts.state import validate_run_id


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["demo", "real"], default="demo", nargs="?")
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex if args.run_id is None else validate_run_id(args.run_id)
    health = _http_json("GET", f"{args.gateway_url}/health")
    print(f"[Gateway] status={health['status']}", flush=True)

    started = _http_json(
        "POST",
        f"{args.gateway_url}/pipelines/run",
        {"mode": args.mode, "run_id": run_id, "epochs": args.epochs, "auto_load_inference": True},
    )
    print(f"[Gateway] Submitted run_id={started['run_id']}", flush=True)

    seen_stages: dict[str, str] = {}
    while True:
        status = _http_json("GET", f"{args.gateway_url}/pipelines/{run_id}")
        for stage, info in status.get("stages", {}).items():
            st = info.get("status")
            if seen_stages.get(stage) != st:
                seen_stages[stage] = st
                print(f"[Pipeline] {stage}: {st}", flush=True)
        if status["status"] == "succeeded":
            print(f"[Done] Pipeline {run_id} succeeded", flush=True)
            break
        if status["status"] == "failed":
            raise RuntimeError(f"Pipeline {run_id} failed:\n{status.get('error')}")
        time.sleep(2.0)

    sample = _http_json(
        "POST",
        f"{args.gateway_url}/predict",
        {
            "numeric": {"accommodates": 4, "bedrooms": 2, "bathrooms": 1.5},
            "categorical": {"city": "Boston", "room_type": "Entire home/apt"},
        },
    )
    print(f"[Gateway /predict] {sample}", flush=True)


if __name__ == "__main__":
    main()
