"""Persistent async job runner for containerized SOA batch services."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import subprocess
import sys
import threading
import uuid

from contracts.state import resolve_data_root
from contracts.storage import join_uri, list_json_uris, read_text, write_text


class JobManager:
    """Runs service CLI commands asynchronously and persists status/logs to MinIO or local."""

    def __init__(self, service_name: str, max_workers: int = 1) -> None:
        self.service_name = service_name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._store_prefix = join_uri(resolve_data_root("/data"), "_service_jobs", service_name)
        self._jobs: dict[str, dict] = {}
        self._load_persisted()

    def _job_uri(self, job_id: str) -> str:
        return join_uri(self._store_prefix, f"{job_id}.json")

    def _persist(self, record: dict) -> None:
        try:
            write_text(self._job_uri(record["job_id"]), json.dumps(record, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[{self.service_name}] Warning: failed to persist job {record['job_id']}: {exc}", flush=True)

    def _load_persisted(self) -> None:
        try:
            for uri in list_json_uris(self._store_prefix):
                rec = json.loads(read_text(uri, encoding="utf-8"))
                if rec.get("status") in ("queued", "running"):
                    rec["status"] = "interrupted"
                    rec["finished_at"] = datetime.now(timezone.utc).isoformat()
                    self._persist(rec)
                self._jobs[rec["job_id"]] = rec
        except Exception as exc:
            print(f"[{self.service_name}] Warning: could not load persisted jobs: {exc}", flush=True)

    def submit(self, action: str, module: str, args: list[str]) -> dict:
        job_id = uuid.uuid4().hex[:12]
        record = {
            "job_id": job_id,
            "service": self.service_name,
            "action": action,
            "command": [sys.executable, "-m", module, *args],
            "status": "queued",
            "returncode": None,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "finished_at": None,
            "log_tail": [],
        }
        with self._lock:
            self._jobs[job_id] = record
            self._persist(record)
        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record["status"] = "running"
            record["started_at"] = datetime.now(timezone.utc).isoformat()
            cmd = list(record["command"])
            self._persist(record)
        tail: deque[str] = deque(maxlen=40)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.rstrip("\r\n")
            print(f"[{self.service_name}:{job_id}] {clean}", flush=True)
            tail.append(clean)
            with self._lock:
                record["log_tail"] = list(tail)
        code = proc.wait()
        with self._lock:
            record["returncode"] = code
            record["status"] = "succeeded" if code == 0 else "failed"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["log_tail"] = list(tail)
            self._persist(record)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return dict(rec) if rec is not None else None

    def list_all(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._jobs.values()]
