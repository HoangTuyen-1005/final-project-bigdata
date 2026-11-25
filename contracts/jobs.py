"""Lightweight async job runner for containerized SOA batch services."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import subprocess
import sys
import threading
import uuid


class JobManager:
    """Runs service CLI commands asynchronously and tracks their status/logs."""

    def __init__(self, service_name: str, max_workers: int = 1) -> None:
        self.service_name = service_name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

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
        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record["status"] = "running"
            record["started_at"] = datetime.now(timezone.utc).isoformat()
            cmd = list(record["command"])
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

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return dict(rec) if rec is not None else None

    def list_all(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._jobs.values()]
