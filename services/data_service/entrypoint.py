"""Submit the data pipeline from a driver in the Spark network."""
import json
import os
import socket
import sys
import time
from urllib.request import urlopen


def wait_for_workers(endpoint: str, expected: int, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    alive = []
    while time.monotonic() < deadline:
        try:
            with urlopen(endpoint, timeout=2) as response:
                status = json.load(response)
            alive = [worker["host"] for worker in status["workers"] if worker["state"] == "ALIVE"]
        except (OSError, ValueError, KeyError):
            time.sleep(1)
            continue
        if len(alive) >= expected:
            print(f"[Spark Standalone] Registered workers: {', '.join(alive)}", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"Spark master at {endpoint} registered {len(alive)} of {expected} required workers")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: data-service {prepare,finalize,benchmark,serve} [options]")
        return
    phase, *options = args
    if phase == "serve":
        port = os.environ.get("PORT", "8001")
        os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "services.data_service.api:app", "--host", "0.0.0.0", "--port", port, *options])
    if phase not in ("prepare", "finalize", "benchmark"):
        raise SystemExit(f"Unknown data stage: {phase}")
    if os.environ.get("SPARK_MASTER", "").startswith("spark://") and not {"-h", "--help"}.intersection(options):
        wait_for_workers(os.environ["SPARK_MASTER_UI"], int(os.environ["SPARK_EXPECTED_WORKERS"]))
        if not os.environ.get("SPARK_DRIVER_HOST"):
            # A one-off Compose driver advertises its own address to both workers.
            os.environ["SPARK_DRIVER_HOST"] = socket.gethostbyname(socket.gethostname())
    command = (["-m", "spark_etl.benchmark", *options] if phase == "benchmark"
               else ["-m", "spark_etl.run_etl", phase, *options])
    os.execv(sys.executable, [sys.executable, *command])


if __name__ == "__main__":
    main()
