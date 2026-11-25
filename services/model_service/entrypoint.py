"""Run a model pipeline batch inside the model container."""
import os
import sys


COMMANDS = {
    "embed": ("extract_embedding",),
    "train": ("main",),
    "compare": ("experiments.compare_modalities",),
    "diagnose": ("experiments.diagnose_modalities",),
    "demo-input": ("services.model_service.demo", "input"),
    "demo-embed": ("services.model_service.demo", "embed"),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(f"Usage: model-service {{{','.join(COMMANDS)},serve}} [options]")
        return
    phase, *options = args
    if phase == "serve":
        port = os.environ.get("PORT", "8002")
        os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "services.model_service.api:app", "--host", "0.0.0.0", "--port", port, *options])
    if phase not in COMMANDS:
        raise SystemExit(f"Unknown model stage: {phase}")
    os.execv(sys.executable, [sys.executable, "-m", *COMMANDS[phase], *options])


if __name__ == "__main__":
    main()
