"""Run the downloaded CPU model server on localhost."""
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import PROJECT_ROOT, Settings
from scripts.setup_models import LLAMA_BUILD, QWEN_FILE


def main():
    settings = Settings()
    if settings.llm_provider != "llamacpp":
        raise SystemExit("For Ollama, start ollama serve and pull the configured model instead.")
    runtime = PROJECT_ROOT / ".cache" / "llamacpp" / LLAMA_BUILD / "llama-server.exe"
    runtime = Path(os.environ.get("LLAMACPP_SERVER", str(runtime)))
    model = settings.data_dir / "models" / QWEN_FILE
    if not runtime.exists() or not model.exists():
        raise SystemExit("Run python scripts/setup_models.py first, or set LLAMACPP_SERVER on Linux/macOS.")
    url = urlparse(settings.llm_url)
    if url.hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit("The local model launcher only binds to localhost.")
    command = [str(runtime), "--model", str(model), "--alias", settings.llm_model,
        "--host", "127.0.0.1", "--port", str(url.port or 8012),
        "--ctx-size", str(settings.llm_context_tokens), "--parallel", "1",
        "--threads", str(min(6, os.cpu_count() or 2)), "--n-gpu-layers", "0"]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
