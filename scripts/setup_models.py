"""Download pinned public models and an optional portable Windows CPU runtime."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import PROJECT_ROOT, Settings

EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
QWEN_REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
QWEN_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLAMA_BUILD = "b11574"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(client, url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected or digest(path) == expected):
        print(f"Already downloaded: {path.name}", flush=True)
        return digest(path)
    partial = path.with_name(path.name + ".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        total = 0
        checkpoint = 0
        with partial.open("wb") as file:
            for block in response.iter_bytes(1024 * 1024):
                file.write(block)
                total += len(block)
                if total - checkpoint >= 50 * 1024 * 1024:
                    print(f"{path.name}: {total // 1048576} MiB", flush=True)
                    checkpoint = total
    actual = digest(partial)
    if expected and actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}; downloaded file was not activated")
    partial.replace(path)
    print(f"Downloaded: {path.name}", flush=True)
    return actual


def hf_files(client, repo, revision):
    response = client.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    response.raise_for_status()
    return {item["rfilename"]: item.get("lfs", {}).get("sha256")
            for item in response.json()["siblings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-only", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        metadata = hf_files(client, EMBEDDING_REPO, settings.embedding_revision)
        checksums = {}
        for filename in ("onnx/model.onnx", "tokenizer.json", "config.json", "1_Pooling/config.json"):
            local_name = "pooling_config.json" if filename.startswith("1_Pooling") else Path(filename).name
            checksums[local_name] = download(client,
                f"https://huggingface.co/{EMBEDDING_REPO}/resolve/{settings.embedding_revision}/{filename}",
                settings.model_dir / local_name, metadata.get(filename))
        manifest = {"model": EMBEDDING_REPO, "revision": settings.embedding_revision,
                    "dimension": 384, "checksums": checksums}
        (settings.model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.embedding_only:
            return
        metadata = hf_files(client, QWEN_REPO, QWEN_REVISION)
        checksum = download(client,
            f"https://huggingface.co/{QWEN_REPO}/resolve/{QWEN_REVISION}/{QWEN_FILE}",
            settings.data_dir / "models" / QWEN_FILE, metadata[QWEN_FILE])
        (settings.data_dir / "models" / "generator.json").write_text(json.dumps({
            "model": QWEN_REPO, "revision": QWEN_REVISION, "file": QWEN_FILE,
            "sha256": checksum, "llama_build": LLAMA_BUILD}, indent=2), encoding="utf-8")
        if sys.platform == "win32":
            filename = f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip"
            response = client.get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_BUILD}")
            response.raise_for_status()
            asset = next(item for item in response.json()["assets"] if item["name"] == filename)
            checksum = asset.get("digest", "").removeprefix("sha256:") or None
            archive = PROJECT_ROOT / ".cache" / "llamacpp" / filename
            download(client, asset["browser_download_url"], archive, checksum)
            runtime = PROJECT_ROOT / ".cache" / "llamacpp" / LLAMA_BUILD
            runtime.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as file:
                for entry in file.infolist():
                    target = (runtime / entry.filename).resolve()
                    if not target.is_relative_to(runtime.resolve()):
                        raise RuntimeError("Runtime archive contains an unsafe path")
                file.extractall(runtime)
    print("Models ready. Run scripts/run-model.ps1, then start the backend and frontend.", flush=True)


if __name__ == "__main__":
    main()
