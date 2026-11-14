"""Download external model weights required for Triton."""

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

destination = ROOT / "model_repository" / "llm_generator" / "1"
destination.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_REPO}/{MODEL_FILE}")

path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir=destination,
)

print(f"Downloaded: {path}")
print(f"Size: {Path(path).stat().st_size / 1024 / 1024:.1f} MiB")
