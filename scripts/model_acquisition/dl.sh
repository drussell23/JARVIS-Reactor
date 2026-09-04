set -e
export HF_HOME=/home/jarvis_svc/.cache/huggingface
/home/jarvis_svc/.local/bin/uv pip install --python /home/jarvis_svc/.venvs/reactor-train/bin/python \
  "huggingface_hub[hf_transfer]" gptqmodel optimum 2>&1 | tail -4
export HF_HUB_ENABLE_HF_TRANSFER=1
/home/jarvis_svc/.venvs/reactor-train/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    max_workers=8,
)
print("DOWNLOADED ->", p)
PY
