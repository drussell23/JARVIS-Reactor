# Wait for the DOWNLOADER specifically. The previous pattern was
# mis-escaped inside a heredoc, matched nothing, and fired instantly.
while pgrep -f 'reactor/\.dl\.sh' >/dev/null 2>&1; do sleep 20; done
echo "=== download finished: $(du -sh ~/.cache/huggingface/hub | cut -f1) ==="

# gptqmodel imports torchvision transitively (for an unrelated InternVL
# definition). Same cu128 index as torch, or pip resolves a CPU build.
/home/jarvis_svc/.local/bin/uv pip install \
  --python /home/jarvis_svc/.venvs/reactor-train/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torchvision 2>&1 | tail -3

cd /mnt/c/Users/Jarvis/Desktop/TrinityAi/reactor || exit 1
export HF_HOME=/home/jarvis_svc/.cache/huggingface
export REACTOR_GRPO_VERIFY_CMD="/home/jarvis_svc/.venvs/ov/bin/python /mnt/c/Users/Jarvis/Desktop/TrinityAi/reactor/scripts/verify_candidate.py"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo
/home/jarvis_svc/.venvs/reactor-train/bin/python scripts/profile_grpo_vram.py \
  --model btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit \
  --pre-quantized --num-generations 4 --max-completion-length 256 --steps 1 \
  --json-out /tmp/grpo_vram_30b.json
