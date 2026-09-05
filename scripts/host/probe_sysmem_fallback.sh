#!/usr/bin/env bash
# Is the WDDM "CUDA - Sysmem Fallback Policy" still letting CUDA spill past
# the card into host RAM? Run from WSL2. Bounded: stops at 33.5 GiB, well
# short of exhausting the fallback (which corrupts the CUDA context).
#
#   FALLBACK ON  -> torch holds >32 GiB on a 32 GiB card. Every byte past
#                   the card is Windows commit on vmmemWSL. Fix it in
#                   NVIDIA Control Panel > Manage 3D Settings > Global >
#                   "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem
#                   Fallback", then re-run this until it says OFF.
#   FALLBACK OFF -> honest OOM at the true ceiling (~31 GiB). Good.
#
# Measured 2026-09-04 22:27: ON (34.0 GiB held, nvidia-smi pinned at 31897 MiB).
set -u
V="${REACTOR_TRAIN_PYTHON:-$HOME/.venvs/reactor-train/bin/python}"
exec "$V" - <<'PY'
import subprocess, torch
GIB = 1 << 30
blocks = []
try:
    for _ in range(40):
        blocks.append(torch.empty(GIB, dtype=torch.uint8, device="cuda"))
        blocks[-1].fill_(1)  # touch it, so it is really backed
        held = torch.cuda.memory_allocated() / GIB
        if held > 33.5:
            print(f"FALLBACK ON: torch holds {held:.1f} GiB on a "
                  f"{torch.cuda.get_device_properties(0).total_memory / GIB:.1f} GiB card")
            break
    else:
        print(f"inconclusive: loop ended at {torch.cuda.memory_allocated() / GIB:.1f} GiB")
except Exception as exc:  # noqa: BLE001
    print(f"FALLBACK OFF: honest {type(exc).__name__} at "
          f"{torch.cuda.memory_allocated() / GIB:.1f} GiB")
print("nvidia-smi:", subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())
del blocks
torch.cuda.empty_cache()
PY
