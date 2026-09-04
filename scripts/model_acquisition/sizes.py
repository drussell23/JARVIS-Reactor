import json, urllib.request, urllib.error
REPOS = [
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "base bf16 (bnb-4bit at load)"),
    ("btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit", "GPTQ 4-bit"),
    ("QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ", "AWQ 4-bit"),
    ("cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit", "AWQ 4-bit"),
]
for repo, label in REPOS:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  {label:<30} {repo}  HTTP {e.code}"); continue
    files = [s["rfilename"] for s in (d.get("siblings") or [])]
    wts = [f for f in files if f.endswith(".safetensors")]
    total = 0
    for f in wts:
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/{repo}/resolve/main/{f}", method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                total += int(r.headers.get("Content-Length") or 0)
        except Exception:
            pass
    qc = (d.get("config") or {}).get("quantization_config") or {}
    print(f"  {label:<30} {total/2**30:6.1f} GiB  shards={len(wts):3d}  "
          f"quant={qc.get('quant_method', '(none)')}")
