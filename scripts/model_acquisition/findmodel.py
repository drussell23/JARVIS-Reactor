import json, urllib.request, urllib.parse
def search(q, **kw):
    p = {"search": q, "limit": 25, "full": "false"}
    p.update(kw)
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(p)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)

print("=== 4-bit / quantized Qwen3-Coder-30B ===")
seen = set()
for q in ("Qwen3-Coder-30B-A3B-Instruct 4bit", "Qwen3-Coder-30B bnb-4bit",
          "Qwen3-Coder-30B-A3B", "Qwen3-Coder-30B AWQ"):
    for m in search(q):
        mid = m["modelId"]
        if mid in seen:
            continue
        seen.add(mid)
        low = mid.lower()
        if "30b" in low and any(t in low for t in ("4bit", "4-bit", "awq", "gptq", "bnb", "int4")):
            print(f"  {mid}   downloads={m.get('downloads', 0)}")
print("\n=== base repo file sizes (needs no auth via resolve HEAD) ===")
import urllib.error
for f in ("model-00001-of-00013.safetensors", "model.safetensors.index.json", "config.json"):
    url = f"https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/resolve/main/{f}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = int(r.headers.get("Content-Length") or 0)
            print(f"  {f:<44} {n/2**20:8.1f} MiB")
    except urllib.error.HTTPError as e:
        print(f"  {f:<44} HTTP {e.code}")
