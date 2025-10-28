#!/usr/bin/env bash
set -euo pipefail

echo "[start] boot"
python3 - <<'PY'
import os, platform
print("[env] python", platform.python_version())
for k in ("HF_HOME","WAN_MODEL_DIR","HF_HUB_ENABLE_HF_TRANSFER",
          "TMPDIR","PIP_CACHE_DIR","XDG_CACHE_HOME"):
    v=os.environ.get(k); 
    if v: print(f"[env] {k}={v}")
PY

# Запускаем runpod handler
python3 - <<'PY'
import runpod, handler
runpod.serverless.start({"handler": handler.handler})
PY
