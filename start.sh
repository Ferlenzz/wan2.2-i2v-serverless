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

# -----------------------------------------------------------------------------
# Зависимости для токенизации текста (ftfy) — нужна CLIP'у/токенайзеру
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_FTFY'
import sys, subprocess
def pip(*a): subprocess.check_call([sys.executable, "-m", "pip", *a])
pip("install", "-q", "--no-cache-dir", "ftfy>=6.1.1", "regex>=2023.0.0")
print("[deps] ftfy installed/ok")
PY_FTFY


# Запускаем runpod handler
python3 - <<'PY'
import runpod, handler
runpod.serverless.start({"handler": handler.handler})
PY
