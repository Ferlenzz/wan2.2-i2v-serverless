# CUDA runtime с cuDNN. Лёгкий рантайм, python ставим через apt.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# ---------- БАЗОВЫЕ ENV ----------
# Кэш HF на томе
ENV HF_HOME=/runpod-volume/cache/hf
# Папка с моделью на томе (ты уже скачал туда Wan2.2-TI2V-5B-Diffusers)
ENV WAN_MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers
# Отключаем fast-transfer у HF (иначе может ругаться на отсутствие hf_transfer)
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# Перенос временных директорий/кэшей на том (чтобы не упираться в слой контейнера)
ENV TMPDIR=/runpod-volume/tmp
ENV PIP_CACHE_DIR=/runpod-volume/pip
ENV XDG_CACHE_HOME=/runpod-volume/.cache

# Создадим каталоги на всякий случай
RUN mkdir -p $HF_HOME $TMPDIR $PIP_CACHE_DIR $XDG_CACHE_HOME

WORKDIR /app

# ---------- Python зависимости ----------
# Torch под CUDA 12.1 + основные либы (numpy<2 для совместимости с py3.8/3.10 образами)
RUN python3 -m pip install -U pip \
 && python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1
 && python3 -m pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers" \
      transformers accelerate safetensors pillow "numpy<2" \
      imageio imageio-ffmpeg runpod

# ---------- КОД ----------
COPY handler.py /app/handler.py
COPY start.sh   /app/start.sh
RUN chmod +x /app/start.sh

# ---------- CMD ----------
CMD ["/app/start.sh"]
