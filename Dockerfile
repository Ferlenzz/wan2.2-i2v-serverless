# CUDA runtime + cuDNN (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- базовые пакеты ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- ENV и каталоги на Network Volume ---
ENV HF_HOME=/runpod-volume/cache/hf
ENV WAN_MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TMPDIR=/runpod-volume/tmp
ENV PIP_CACHE_DIR=/runpod-volume/pip
ENV XDG_CACHE_HOME=/runpod-volume/.cache

RUN mkdir -p $HF_HOME $TMPDIR $PIP_CACHE_DIR $XDG_CACHE_HOME

WORKDIR /app

# --- Python зависимости ---
# Torch/TV для CUDA 12.1 (тут уже есть torch.nn.RMSNorm)
RUN python3 -m pip install -U pip \
 && python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1 torchvision==0.19.1 \
 && python3 -m pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers" \
      transformers accelerate safetensors pillow "numpy<2" \
      imageio imageio-ffmpeg runpod

# --- Код ---
COPY handler.py /app/handler.py
COPY start.sh   /app/start.sh
RUN chmod +x /app/start.sh

# --- Entrypoint ---
CMD ["/app/start.sh"]
