FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Кэш HF на томе
ENV HF_HOME=/runpod-volume/hf_cache
RUN mkdir -p $HF_HOME

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# КОГДА будешь подключать Wan2.2:
# 1) Добавь сюда их требования (или отдельный requirements-wan.txt)
# 2) pip install нужных версий torch/diffusers/transformers/...
# 3) (опц.) git clone официального Wan в /app/Wan2.2 и pip install -e .

COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
