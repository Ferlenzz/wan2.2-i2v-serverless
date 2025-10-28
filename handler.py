# handler.py
import base64, io, os, json
from typing import Any, Dict

import runpod
from PIL import Image

# ------------------ CONFIG ------------------
# Кэш HF на томе (Runpod Network Volume)
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/hf_cache")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# Где лежат веса Wan (на томе)
WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/runpod-volume/wan_ckpt")

# Эконом-профиль по умолчанию
DEF_W, DEF_H = 384, 384
DEF_FRAMES   = int(os.environ.get("WAN_FORCE_FRAMES", "12") or "12")
DEF_STEPS    = 8
DEF_FPS      = 12
DEF_CFG      = 2.0

# ------------------ MODEL SINGLETON ------------------
_WAN = None
def load_model():
    """
    Ожидаем, что официальная библиотека Wan2.2 установлена в образе.
    Веса (UNet/VAE/токенайзер и пр.) — на томе WAN_CKPT_DIR.
    """
    global _WAN
    if _WAN is not None:
        return _WAN

    # Импортируем «ванильный» Wan2.2
    from wan.image2video import WanI2V  # из официального репозитория Wan 2.2

    # Инициализация модели один раз на процесс
    # Если у WanI2V иные параметры, см. их README; обычно достаточно указать путь к чекпоинтам.
    _WAN = WanI2V(ckpt_dir=WAN_CKPT_DIR)
    return _WAN

# ------------------ UTILS ------------------
def decode_image_b64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def encode_video_mp4_from_frames(frames, fps=DEF_FPS) -> bytes:
    """
    Кодируем список кадров (numpy HxWx3, uint8) в MP4 (libx264, yuv420p).
    Требуется imageio-ffmpeg (есть в requirements) и ffmpeg в системе.
    """
    import imageio
    with io.BytesIO() as buf:
        writer = imageio.get_writer(
            buf, format="ffmpeg", mode="I",
            fps=int(fps), codec="libx264", pixelformat="yuv420p"
        )
        for fr in frames:
            writer.append_data(fr)
        writer.close()
        return buf.getvalue()

def try_generate(model, **kw):
    """
    Универсальный вызов: у некоторых версий WanI2V ключ 'image',
    у других — 'img'/'x'. Пробуем несколько вариантов.
    """
    import numpy as np

    def pil_to_np(img: Image.Image):
        import numpy as _np
        return _np.array(img).astype("uint8")

    image = kw.pop("image", None)
    if image is None:
        raise ValueError("image is required")

    # Пробуем разные имена параметров
    for key in ("image", "img", "x"):
        try:
            out = model.generate(**{key: image}, **kw)
            # Приводим результат к списку кадров (numpy HxWx3)
            if isinstance(out, list) and len(out) and hasattr(out[0], "shape"):
                return out
            # Если вернулся PIL или np.array одного кадра — оборачиваем
            if hasattr(out, "shape"):
                return [out]
            if isinstance(out, Image.Image):
                return [pil_to_np(out)]
        except TypeError:
            continue
    # Если объект модели callable:
    try:
        out = model(image, **kw)
        if isinstance(out, list) and len(out) and hasattr(out[0], "shape"):
            return out
    except Exception as e:
        raise e
    raise RuntimeError("WanI2V.generate() did not return frames")

# ------------------ HANDLER ------------------
def handler(event: Dict[str, Any]):
    """
    Ожидаем payload:
    {
      "input": {
        "action": "gen",
        "user_id": "u1",
        "image_base64": "...",
        "prompt": "cat jumps",
        "width": 384, "height": 384,
        "length": 12, "fps": 12,
        "steps": 8, "cfg": 2.0,
        "return_video_b64": true
      }
    }
    """
    inp = (event or {}).get("input") or {}
    if inp.get("action") != "gen":
        return {"error": "unsupported action"}

    # Параметры генерации
    width   = int(inp.get("width",  DEF_W))
    height  = int(inp.get("height", DEF_H))
    length  = int(inp.get("length", DEF_FRAMES))
    fps     = int(inp.get("fps",    DEF_FPS))
    steps   = int(inp.get("steps",  DEF_STEPS))
    cfg     = float(inp.get("cfg",  DEF_CFG))
    prompt  = inp.get("prompt", "")

    # Входная картинка
    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "image_base64 required"}
    img = decode_image_b64(b64).resize((width, height), Image.BICUBIC)

    # Модель
    model = load_model()

    # Генерация (официальный Wan2.2)
    # Большинство сборок WanI2V ожидает PIL.Image / np.array и такие ключи:
    # width/height/length/fps/steps/cfg/prompt
    frames = try_generate(
        model,
        image=img,
        prompt=prompt,
        width=width,
        height=height,
        length=length,  # число кадров
        fps=fps,
        steps=steps,
        cfg=cfg,
    )

    # Кодирование MP4
    mp4_bytes = encode_video_mp4_from_frames(frames, fps=fps)
    video_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    return {
        "video_b64": video_b64,
        "meta": {
            "width": width, "height": height, "length": length, "fps": fps, "steps": steps, "cfg": cfg,
            "prompt": prompt
        }
    }

runpod.serverless.start({"handler": handler})

