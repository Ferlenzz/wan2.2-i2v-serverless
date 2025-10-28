import os
import io
import base64
from typing import Any, Dict

import torch
from PIL import Image
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# ---------- ENV ----------
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/cache/hf")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME

MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEF_FPS    = int(os.environ.get("WAN_FPS", "24"))
DEF_FRAMES = int(os.environ.get("WAN_FRAMES", "121"))  # ~5с @ 24fps
DEF_STEPS  = int(os.environ.get("WAN_STEPS", "30"))
DEF_CFG    = float(os.environ.get("WAN_CFG", "5.0"))

# ---------- SINGLETON PIPE ----------
_PIPE = None

def _load_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"WAN_MODEL_DIR not found: {MODEL_DIR}")
    # vae отдельно (как советует карточка модели)
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=DTYPE)
    pipe.to(DEVICE)
    _PIPE = pipe
    return _PIPE

# ---------- UTILS ----------
def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------- HANDLER ----------
def handler(event: Dict[str, Any]):
    """
    input:
      prompt: str (optional)
      image_base64: str (optional; если есть — I2V, если нет — T2V)
      width, height: int
      length: int (num_frames)
      fps: int
      steps: int (diffusion steps)
      cfg: float (guidance_scale)
    """
    try:
        inp = (event or {}).get("input") or {}
        prompt = inp.get("prompt", "")
        width  = int(inp.get("width",  1280))
        height = int(inp.get("height",  704))
        frames = int(inp.get("length",  DEF_FRAMES))
        fps    = int(inp.get("fps",     DEF_FPS))
        steps  = int(inp.get("steps",   DEF_STEPS))
        cfg    = float(inp.get("cfg",   DEF_CFG))
        img_b64 = inp.get("image_base64")

        pipe = _load_pipe()

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=cfg,
            num_inference_steps=steps,
        )

        if img_b64:
            pil = _b64_to_pil(img_b64)
            kwargs["image"] = pil  # I2V
        # иначе чистый T2V по prompt

        with torch.inference_mode():
            result = pipe(**kwargs)
            frames_list = result.frames[0]  # list[np.ndarray(H,W,3)]

        os.makedirs("/tmp/out", exist_ok=True)
        out_path = "/tmp/out/wan_out.mp4"
        export_to_video(frames_list, out_path, fps=fps)

        return {
            "ok": True,
            "meta": {
                "width": width, "height": height, "frames": frames,
                "fps": fps, "steps": steps, "cfg": cfg, "prompt": prompt
            },
            "video_b64": _file_to_b64(out_path)
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
