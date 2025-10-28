import os, io, base64, inspect
from typing import Any, Dict

import torch
import torch.nn as nn

# ---- RMSNorm shim (должен стоять ДО импортов diffusers/transformers) ----
if not hasattr(nn, "RMSNorm"):
    class _RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x: torch.Tensor):
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return self.weight * (x * rms)
    nn.RMSNorm = _RMSNorm
# ------------------------------------------------------------------------

from PIL import Image

# Импорты diffusers после шима
try:
    from diffusers import WanI2VPipeline as _I2VCls  # новый класс I2V (если есть)
except Exception:
    _I2VCls = None

from diffusers import WanPipeline as _T2VCls, AutoencoderKLWan
from diffusers.utils import export_to_video

# ---------- ENV / defaults ----------
HF_HOME = os.environ.get('HF_HOME', '/runpod-volume/cache/hf')
os.makedirs(HF_HOME, exist_ok=True)
os.environ['HF_HOME'] = HF_HOME

MODEL_DIR = os.environ.get('WAN_MODEL_DIR', '/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEF_FPS    = int(os.environ.get('WAN_FPS', '24'))
DEF_FRAMES = int(os.environ.get('WAN_FRAMES', '121'))   # ~5s @ 24fps
DEF_STEPS  = int(os.environ.get('WAN_STEPS', '30'))
DEF_CFG    = float(os.environ.get('WAN_CFG', '5.0'))

# ---------- pipeline singletons ----------
_T2V = None
_I2V = None

def _load_t2v():
    global _T2V
    if _T2V is not None:
        return _T2V
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f'WAN_MODEL_DIR not found: {MODEL_DIR}')
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder='vae', torch_dtype=torch.float32)
    pipe = _T2VCls.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=DTYPE).to(DEVICE)
    _T2V = pipe
    return _T2V

def _load_i2v():
    global _I2V
    if _I2V is not None:
        return _I2V
    if _I2VCls is None:
        return None
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f'WAN_MODEL_DIR not found: {MODEL_DIR}')
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder='vae', torch_dtype=torch.float32)
    pipe = _I2VCls.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=DTYPE).to(DEVICE)
    _I2V = pipe
    return _I2V

# ---------- helpers ----------
def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert('RGB')

def _file_to_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ---------- handler ----------
def handler(event: Dict[str, Any]):
    try:
        inp = (event or {}).get('input') or {}
        prompt = inp.get('prompt', '')
        width  = int(inp.get('width',  1280))
        height = int(inp.get('height',  704))
        frames = int(inp.get('length',  DEF_FRAMES))
        fps    = int(inp.get('fps',     DEF_FPS))
        steps  = int(inp.get('steps',   DEF_STEPS))
        cfg    = float(inp.get('cfg',   DEF_CFG))
        img_b64 = inp.get('image_base64')
        ret_b64 = bool(inp.get('return_video_b64', True))

        use_i2v = bool(img_b64)
        pipe = _load_i2v() if use_i2v else _load_t2v()
        if pipe is None:
            use_i2v = False
            pipe = _load_t2v()

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=cfg,
            num_inference_steps=steps,
        )

        if use_i2v and img_b64:
            pil = _b64_to_pil(img_b64)
            sig = inspect.signature(pipe.__call__)
            image_param_name = None
            for name in ('image','image_prompt','img','reference_image','ip_adapter_image'):
                if name in sig.parameters:
                    image_param_name = name
                    break
            if image_param_name is None:
                use_i2v = False
            else:
                kwargs[image_param_name] = pil

        with torch.inference_mode():
            result = pipe(**kwargs)
            frames_list = result.frames[0]

        os.makedirs('/tmp/out', exist_ok=True)
        out_path = '/tmp/out/wan_out.mp4'
        export_to_video(frames_list, out_path, fps=fps)

        out = {
            'ok': True,
            'meta': {
                'mode': 'i2v' if use_i2v else 't2v',
                'width': width, 'height': height, 'frames': frames,
                'fps': fps, 'steps': steps, 'cfg': cfg, 'prompt': prompt
            }
        }
        if ret_b64:
            out['video_b64'] = _file_to_b64(out_path)
        else:
            out['video'] = f'data:video/mp4;base64,{_file_to_b64(out_path)}'
        return out

    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}
