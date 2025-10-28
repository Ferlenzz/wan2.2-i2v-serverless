
import os, io, base64
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- RMSNorm shim ----
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

# ---- SDPA shim: remove unknown kw and accept both positional/keyword qkv ----
_SDPA_ORIG = F.scaled_dot_product_attention
def _sdpa_patched(*args, **kwargs):
    # torch <2.3 doesn't know this kw; torch >=2.3 treats it as optional
    kwargs.pop("enable_gqa", None)
    if len(args) >= 3:
        return _SDPA_ORIG(*args, **kwargs)
    # support calls via keyword arguments
    q = kwargs.pop("q", kwargs.pop("query", None))
    k = kwargs.pop("k", kwargs.pop("key", None))
    v = kwargs.pop("v", kwargs.pop("value", None))
    if q is None or k is None or v is None:
        return _SDPA_ORIG(*args, **kwargs)
    return _SDPA_ORIG(q, k, v, **kwargs)
F.scaled_dot_product_attention = _sdpa_patched

from PIL import Image
# Try I2V first (not available in all builds)
try:
    from diffusers import WanI2VPipeline as _I2VCls
except Exception:
    _I2VCls = None
from diffusers import WanPipeline as _T2VCls, AutoencoderKLWan
from diffusers.utils import export_to_video

HF_HOME = os.environ.get('HF_HOME', '/runpod-volume/cache/hf')
os.makedirs(HF_HOME, exist_ok=True)
os.environ['HF_HOME'] = HF_HOME

MODEL_DIR = os.environ.get('WAN_MODEL_DIR', '/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# dtype control (bf16 by default; allow override)
_DTYPE_ENV = os.environ.get('WAN_DTYPE', '').lower()
if _DTYPE_ENV in ('fp16','float16','half'):
    DTYPE = torch.float16
elif _DTYPE_ENV in ('fp32','float32'):
    DTYPE = torch.float32
else:
    # good default on Ampere+
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEF_FPS    = int(os.environ.get('WAN_FPS', '24'))
DEF_FRAMES = int(os.environ.get('WAN_FRAMES', '121'))
DEF_STEPS  = int(os.environ.get('WAN_STEPS', '30'))
DEF_CFG    = float(os.environ.get('WAN_CFG', '5.0'))

_T2V = None
_I2V = None

def _apply_memory_opts(pipe):
    """
    Enable memory-saving features; controlled via env when relevant.
    """
    # attention/vae slicing are lightweight and help on 16–24GB cards
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    # VAE tiling helps for larger widths/heights
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    # Optional CPU offload (slower but big VRAM saver)
    if os.environ.get("WAN_CPU_OFFLOAD","0") == "1" and torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
            return pipe
        except Exception:
            # fallback to .to(DEVICE) if not available
            pass

    # default path: keep on GPU
    try:
        pipe.to(DEVICE)
    except Exception:
        pass
    return pipe

def _load_t2v():
    global _T2V
    if _T2V is not None:
        return _T2V
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f'WAN_MODEL_DIR not found: {MODEL_DIR}')
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder='vae', torch_dtype=torch.float32)
    pipe = _T2VCls.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=DTYPE)
    pipe = _apply_memory_opts(pipe)
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
    pipe = _I2VCls.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=DTYPE)
    pipe = _apply_memory_opts(pipe)
    _I2V = pipe
    return _I2V

def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert('RGB')

def _file_to_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

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

        # Round dims to multiples of 8 to avoid shape/tiling overhead
        width  = int(width // 8 * 8)
        height = int(height // 8 * 8)

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
        print(
            f"[route] use_i2v={use_i2v} has_img={img_b64 is not None} "
            f"len_b64={len(img_b64) if img_b64 else 0}",
            flush=True
        )

        if use_i2v and img_b64:
            pil = _b64_to_pil(img_b64)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            print(f"[i2v] ref image: {pil.size} mode={pil.mode}", flush=True)

            # аккуратно подбираем имя аргумента, которое принимает текущий пайплайн
            call_vars = getattr(pipe.__call__, "__code__", None)
            names = call_vars.co_varnames if call_vars else ()
            if "image" in names:
                kwargs["image"] = pil
            elif "img" in names:
                kwargs["img"] = pil
            else:
                kwargs["image"] = pil  # запасной вариант


        # help allocator after multiple runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                'fps': fps, 'steps': steps, 'cfg': cfg, 'prompt': prompt,
                'dtype': str(DTYPE), 'cpu_offload': os.environ.get("WAN_CPU_OFFLOAD","0")
            }
        }
        if ret_b64:
            out['video_b64'] = _file_to_b64(out_path)
        else:
            out['video'] = f'data:video/mp4;base64,{_file_to_b64(out_path)}'
        return out

    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}
