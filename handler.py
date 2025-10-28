import os
import io
import json
import base64
import runpod
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

def b64_to_pil(b64: str) -> Image.Image:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# "Last image" store
def _last_dir() -> Path:
    # Prefer persistent Runpod volume
    root = Path(os.environ.get("WAN_LAST_IMAGE_DIR", "/runpod-volume/last_images"))
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        root = Path("/tmp/last_images"); root.mkdir(parents=True, exist_ok=True)
    return root

def last_image_path(user_id: str, session_id: Optional[str] = None) -> Path:
    safe_user = (user_id or "default").replace("/", "_")
    safe_sess = (session_id or "").replace("/", "_")
    name = f"{safe_user}{('-' + safe_sess) if safe_sess else ''}.png"
    return _last_dir() / name

def save_last_image(img: Image.Image, user_id: str, session_id: Optional[str] = None) -> str:
    p = last_image_path(user_id, session_id)
    img.save(p, format="PNG")
    return str(p)

def load_last_image(user_id: str, session_id: Optional[str] = None) -> Optional[Image.Image]:
    p = last_image_path(user_id, session_id)
    if p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None
    return None

# VRAM-friendly switches
def enable_memory_savers(pipe: Any):
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
    except Exception:
        pass
    if os.environ.get("WAN_CPU_OFFLOAD","0") == "1":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

# Build pipeline
_PIPE = None
def get_pipe() -> Any:
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    # actual Wan/T2V/I2V pipeline loading
    from diffusers import WanPipeline  # type: ignore
    model_dir = os.environ.get("WAN_MODEL_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers")
    dtype = os.environ.get("WAN_DTYPE", "fp16")
    torch_dtype = {"fp16":"float16","bfloat16":"bfloat16","fp32":"float32"}.get(dtype, "float16")

    _PIPE = WanPipeline.from_pretrained(model_dir, torch_dtype=getattr(__import__('torch'), torch_dtype))  # noqa: E402
    enable_memory_savers(_PIPE)
    return _PIPE

#pipeline output (frames:list[PIL.Image], fps:int)
def normalize_video_output(out: Any, fallback_fps: int = 12) -> Tuple[List[Image.Image], int]:
    fps = getattr(out, "fps", None) or fallback_fps
    for key in ("frames","images"):
        frames = getattr(out, key, None)
        if isinstance(frames, list) and frames:
            if isinstance(frames[0], Image.Image):
                return frames, fps
            try:
                return [Image.fromarray(f) for f in frames], fps
            except Exception:
                pass
    if isinstance(out, dict):
        for key in ("frames","images","videos"):
            if key in out:
                val = out[key]
                if isinstance(val, list) and val:
                    if isinstance(val[0], Image.Image):
                        return val, fps
                    try:
                        return [Image.fromarray(f) for f in val], fps
                    except Exception:
                        pass
    #torch tensors / numpy arrays
    try:
        import numpy as np
        import torch
        arr = None
        if hasattr(out, "videos"):
            arr = out.videos
        elif hasattr(out, "frames"):
            arr = out.frames
        if arr is not None:
            if isinstance(arr, torch.Tensor):
                ten = arr.detach().cpu()
                if ten.ndim == 4 and ten.shape[1] in (1,3):      # (T,C,H,W)
                    ten = ten.permute(0,2,3,1)
                npy = (ten.numpy()*255).astype("uint8") if ten.max()<=1.0 else ten.numpy().astype("uint8")
                return [Image.fromarray(npy[i]) for i in range(npy.shape[0])], fps
            if isinstance(arr, np.ndarray):
                if arr.ndim == 4 and arr.shape[-1] in (1,3):    # (T,H,W,C)
                    npy = (arr*255).astype("uint8") if arr.max()<=1.0 else arr.astype("uint8")
                    return [Image.fromarray(npy[i]) for i in range(npy.shape[0])], fps
    except Exception:
        pass
    return [], fps

# Encode frames MP4
def frames_to_mp4_bytes(frames: List[Image.Image], fps: int) -> bytes:
    try:
        import imageio.v2 as iio
        import numpy as np
        buf = io.BytesIO()
        writer = iio.get_writer(buf, format="mp4", mode="I", fps=fps, codec="libx264")
        for im in frames:
            writer.append_data(np.array(im.convert("RGB")))
        writer.close()
        return buf.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to encode MP4: {e}")

# Runpod handler
def handler(event):
    ip = event.get("input") or {}
    action = (ip.get("action") or "").lower().strip() or ("i2v" if ip.get("image_base64") else "t2v")
    user_id = str(ip.get("user_id") or "user")
    session_id = ip.get("session_id")  # optional routing key for parallel sessions

    use_last = bool(ip.get("use_last_image", False)) or (action in ("continue","i2v_continue","continue_i2v"))
    save_last = ip.get("save_last_image", True) is not False

    prompt = ip.get("prompt","")
    width  = int(ip.get("width",  384))
    height = int(ip.get("height", 384))
    fps    = int(ip.get("fps",    int(os.environ.get("WAN_FPS","12"))))
    steps  = int(ip.get("steps",  int(os.environ.get("WAN_STEPS","10"))))
    cfg    = float(ip.get("cfg",  float(os.environ.get("WAN_CFG","2.0"))))
    frames_n = int(ip.get("length", int(os.environ.get("WAN_FRAMES","12"))))
    extra_keys = ["strength","image_weight","cond_aug","noise_schedule","seed",
                  "motion_bucket_id","cond_aug_scale"]
    extra = {k: ip[k] for k in extra_keys if k in ip}
    # image selection
    img_b64 = ip.get("image_base64")
    pil = None
    if img_b64:
        pil = b64_to_pil(img_b64)
        print(f"[i2v] ref image (request): {pil.size} mode={pil.mode}")
    elif use_last:
        pil = load_last_image(user_id, session_id)
        if pil is not None:
            print(f"[i2v] ref image (last): {pil.size} mode={pil.mode}")
        else:
            print("[i2v] requested use_last_image but no last image found")

    use_i2v = (action.startswith("i2v") or (pil is not None))

    print(f"[route] action={action!r} use_i2v={use_i2v} has_img={pil is not None} len_b64={(len(img_b64) if img_b64 else 0)}")

    pipe = get_pipe()
    enable_memory_savers(pipe)

    # robust kwarg matching to the pipeline's __call__
    call_vars = getattr(pipe.__call__, "__code__", None)
    names = call_vars.co_varnames if call_vars else ()
    kwargs: Dict[str, Any] = dict(
        prompt = prompt,
        width = width,
        height = height,
        num_frames = frames_n,
        num_inference_steps = steps,
        guidance_scale = cfg,
    )
    # pass image to the correct parameter name
    if pil is not None:
        if "image" in names: kwargs["image"] = pil
        elif "img" in names: kwargs["img"] = pil
        else: kwargs["image"] = pil

    # pass optional extras only if supported
    for k, v in extra.items():
        if k in names:
            kwargs[k] = v

    # run
    out = pipe(**kwargs)

    # normal frames
    frames, fps_out = normalize_video_output(out, fps)
    if not frames:
        raise RuntimeError("Pipeline returned empty frames; cannot encode video.")

    # save last image if requested
    last_path = None
    if save_last:
        try:
            last_frame = frames[-1]
            last_path = save_last_image(last_frame, user_id, session_id)
            print(f"[last] saved last frame: {last_path}")
        except Exception as e:
            print(f"[last][warn] cannot save last frame: {e}")

    # encode to mp4
    mp4_bytes = frames_to_mp4_bytes(frames, fps_out)
    b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    return {
        "video_b64": b64,
        "fps": fps_out,
        "frames": len(frames),
        "last_image_path": last_path,
        "used_last_image": bool(use_last and pil is not None and not img_b64),
        "route": "i2v" if use_i2v else "t2v",
    }

runpod.serverless.start({"handler": handler})
