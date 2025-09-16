# ============================================================
#  qwen_imaging_api.py
#
#  Imaging API Server with Qwen-Image Pipelines
#    - Generate (Qwen-Image)
#    - Edit (Qwen-Image-Edit)
#    - Original API + OpenAI Image API compatible endpoints
#
#  Author: Daisuke Takago (Kanazawa Institute of Technology)
#  Supported by ChatGPT (OpenAI)
#
#  Development date: September 2025
#  License: MIT
# ============================================================

import io
import uuid
import time
import os
import base64
import httpx
from fastapi import FastAPI, Form, File, UploadFile, Body
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

import torch
from diffusers import DiffusionPipeline


# ファイルサーバのベースURL（環境変数から取得，存在しない場合は None）
FILE_SERVER = os.getenv("FILE_SERVER", None)

# --- デフォルト設定 ---
DEFAULTS = {
    "edit": {"true_cfg_scale": 1.0, "num_inference_steps": 8},
    "generate": {"true_cfg_scale": 1.0, "num_inference_steps": 8},
}

# --- モデル/LoRA情報 ---
MODEL_INFO = {
    "edit": {
        "base_model": "Qwen/Qwen-Image-Edit",
        "loras": [],
    },
    "generate": {
        "base_model": "Qwen/Qwen-Image",
        "loras":[],
    },
}

# --- 生成用パイプライン (Qwen-Image) ---
import math
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
 
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the model
transformer_gen = NunchakuQwenImageTransformer2DModel.from_pretrained(
        f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r128-qwen-image-lightningv1.1-8steps.safetensors"
)
pipe_gen = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    transformer=transformer_gen,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)
 
transformer_gen.set_offload(
        True,
        use_pin_memory=False, 
        num_blocks_on_gpu=1
) 
pipe_gen._exclude_from_cpu_offload.append("transformer")
pipe_gen.enable_sequential_cpu_offload()

# --- 編集用パイプライン (Qwen-Image-Edit) ---
from diffusers import QwenImageEditPipeline

transformer_edit = NunchakuQwenImageTransformer2DModel.from_pretrained(
     f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r128-qwen-image-edit-lightningv1.0-8steps.safetensors")

 
pipe_edit = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    transformer=transformer_edit,
    text_encoder=pipe_gen.text_encoder, # テキストエンコーダは共有
    scheduler=scheduler, 
    torch_dtype=torch.bfloat16
)

transformer_edit.set_offload(
        True,
        use_pin_memory=False,
        num_blocks_on_gpu=1
) 
pipe_edit._exclude_from_cpu_offload.append("transformer")
pipe_edit.enable_sequential_cpu_offload()


# --- モード判定 ---
def detect_mode(input_image_url, prompt, init_image):
    if (input_image_url or init_image) and prompt:
        return "edit"
    elif not (input_image_url or init_image) and prompt:
        return "generate"
    else:
        return None


# --- 乱数生成器 ---
def get_generator(seed: int | None, i: int = 0):
    if seed is None:
        gen_seed = int(torch.seed())  # ランダム
    else:
        gen_seed = seed + i
    return torch.Generator().manual_seed(gen_seed), gen_seed


# ---------- 共通処理 ----------
async def run_pipeline(input_image_url, prompt, bearer_token, seed, width, height,
                       true_cfg_scale, num_inference_steps, input_file: UploadFile = None, i: int = 0,
                       negative_prompt: str = " "):

    init_image = None
    if input_file:
        data = await input_file.read()
        init_image = Image.open(io.BytesIO(data)).convert("RGB")
        input_image_url = None

    pipeline_mode = detect_mode(input_image_url, prompt, init_image)
    if pipeline_mode is None:
        return None, None, None

    generator, used_seed = get_generator(seed, i)

    if pipeline_mode == "edit":
        if init_image is None and input_image_url:
            headers = {}
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"
            async with httpx.AsyncClient(verify=False) as client:
                resp = await client.get(input_image_url, headers=headers)
                resp.raise_for_status()
                init_image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        if init_image is None:
            raise RuntimeError("No valid init_image for edit mode")
        
        result = pipe_edit(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width or init_image.size[0],
            height=height or init_image.size[1],
            num_inference_steps=num_inference_steps or DEFAULTS["edit"]["num_inference_steps"],
            true_cfg_scale=true_cfg_scale or DEFAULTS["edit"]["true_cfg_scale"],
            generator=generator,
        )
    else:  # generate
        result = pipe_gen(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width or 1024,
            height=height or 1024,
            num_inference_steps=num_inference_steps or DEFAULTS["generate"]["num_inference_steps"],
            true_cfg_scale=true_cfg_scale or DEFAULTS["generate"]["true_cfg_scale"],
            generator=generator,
        )

    processed_img = result.images[0]
    out_buf = io.BytesIO()
    processed_img.save(out_buf, format="PNG")
    out_buf.seek(0)
    return processed_img, out_buf, used_seed


# ============================================================
#                 FastAPI アプリケーション
# ============================================================

app = FastAPI()

# ---------- オリジナルエンドポイント ----------
@app.post("/process")
async def process_image(
    input_image_url: str = Form(None),
    prompt: str = Form(""),
    bearer_token: str = Form(None),
    seed: int = Form(None),
    width: int = Form(None),
    height: int = Form(None),
    true_cfg_scale: float = Form(None),
    num_inference_steps: int = Form(None),
    negative_prompt: str = Form(" "),
    input_file: UploadFile = File(None),
):
    img, buf, used_seed = await run_pipeline(
        input_image_url, prompt, bearer_token, seed,
        width, height, true_cfg_scale, num_inference_steps, input_file,
        negative_prompt=negative_prompt,
    )
    if img is None:
        return JSONResponse({"error": "invalid input"}, status_code=400)

    # 出力画像サイズ
    output_width, output_height = img.width, img.height

    # 実行モード
    pipeline_mode = detect_mode(input_image_url, prompt, input_file)

    # モデル情報
    model_info = {
        "model": MODEL_INFO[pipeline_mode]["base_model"],
        "loras": MODEL_INFO[pipeline_mode]["loras"],
    }

    # 整理したレスポンス
    metadata = {
        "mode": pipeline_mode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": used_seed,
        "true_cfg_scale": true_cfg_scale or DEFAULTS[pipeline_mode]["true_cfg_scale"],
        "num_inference_steps": num_inference_steps or DEFAULTS[pipeline_mode]["num_inference_steps"],
        "width": output_width,
        "height": output_height,
        **model_info,
    }

    if FILE_SERVER:
        upload_url = f"{FILE_SERVER}/upload"
        files = {"file": (f"{uuid.uuid4()}.png", buf, "image/png")}
        async with httpx.AsyncClient(verify=False) as client:
            up_resp = await client.post(upload_url, files=files)
            up_resp.raise_for_status()
            up_result = up_resp.json()
        metadata["result_image_url"] = f"{FILE_SERVER}{up_result['url']}"
        return metadata
    else:
        metadata["result_image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        return metadata


@app.post("/process/raw")
async def process_image_raw(
    input_image_url: str = Form(None),
    prompt: str = Form(""),
    bearer_token: str = Form(None),
    seed: int = Form(None),
    width: int = Form(None),
    height: int = Form(None),
    true_cfg_scale: float = Form(None),
    num_inference_steps: int = Form(None),
    negative_prompt: str = Form(" "),
    input_file: UploadFile = File(None),
):
    img, buf, used_seed = await run_pipeline(input_image_url, prompt, bearer_token, seed, width, height,
                                             true_cfg_scale, num_inference_steps, input_file,
                                             negative_prompt=negative_prompt)
    if img is None:
        return JSONResponse({"error": "invalid input"}, status_code=400)
    return StreamingResponse(buf, media_type="image/png")


# ---------- OpenAI Image API 互換エンドポイント ----------
@app.post("/v1/images/generations")
async def openai_image_generate(body: dict = Body(...)):
    prompt = body.get("prompt")
    negative_prompt = body.get("negative_prompt", " ")
    n = body.get("n", 1)
    size = body.get("size", "1024x1024")
    response_format = body.get("response_format", "url")
    seed = body.get("seed")
    true_cfg_scale = body.get("true_cfg_scale")
    num_inference_steps = body.get("num_inference_steps")

    width, height = map(int, size.split("x"))
    results = []
    for i in range(n):
        img, buf, used_seed = await run_pipeline(None, prompt, None, seed, width, height,
                                                 true_cfg_scale, num_inference_steps, None, i,
                                                 negative_prompt=negative_prompt)
        if img is None:
            return JSONResponse({"error": "invalid input"}, status_code=400)
        if response_format == "b64_json":
            results.append({"b64_json": base64.b64encode(buf.getvalue()).decode("utf-8"), "seed": used_seed})
        else:
            if not FILE_SERVER:
                return JSONResponse({"error": "FILE_SERVER not configured"}, status_code=500)
            upload_url = f"{FILE_SERVER}/upload"
            files = {"file": (f"{uuid.uuid4()}.png", buf, "image/png")}
            async with httpx.AsyncClient(verify=False) as client:
                up_resp = await client.post(upload_url, files=files)
                up_resp.raise_for_status()
                up_result = up_resp.json()
            results.append({"url": f"{FILE_SERVER}{up_result['url']}", "seed": used_seed})
    return {"created": int(time.time()), "data": results}


@app.post("/v1/images/edits")
async def openai_image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url"),
    seed: int = Form(None),
    true_cfg_scale: float = Form(None),
    num_inference_steps: int = Form(None),
):
    width, height = map(int, size.split("x"))
    raw_bytes = await image.read()
    results = []
    for i in range(n):
        img_file = UploadFile(filename=image.filename, file=io.BytesIO(raw_bytes))
        img, buf, used_seed = await run_pipeline(None, prompt, None, seed, width, height,
                                                 true_cfg_scale, num_inference_steps,
                                                 img_file, i, negative_prompt=negative_prompt)
        if img is None:
            return JSONResponse({"error": "invalid input"}, status_code=400)
        if response_format == "b64_json":
            results.append({"b64_json": base64.b64encode(buf.getvalue()).decode("utf-8"), "seed": used_seed})
        else:
            if not FILE_SERVER:
                return JSONResponse({"error": "FILE_SERVER not configured"}, status_code=500)
            upload_url = f"{FILE_SERVER}/upload"
            files = {"file": (f"{uuid.uuid4()}.png", buf, "image/png")}
            async with httpx.AsyncClient(verify=False) as client:
                up_resp = await client.post(upload_url, files=files)
                up_resp.raise_for_status()
                up_result = up_resp.json()
            results.append({"url": f"{FILE_SERVER}{up_result['url']}", "seed": used_seed})
    return {"created": int(time.time()), "data": results}

