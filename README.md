# Qwen Imaging API
Educational imaging API server using Qwen+Nunchaku (generation, editing with OpenAI-compatible endpoints.

  * Nunchakuエンジン(v1.0)を用いたQwen-Image/Qwen-Image-Edit（lightning）サーバです
  * VRAM消費を抑えつつ，高速な画像生成が可能です．
  * OpenAI-Image API互換のエンドポイントを持っているので，「OpenWebUI」などからも利用できます．
  * Image-Editの入力画像は，ローカルファイルだけでなく，ネットワーク上のファイルも指定可能です．
       
  * 実行には https://github.com/takago/flux_imaging_api/blob/main/image_file_server.py も必要です．



  --------------------------------------------------------------------------------------------------

## インストール・環境構築 ##

```bash
$ conda create -n nunchaku -c conda-forge python=3.11 gcc_linux-64=11 gxx cuda-toolkit=12.8 cmake
$ conda activate nunchaku
(nunchaku)$ pip install uv
(nunchaku)$ uv pip install "torch==2.8" torchvision --index-url https://download.pytorch.org/whl/cu128
(nunchaku)$ uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
(nunchaku)$ uv pip install triton sageattention $ uv pip install controlnet_aux dwpose matplotlib
(nunchaku)$ uv pip install gguf protobuf imageio imageio-ffmpeg
(nunchaku)$ uv pip install "accelerate>=0.26.0" transformers bitsandbytes peft sentencepiece
(nunchaku)$ uv pip install git+https://github.com/huggingface/diffusers.git "peft>=0.17.0"
(nunchaku)$ uv pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.8-cp311-cp311-linux_x86_64.whl 
(nunchaku)$ uv pip install uvicorn httpx fastapi python-multipart
```


## 起動方法 ##

1.  ファイルサーバーを起動：
```bash
    (nunchaku)$ uvicorn image_file_server:app --host 127.0.0.1 --port 8484
```
2.  画像生成 API サーバーを起動：
```bash
    (nunchaku)$ FILE_SERVER="http://localhost:8484" uvicorn qwen_nunchaku_imaging_api:app --host 127.0.0.1 --port 8444
```
（リモートから使う場合はCaddy等でSSL化した方がよいでしょう）


## 画像生成・編集の方法 ##
 
  https://github.com/takago/flux_imaging_api を参考にしてください．


----------
## 深謝 ##
- Qwen-Image: https://huggingface.co/Qwen/Qwen-Image
- Qwen-Image-Edit: https://huggingface.co/Qwen/Qwen-Image-Edit
- diffusers: https://github.com/huggingface/diffusers
- Nunchaku: https://github.com/nunchaku-tech/nunchaku
- OpenAI: https://openai.com/
- Hugging Face: https://huggingface.co/
