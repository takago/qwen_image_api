# Qwen Imaging API
Educational imaging API server using Qwen+Nunchaku (generation, editing with OpenAI-compatible endpoints.

  * Nunchakuエンジンを用いたQwen-Image/Qwen-Image-Editサーバを作りました．
  * VRAM消費を抑えつつ，早い画像生成が期待できます．
  * OpenAI-Image API互換のエンドポイントを持っているので，OpenWebUIなどからも利用できます．
  * CaddyでSSL接続して使うことをおすすめします．
  * 実行には https://github.com/takago/flux_imaging_api/blob/main/image_file_server.py も必要です．



  --------------------------------------------------------------------------------------------------

## インストール・環境構築 ##

```bash
$ conda create -n nunchaku -c conda-forge python=3.11 gcc_linux-64=11 gxx cuda-toolkit=12.8 cmake
$ conda activate nunchaku $ pip install uv
$ uv pip install "torch==2.8" torchvision --index-url https://download.pytorch.org/whl/cu128
$ uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
$ uv pip install triton sageattention $ uv pip install controlnet_aux dwpose matplotlib
$ uv pip install gguf protobuf imageio imageio-ffmpeg
$ uv pip install "accelerate>=0.26.0" transformers bitsandbytes peft sentencepiece
$ uv pip install git+https://github.com/huggingface/diffusers.git "peft>=0.17.0"
$ uv pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.8-cp311-cp311-linux_x86_64.whl 
$ uv pip install uvicorn httpx fastapi python-multipart
```


## 起動方法 ##

1.  ファイルサーバーを起動：
```bash
    uvicorn image_file_server:app –host 0.0.0.0 –port 8484
```
3.  画像生成 API サーバーを起動：
```bash
    FILE_SERVER=“http://localhost:8484” uvicorn qwen_nunchaku_imaging_api:app –host 127.0.0.1 –port 8444
```

## 画像生成・編集の方法 ##
  curlをつかった使い方を https://github.com/takago/flux_imaging_api に紹介していますので，参考にしてください．


----------
## 謝辞 ##

本プロジェクトの開発にあたり，以下のプロジェクト・コミュニティの成果に深く感謝いたします．
- Qwen: 高性能な基盤モデルの提供に感謝します．
- Nunchaku: 効率的な推論エンジンの提供に感謝します．
- OpenAI: API 仕様とエコシステムの提供に感謝します．


