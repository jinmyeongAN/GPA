# LLM Server

```LLM``` folder contains codes for hosting LLM server.

## Environment Settings

**Create conda environment**

```$ conda create -y -n trans_pytorch_2.0 python=3.11```

```$ conda activate trans_pytorch_2.0```

**Install torch and huggingface packages**

```$ conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia```

```$ conda install -y -c conda-forge transformers=4.34.0 tokenizers=0.14.0 datasets=2.12.0 accelerate```

**Install packages for hosting**

```$ conda install -c conda-forge fastapi uvicorn pydantic```

**(Optional, can be changed) Install OCR package**

```$ pip install pix2tex```

## Start LLM hosting server

TODO