FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/cache
RUN mkdir -p /app/cache

RUN python -c "import sys, types; from unittest.mock import MagicMock; m = types.ModuleType('torchvision'); m.__spec__ = sys.__spec__; t = types.ModuleType('torchvision.transforms'); t.InterpolationMode = MagicMock(); m.transforms = t; sys.modules['torchvision'] = m; sys.modules['torchvision.transforms'] = t; from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet-portrait', trust_remote_code=True)"

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
