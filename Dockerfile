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

RUN python -c "from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet-portrait', trust_remote_code=True)"

# Pre-download u2net_human_seg ONNX model for CPU fallback
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')"

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
