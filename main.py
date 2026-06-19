import sys
import types
from unittest.mock import MagicMock

try:
    import torchvision
except ImportError:
    m = types.ModuleType('torchvision')
    m.__spec__ = sys.__spec__
    t = types.ModuleType('torchvision.transforms')
    t.InterpolationMode = MagicMock()
    m.transforms = t
    sys.modules['torchvision'] = m
    sys.modules['torchvision.transforms'] = t

import os
import requests
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoModelForImageSegmentation

app = FastAPI(title="Miss Galaxia GPU Image Processor")

model = None
device = None
template_img = None

POSTER_URL = "https://beachpleaseapp.b-cdn.net/site/Galaxia_Colaj_3_4%20(1080x1440).jpg"
TEMPLATE_PATH = "/tmp/afis-galaxia.jpg"

class ProcessRequest(BaseModel):
    imageUrl: str
    scale: float = 0.7
    quality: int = 85

@app.on_event("startup")
def load_resources():
    global model, device, template_img
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BiRefNet-portrait model on device: {device}...")
    
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet-portrait", 
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    print(f"Downloading background poster template from {POSTER_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(POSTER_URL, headers=headers, timeout=15)
        if response.status_code == 200:
            template_img = Image.open(BytesIO(response.content)).convert("RGBA")
            print(f"Template loaded successfully. Dimensions: {template_img.size}")
        else:
            print(f"Warning: Failed to fetch template. Status: {response.status_code}")
    except Exception as e:
        print(f"Warning: Exception fetching template on startup: {e}")

@app.get("/")
@app.get("/health")
def health_check():
    has_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    return {
        "status": "ok",
        "device": str(device),
        "gpu_available": has_gpu,
        "gpu_name": device_name,
        "template_loaded": template_img is not None
    }

def run_birefnet(subject_img):
    orig_im = subject_img.convert("RGB")
    w, h = orig_im.size
    
    im_resize = orig_im.resize((1024, 1024), Image.Resampling.LANCZOS)
    im_data = np.array(im_resize)
    
    im_tensor = torch.tensor(im_data, dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    im_tensor = (im_tensor - mean) / std
    im_tensor = im_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(im_tensor)
        if isinstance(preds, (list, tuple)):
            pred_tensor = preds[-1]
        else:
            pred_tensor = preds
        pred_tensor = pred_tensor.sigmoid().squeeze().cpu().numpy()
        
    mask_data = (pred_tensor * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_data).resize((w, h), Image.Resampling.BILINEAR)
    
    cutout = Image.new("RGBA", (w, h))
    cutout.paste(orig_im, (0, 0), mask=mask_img)
    return cutout

@app.post("/process")
def process_image(request: ProcessRequest):
    global template_img
    
    if template_img is None:
        try:
            print("Template was not loaded on startup. Downloading now...")
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(POSTER_URL, headers=headers, timeout=15)
            if response.status_code == 200:
                template_img = Image.open(BytesIO(response.content)).convert("RGBA")
            else:
                raise HTTPException(status_code=500, detail="Failed to download poster template from storage.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading poster template: {str(e)}")

    try:
        print(f"Fetching subject image from: {request.imageUrl}")
        headers = {"User-Agent": "Mozilla/5.0"}
        subject_res = requests.get(request.imageUrl, headers=headers, timeout=15)
        if subject_res.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch subject image. Status: {subject_res.status_code}")
        subject_img = Image.open(BytesIO(subject_res.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image URL or download failed: {str(e)}")

    orig_w, orig_h = subject_img.size
    max_input_dim = 1024
    if max(orig_w, orig_h) > max_input_dim:
        if orig_w > orig_h:
            new_w = max_input_dim
            new_h = int(orig_h * (max_input_dim / orig_w))
        else:
            new_h = max_input_dim
            new_w = int(orig_w * (max_input_dim / orig_h))
        subject_img = subject_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    try:
        cutout = run_birefnet(subject_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

    bbox = cutout.getbbox()
    if not bbox:
        raise HTTPException(status_code=400, detail="No subject detected in the image.")
    
    cropped_cutout = cutout.crop(bbox)
    c_w, c_h = cropped_cutout.size

    bg_img = template_img.copy()
    bg_w, bg_h = bg_img.size
    
    target_h = int(bg_h * request.scale)
    aspect_ratio = c_w / c_h
    target_w = int(target_h * aspect_ratio)
    
    min_allowed_w = int(bg_w * 0.55)
    max_allowed_h = int(bg_h * 0.85)
    
    if target_w < min_allowed_w:
        new_target_w = min_allowed_w
        new_target_h = int(new_target_w / aspect_ratio)
        if new_target_h <= max_allowed_h:
            target_w = new_target_w
            target_h = new_target_h
        else:
            target_h = max_allowed_h
            target_w = int(target_h * aspect_ratio)

    resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    pos_x = (bg_w - target_w) // 2
    pos_y = bg_h - target_h
    
    bg_img.paste(resized_subject, (pos_x, pos_y), mask=resized_subject)
    final_img = bg_img.convert("RGB")

    out_io = BytesIO()
    final_img.save(out_io, "WEBP", quality=request.quality, method=6)
    out_io.seek(0)
    
    return Response(content=out_io.getvalue(), media_type="image/webp")
