import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
from rembg import remove, new_session

app = FastAPI(title="Miss Galaxia CPU Image Processor")

template_img = None
rembg_session = None

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")
POSTER_URL = "https://beachpleaseapp.b-cdn.net/site/Galaxia_Colaj_3_4%20(1080x1440).jpg"

class ProcessRequest(BaseModel):
    imageUrl: str
    scale: float = 0.7
    quality: int = 85

@app.on_event("startup")
def load_resources():
    global template_img, rembg_session
    
    print("Loading rembg u2net_human_seg session on CPU...")
    rembg_session = new_session("u2net_human_seg")
    print("Rembg session loaded successfully!")
    
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
    return {
        "status": "ok",
        "device": "CPU",
        "template_loaded": template_img is not None
    }

@app.post("/process")
def process_image(request: ProcessRequest, authorization: str = Header(None)):
    global template_img, rembg_session
    
    if API_BEARER_TOKEN:
        if not authorization or authorization != f"Bearer {API_BEARER_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")
    
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
        print("Running u2net_human_seg on CPU...")
        cutout = remove(subject_img, session=rembg_session)
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
