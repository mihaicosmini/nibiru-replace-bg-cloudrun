import os
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
from rembg import remove, new_session

app = FastAPI(title="Miss Galaxia CPU Image Processor")

template_img = None
rembg_session = None

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")
POSTER_URL = "https://beachpleaseapp.b-cdn.net/miss-galaxia/template.webp"

class ProcessRequest(BaseModel):
    imageUrl: str
    scale: float = 0.7
    quality: int = 85
    instagramHandle: str = None

@app.on_event("startup")
def load_resources():
    global template_img, rembg_session
    
    print("Loading rembg u2net_human_seg session on CPU...")
    rembg_session = new_session("u2net_human_seg")
    print("Rembg session loaded successfully!")
    
    # Try loading template.webp locally first
    local_template_path = os.path.join(os.path.dirname(__file__), "template.webp")
    if os.path.exists(local_template_path):
        print(f"Loading local template from {local_template_path}...")
        try:
            template_img = Image.open(local_template_path).convert("RGBA")
            print(f"Template loaded successfully from local file. Dimensions: {template_img.size}")
            return
        except Exception as e:
            print(f"Error loading local template file: {e}. Falling back to download...")
            
    print(f"Downloading background poster template from {POSTER_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(POSTER_URL, headers=headers, timeout=15)
        if response.status_code == 200:
            template_img = Image.open(BytesIO(response.content)).convert("RGBA")
            print(f"Template loaded successfully from URL. Dimensions: {template_img.size}")
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
            print("Template was not loaded on startup. Attempting local load fallback...")
            local_template_path = os.path.join(os.path.dirname(__file__), "template.webp")
            if os.path.exists(local_template_path):
                template_img = Image.open(local_template_path).convert("RGBA")
            else:
                print("Downloading template now...")
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(POSTER_URL, headers=headers, timeout=15)
                if response.status_code == 200:
                    template_img = Image.open(BytesIO(response.content)).convert("RGBA")
                else:
                    raise HTTPException(status_code=500, detail="Failed to download poster template from storage.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading/downloading poster template: {str(e)}")

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
    aspect_ratio = c_w / c_h

    bg_img = template_img.copy()
    bg_w, bg_h = bg_img.size
    
    target_h = int(bg_h * request.scale)
    target_w = int(target_h * aspect_ratio)
    
    # ENFORCE BOUNDS: Top 350px must be clean, so max height of subject is bg_h - 350 = 1090
    max_allowed_h = bg_h - 350
    if target_h > max_allowed_h:
        print(f"Scaling down subject: height {target_h} is larger than max allowed {max_allowed_h}")
        target_h = max_allowed_h
        target_w = int(target_h * aspect_ratio)

    resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    pos_x = (bg_w - target_w) // 2
    pos_y = bg_h - target_h
    
    bg_img.paste(resized_subject, (pos_x, pos_y), mask=resized_subject)

    # 3b. Create and apply a smooth linear black gradient at the bottom (fade to black)
    try:
        y_start = 980
        gradient_h = bg_h - y_start
        gradient_band = Image.new("RGBA", (1, gradient_h))
        for y in range(gradient_h):
            alpha = int((y / gradient_h) * 180)
            gradient_band.putpixel((0, y), (0, 0, 0, alpha))
            
        gradient_overlay = gradient_band.resize((bg_w, gradient_h))
        bg_img.paste(gradient_overlay, (0, y_start), mask=gradient_overlay)
    except Exception as grad_err:
        print(f"Warning: Exception applying gradient overlay: {grad_err}")

    # 4. Draw texts (footer static texts + dynamic instagram username)
    try:
        draw = ImageDraw.Draw(bg_img)
        font_dir = os.path.dirname(__file__)
        font_path = os.path.join(font_dir, "SparTakus Round.ttf")
        
        if os.path.exists(font_path):
            # Draw Footer Line A: GET YOUR TICKETS NOW (size 24 at 123px from bottom)
            text_a = "GET YOUR TICKETS NOW"
            font_a = ImageFont.truetype(font_path, 24)
            bbox_a = draw.textbbox((0, 0), text_a, font=font_a)
            w_a = bbox_a[2] - bbox_a[0]
            h_a = bbox_a[3] - bbox_a[1]
            x_a = (bg_w - w_a) // 2
            y_a = bg_h - 123 - h_a
            draw.text((x_a, y_a), text_a, font=font_a, fill=(255, 255, 255))
            
            # Draw Footer Line B: NIBIRU.NET/GALAXIA (size 40 at 75px from bottom)
            text_b = "NIBIRU.NET/GALAXIA"
            font_b = ImageFont.truetype(font_path, 40)
            bbox_b = draw.textbbox((0, 0), text_b, font=font_b)
            w_b = bbox_b[2] - bbox_b[0]
            h_b = bbox_b[3] - bbox_b[1]
            x_b = (bg_w - w_b) // 2
            y_b = bg_h - 75 - h_b
            draw.text((x_b, y_b), text_b, font=font_b, fill=(255, 255, 255))
            
            # Draw dynamic instagram handle
            if request.instagramHandle:
                # Text 1: MISS & MISTER GALAXIA
                text1 = "MISS & MISTER GALAXIA"
                font1 = ImageFont.truetype(font_path, 35)
                
                # Draw at 318px from bottom (y = 1440 - 318 = 1122)
                y1 = bg_h - 318
                bbox1 = draw.textbbox((0, 0), text1, font=font1)
                text1_w = bbox1[2] - bbox1[0]
                text1_h = bbox1[3] - bbox1[1]
                x1 = (bg_w - text1_w) // 2
                
                draw.text((x1, y1), text1, font=font1, fill=(255, 255, 255))
                
                # Text 2: Instagram Username
                raw_username = request.instagramHandle.strip().lstrip("@")
                display_username = raw_username
                
                username_len = len(raw_username)
                if username_len <= 12:
                    font_size = 99
                else:
                    font_size = max(38, int((90 - (username_len - 12) * 10) * 1.1))
                    
                # Safety Check: scale down if the text is wider than 900px
                max_text_width = 900
                font2 = ImageFont.truetype(font_path, font_size)
                bbox2 = draw.textbbox((0, 0), display_username, font=font2)
                text2_w = bbox2[2] - bbox2[0]
                
                while text2_w > max_text_width and font_size > 25:
                    font_size -= 2
                    font2 = ImageFont.truetype(font_path, font_size)
                    bbox2 = draw.textbbox((0, 0), display_username, font=font2)
                    text2_w = bbox2[2] - bbox2[0]
                    
                text2_h = bbox2[3] - bbox2[1]
                x2 = (bg_w - text2_w) // 2
                
                # Calculate gap dynamically based on font size to align visual centers
                gap = int(71 - 0.5 * font_size)
                y2 = y1 + text1_h + gap
                
                draw.text((x2, y2), display_username, font=font2, fill=(255, 255, 255))
                print(f"Drew instagram handle: {display_username} (size {font_size})")
        else:
            print(f"Warning: Font file not found at {font_path}. Skipping text overlays.")
    except Exception as e:
        print(f"Warning: Exception drawing text overlays: {e}")

    final_img = bg_img.convert("RGB")

    out_io = BytesIO()
    final_img.save(out_io, "WEBP", quality=request.quality, method=6)
    out_io.seek(0)
    
    return Response(content=out_io.getvalue(), media_type="image/webp")
