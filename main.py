import os
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from io import BytesIO
from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel

app = FastAPI(title="Miss Galaxia Cloudflare Image Processor")

template_img = None

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")
POSTER_URL = "https://beachpleaseapp.b-cdn.net/miss-galaxia/template.webp"

class ProcessRequest(BaseModel):
    imageUrl: str
    scale: float = 0.7
    quality: int = 100
    instagramHandle: str = None

@app.on_event("startup")
def load_resources():
    global template_img
    
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
        "template_loaded": template_img is not None,
        "version": "cfv2"
    }

def apply_edge_fade(img, left_fade=False, right_fade=False, fade_percentage=0.04):
    """
    Applies a smooth alpha gradient fade to the left and/or right edges of an RGBA PIL Image.
    """
    if not (left_fade or right_fade):
        return img
        
    w, h = img.size
    np_img = np.array(img)
    alpha = np_img[:, :, 3].astype(float)
    
    fade_w = int(w * fade_percentage)
    if fade_w < 5:
        fade_w = 5
        
    if left_fade:
        # Fade from x = 0 (alpha * 0) to x = fade_w (alpha * 1)
        for x in range(fade_w):
            factor = x / fade_w
            alpha[:, x] = alpha[:, x] * factor
            
    if right_fade:
        # Fade from x = w - fade_w (alpha * 1) to x = w - 1 (alpha * 0)
        for x in range(fade_w):
            factor = x / fade_w
            col = w - 1 - x
            alpha[:, col] = alpha[:, col] * factor
            
    np_img[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

@app.post("/process")
def process_image(request: ProcessRequest, authorization: str = Header(None)):
    global template_img
    
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

    # Request segmented cutout from Cloudflare Images
    try:
        print(f"Requesting Cloudflare background removal for: {request.imageUrl}")
        cf_url = f"https://cazare-beach-please.ro/cdn-cgi/image/width=1024,height=1024,fit=scale-down,segment=foreground/{request.imageUrl}"
        cf_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        cf_res = requests.get(cf_url, headers=cf_headers, timeout=20)
        if cf_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Cloudflare background removal failed with status {cf_res.status_code}")
        
        print("Successfully retrieved cutout from Cloudflare!")
        cutout = Image.open(BytesIO(cf_res.content)).convert("RGBA")
        
        # Apply High-Key Beach Relighting (V4)
        r, g, b, a = cutout.split()
        r_new = r.point(lambda i: min(255, int(i * 1.05)))
        b_new = b.point(lambda i: int(i * 0.96))
        warmed = Image.merge("RGBA", (r_new, g, b_new, a))
        contrast = ImageEnhance.Contrast(warmed).enhance(1.15)
        subject_img = ImageEnhance.Brightness(contrast).enhance(1.08)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Cloudflare background removal request failed: {str(e)}")

    bbox = cutout.getbbox()
    if not bbox:
        raise HTTPException(status_code=400, detail="No subject detected in the image.")
    
    cropped_cutout = cutout.crop(bbox)
    c_w, c_h = cropped_cutout.size
    aspect_ratio = c_w / c_h

    bg_img = template_img.copy()
    bg_w, bg_h = bg_img.size
    
    # 5. Resize and position subject using dynamic auto-framing scaling
    # ENFORCE BOUNDS: Top 350px must be clean, so max height of subject is bg_h - 350 = 1090
    max_allowed_h = bg_h - 350
    effective_max_h = min(max_allowed_h, int(bg_h * 0.90))
    
    # Check if this is a selfie / cut-off image touching the borders of the pre-scaled image
    # We use a threshold of 1.5% of the dimensions to detect if it touches the edges
    tolerance_w = max(6, int(subject_img.width * 0.015))
    tolerance_h = max(6, int(subject_img.height * 0.015))
    
    left_touch = (bbox[0] <= tolerance_w)
    right_touch = (bbox[2] >= subject_img.width - tolerance_w)
    bottom_touch = (bbox[3] >= subject_img.height - tolerance_h)
    
    print(f"Border detection: left_touch={left_touch} (left={bbox[0]}/{subject_img.width}), "
          f"right_touch={right_touch} (right={bbox[2]}/{subject_img.width}), "
          f"bottom_touch={bottom_touch} (lower={bbox[3]}/{subject_img.height})")
          
    is_selfie = left_touch or right_touch or bottom_touch
    
    # Apply soft 4% fade-out to cut-off sides
    cropped_cutout = apply_edge_fade(cropped_cutout, left_fade=left_touch, right_fade=right_touch, fade_percentage=0.04)
    c_w, c_h = cropped_cutout.size
    
    if is_selfie:
        print("Selfie / cut-off subject detected. Applying special framing rules.")
        # Case A: Subject is cut off on BOTH left and right sides
        if left_touch and right_touch:
            print("Subject touches both left and right sides. Scaling to full poster width and cropping bottom.")
            target_w = bg_w
            target_h = int(target_w / aspect_ratio)
            pos_x = 0
            pos_y = bg_h - min(target_h, effective_max_h)
            
            resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
            if target_h > effective_max_h:
                resized_subject = resized_subject.crop((0, 0, target_w, effective_max_h))
                target_h = effective_max_h
            
        # Case B: Subject touches the left side only (cut-off on the left)
        elif left_touch:
            print("Subject touches left side. Aligning to left edge of poster.")
            target_h = int(bg_h * max(request.scale, 0.8))  # Scale up slightly for selfies (min 80%)
            if target_h > max_allowed_h:
                target_h = max_allowed_h
            target_w = int(target_h * aspect_ratio)
            if target_w > bg_w:
                target_w = bg_w
                target_h = int(target_w / aspect_ratio)
            pos_x = 0
            pos_y = bg_h - target_h
            resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
        # Case C: Subject touches the right side only (cut-off on the right)
        elif right_touch:
            print("Subject touches right side. Aligning to right edge of poster.")
            target_h = int(bg_h * max(request.scale, 0.8))  # Scale up slightly for selfies (min 80%)
            if target_h > max_allowed_h:
                target_h = max_allowed_h
            target_w = int(target_h * aspect_ratio)
            if target_w > bg_w:
                target_w = bg_w
                target_h = int(target_w / aspect_ratio)
            pos_x = bg_w - target_w
            pos_y = bg_h - target_h
            resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
        # Case D: Subject touches only the bottom (typical portrait close-up)
        else:
            print("Subject touches bottom side only. Centering at bottom.")
            target_h = int(bg_h * max(request.scale, 0.78))  # Scale up slightly (min 78%)
            if target_h > max_allowed_h:
                target_h = max_allowed_h
            target_w = int(target_h * aspect_ratio)
            if target_w > bg_w:
                target_w = bg_w
                target_h = int(target_w / aspect_ratio)
            pos_x = (bg_w - target_w) // 2
            pos_y = bg_h - target_h
            resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
    else:
        # Standard non-cut-off subject (centered, floating)
        target_h = int(bg_h * request.scale)
        if target_h > max_allowed_h:
            target_h = max_allowed_h
        target_w = int(target_h * aspect_ratio)
        
        # Auto-zoom/framing adjustment
        min_allowed_w = int(bg_w * 0.55)
        
        if target_w < min_allowed_w:
            new_target_w = min_allowed_w
            new_target_h = int(new_target_w / aspect_ratio)
            effective_max_h_std = min(max_allowed_h, int(bg_h * 0.85))
            
            if new_target_h <= effective_max_h_std:
                target_w = new_target_w
                target_h = new_target_h
                print(f"Auto-framing: scaled up subject width to {target_w}px (height {target_h}px) to fill horizontal space.")
            else:
                target_h = effective_max_h_std
                target_w = int(target_h * aspect_ratio)
                print(f"Auto-framing: subject is extremely narrow. Capped height at {target_h}px (width {target_w}px) to avoid overflow.")
                
        pos_x = (bg_w - target_w) // 2
        pos_y = bg_h - target_h
        resized_subject = cropped_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)

    print(f"Final overlay size for subject: {target_w}x{target_h}")
    print(f"Positioning subject cutout at ({pos_x}, {pos_y}) on the template")
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
            # Draw dynamic instagram handle
            if request.instagramHandle:
                # Text 1: MISS & MISTER GALAXIA
                text1 = "MISS & MISTER GALAXIA"
                font1 = ImageFont.truetype(font_path, 35)
                
                # Draw at 230px from bottom (y = 1440 - 230 = 1210)
                y1 = bg_h - 230
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
