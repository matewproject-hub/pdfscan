from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import cv2
import torch
import re
import json

# -------------------------------------------------------------------
# OCR INIT
# -------------------------------------------------------------------

# Paddle Detection only
detector = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    rec=False,
    show_log=False,
    det_limit_side_len=3000
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(device)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATE_RE = re.compile(r'\d{2}[-/.]\d{2}[-/.]\d{4}')
DL_RE = re.compile(r'[A-Z]{2}\s?[0-9]{2}\s?[0-9]{11}', re.I)
PIN_RE = re.compile(r'\b\d{6}\b')
BG_RE = re.compile(r'(A|AB|B|O)[+-]', re.I)

VALID_BLOOD_GROUPS = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}

FIELD_ALIASES = {
    "name": "name",
    "dob": "date_of_birth",
    "date of birth": "date_of_birth",
    "issue date": "issue_date",
    "issuedate": "issue_date",
    "issue": "issue_date",
    "validity nt": "validity_nt",
    "validity tr": "validity_tr",
    "validity (nt)": "validity_nt",
    "validity (tr)": "validity_tr",
    "blood group": "blood_group",
    "bloodgroup": "blood_group",
    "address": "address",
    "s/d/w": "relative_name",
    "son/daughter/wife": "relative_name",
    "son/daughter/wife of": "relative_name"
}

LABELS = sorted(FIELD_ALIASES.keys(), key=len, reverse=True)

NOISE_WORDS = {
    "signature", "holder", "holders", "donor", "organ",
    "transport", "issuing", "authority", "dlr", "sign",
}

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r'[:;,.()\-]', '', text)
    return text.strip()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def is_noise(text):
    t = re.sub(r'[^a-zA-Z]', '', text.lower())
    if not t: return False
    for n in NOISE_WORDS:
        if n in t: return True
    return False

def looks_like_name(text):
    if any(ch.isdigit() for ch in text): return False
    words = text.split()
    alpha = [re.sub(r'[^a-zA-Z]', '', w) for w in words]
    alpha = [w for w in alpha if len(w) >= 2]
    return len(alpha) > 0

# -------------------------------------------------------------------
# IMAGE PREPROCESS
# -------------------------------------------------------------------

def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if w < 2500:
        scale = 2500 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return Image.fromarray(gray)

# -------------------------------------------------------------------
# TROCR RECOGNITION
# -------------------------------------------------------------------

def trocr_recognize(pil_crop):
    # Ensure grayscale for some pre-processing, but feed color/gray to TrOCR
    crop_np = np.array(pil_crop)
    
    # If 3 channels, convert to gray for thresholding but keep original for TrOCR? 
    # Actually, let's just use the original crop but maybe upscale it.
    if len(crop_np.shape) == 3:
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_np

    # Upscale for better recognition
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # We'll skip hard thresholding as it often hurts TrOCR
    final_crop = Image.fromarray(upscaled).convert("RGB")

    pixel_values = processor(images=final_crop, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return clean_text(text)

# -------------------------------------------------------------------
# BOX HELPERS
# -------------------------------------------------------------------

def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def build_boxes(raw_boxes, image):
    boxes = []
    heights = []
    for pts in raw_boxes:
        # PaddleOCR detector format: [[x,y], [x,y], [x,y], [x,y]]
        try:
            x1, y1, x2, y2 = bbox(pts)
        except Exception:
            # Fallback if points are already [x1, y1, x2, y2]
            if len(pts) == 4: x1, y1, x2, y2 = pts
            else: continue
            
        pad = 4
        crop = image.crop((
            max(0, int(x1 - pad)),
            max(0, int(y1 - pad)),
            min(image.width, int(x2 + pad)),
            min(image.height, int(y2 + pad))
        ))
        
        text = trocr_recognize(crop)
        if not text: continue
        
        box = {
            "text": text,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "h": y2 - y1
        }
        boxes.append(box)
        heights.append(y2 - y1)
        
    avg_h = sum(heights) / len(heights) if heights else 20
    return boxes, avg_h

# -------------------------------------------------------------------
# SPATIAL LOGIC
# -------------------------------------------------------------------

def same_row(a, b, avg_h):
    return abs(a["cy"] - b["cy"]) < avg_h * 0.6

def find_label(text):
    cleaned = normalize(text)
    for lbl in LABELS:
        if cleaned == normalize(lbl) or cleaned.startswith(normalize(lbl)):
            return lbl
    return None

def extract_fields(boxes, avg_h):
    output = {}
    ordered = sorted(boxes, key=lambda b: (b["y1"], b["x1"]))

    # DL Number (Global Search)
    all_text = " ".join([b["text"] for b in boxes])
    dl = DL_RE.search(all_text)
    if dl: output["license_number"] = {"value": dl.group(0).upper(), "confidence": 0.99}

    for label_box in ordered:
        matched = find_label(label_box["text"])
        if not matched: continue
        field = FIELD_ALIASES[matched]
        if field in output and field != "validity_nt": continue

        value = None
        
        # Name / Relative Name
        if field in ("name", "relative_name"):
            candidates = []
            for b in boxes:
                if b is label_box or is_noise(b["text"]) or not looks_like_name(b["text"]): continue
                v_gap = b["y1"] - label_box["y2"]
                if 0 <= v_gap <= avg_h * 2.5 and abs(b["cx"] - label_box["cx"]) < avg_h * 10:
                    candidates.append(b)
            if candidates:
                candidates.sort(key=lambda x: (x["y1"], x["x1"]))
                first_y = candidates[0]["cy"]
                same_line = [c["text"] for c in candidates if abs(c["cy"] - first_y) < avg_h * 0.5]
                value = " ".join(same_line)

        # Dates
        elif field in ("issue_date", "validity_nt", "validity_tr", "date_of_birth"):
            candidates = []
            for b in boxes:
                if b is label_box: continue
                m = DATE_RE.search(b["text"])
                if not m: continue
                if same_row(label_box, b, avg_h): candidates.append((0, abs(b["cx"] - label_box["cx"]), b))
                else:
                    gap = b["y1"] - label_box["y2"]
                    if 0 <= gap <= avg_h * 2: candidates.append((1, gap, b))
            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]))
                best_box = candidates[0][2]
                all_dates = DATE_RE.findall(best_box["text"])
                # Fuzzy index mapping
                idx = 0
                if field == "validity_nt": idx = 1
                elif field == "validity_tr": idx = 2
                value = all_dates[min(idx, len(all_dates)-1)]

        # Blood Group
        elif field == "blood_group":
            for b in boxes:
                if same_row(label_box, b, avg_h) and b["x1"] > label_box["x2"]:
                    bg = BG_RE.search(b["text"])
                    if bg:
                        value = bg.group(0).upper()
                        break

        # Address
        elif field == "address":
            parts = []
            for b in boxes:
                if b is label_box or is_noise(b["text"]): continue
                gap = b["y1"] - label_box["y2"]
                if 0 <= gap <= avg_h * 6 and abs(b["cx"] - label_box["cx"]) < avg_h * 15:
                    parts.append(b)
            if parts:
                parts.sort(key=lambda x: (x["y1"], x["x1"]))
                value = " ".join([p["text"] for p in parts if p["text"].lower() != "dlr"])

        if value:
            output[field] = {"value": clean_text(value), "confidence": 0.95}

    return output

def reconstruct_layout(boxes, avg_h):
    if not boxes: return ""
    boxes = sorted(boxes, key=lambda x: (x["cy"], x["x1"]))
    lines = []
    if not boxes: return ""
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b["cy"] - current[-1]["cy"]) < avg_h * 0.5: current.append(b)
        else:
            lines.append(current)
            current = [b]
    lines.append(current)
    rows = []
    for line in lines:
        line = sorted(line, key=lambda x: x["x1"])
        rows.append(" ".join([b["text"] for b in line]))
    return "\n".join(rows)

# -------------------------------------------------------------------
# MAIN ENGINE
# -------------------------------------------------------------------

def extract_text_from_image(image_input, preprocess=True, return_dict=False, **kwargs):
    try:
        if isinstance(image_input, str): image = Image.open(image_input).convert("RGB")
        else: image = image_input.convert("RGB")

        processed = preprocess_image(image) if preprocess else image
        detection = detector.ocr(np.array(processed), rec=False)
        if not detection or not detection[0]: return "No text could be extracted."
        
        raw_boxes = detection[0]
        # Use processed image for recognition to maintain consistency with detection
        boxes, avg_h = build_boxes(raw_boxes, processed)
        fields = extract_fields(boxes, avg_h)
        layout = reconstruct_layout(boxes, avg_h)

        if return_dict: return {"dynamic_fields": fields, "full_layout": layout}
        return f"--- EXTRACTED SPATIAL JSON ---\n{json.dumps(fields, indent=4)}\n\n--- STRUCTURED DOCUMENT LAYOUT ---\n{layout}"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"OCR Pipeline Failure: {exc}") from exc

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: print(extract_text_from_image(sys.argv[1]))
    else: print("Usage: python3 ocr_engine.py <image_path>")