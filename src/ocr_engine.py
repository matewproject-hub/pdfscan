import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from transformers import pipeline

# Load TrOCR model (printed text) once at import time
_trocr = pipeline("image-to-text", model="microsoft/trocr-base-printed")
import re
import os
from PIL import Image

def clean_text(text):
    """
    Cleans extracted text to remove unwanted symbols and normalize whitespace.
    """
    # Remove non-printable characters
    text = "".join(char for char in text if char.isprintable() or char in "\n\r\t")
    
    # Normalize multiple newlines and spaces
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def preprocess_image(image, debug_name=None):
    """
    Advanced OpenCV preprocessing to maximize OCR accuracy.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Rescale (Upscale if too small)
    height, width = gray.shape
    if width < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 3. Increase Contrast (helps with phone photos)
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # 4. Denoising
    denoised = cv2.fastNlMeansDenoising(contrast, h=10)
    
    # 5. Thresholding (Try Otsu first, then adaptive if needed)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Save debug image
    if debug_name:
        debug_dir = "output/debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, f"processed_{debug_name}.png"), thresh)
    
    return Image.fromarray(thresh)

def extract_text_from_image(image, page_num=1):
    """
    Extracts and cleans text from a single PIL Image using TrOCR.
    """
    # Use page number for debug image naming
    debug_name = f"page_{page_num}"
    processed_image = preprocess_image(image, debug_name=debug_name)
    
    # Run TrOCR pipeline (expects a PIL Image)
    result = _trocr(processed_image)
    # The pipeline returns a list of dicts; take the first entry
    text = result[0]["generated_text"] if result else ""
    return clean_text(text)
