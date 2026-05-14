import os
import glob
from PIL import Image
from src.ocr_engine import extract_text_from_image

def test_inference():
    # Pick a random sample image
    img_files = glob.glob("training_data/*.jpg")
    if not img_files:
        print("No images found in training_data/")
        return
        
    test_img = "training_data/sample_1778666324.jpg"
    print(f"Testing image: {test_img}")
    
    # Run extraction with PREPROCESS=TRUE
    print("\n--- [TEST 1] WITH PREPROCESSING ---")
    image = Image.open(test_img)
    result_pre = extract_text_from_image(image, preprocess=True)
    print(result_pre)
    
    # Run extraction with PREPROCESS=FALSE
    print("\n--- [TEST 2] WITHOUT PREPROCESSING ---")
    image = Image.open(test_img)
    result_raw = extract_text_from_image(image, preprocess=False)
    print(result_raw)

if __name__ == "__main__":
    test_inference()
