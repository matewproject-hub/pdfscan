import os
from pdf2image import convert_from_path, convert_from_bytes

def pdf_to_images(pdf_path):
    """
    Converts a PDF file into a list of PIL Image objects with 300 DPI.
    """
    try:
        print(f"Converting {pdf_path} to images...")
        images = convert_from_path(pdf_path, dpi=300)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def pdf_to_images_from_bytes(pdf_bytes):
    """
    Converts PDF bytes into a list of PIL Image objects with 300 DPI.
    """
    try:
        print("Converting PDF bytes to images...")
        images = convert_from_bytes(pdf_bytes, dpi=300)
        return images
    except Exception as e:
        print(f"Error converting PDF bytes to images: {e}")
        return []
