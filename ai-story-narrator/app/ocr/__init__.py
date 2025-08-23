# app/ocr/__init__.py

from .ocr_engine import image_to_text
from .cleaner import clean_text

def extract_clean_text(image_path):
    raw = image_to_text(image_path)
    return clean_text(raw)
