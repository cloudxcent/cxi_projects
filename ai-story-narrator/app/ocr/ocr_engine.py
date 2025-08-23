import os
import time
import requests
import cv2
import numpy as np
import re
from dotenv import load_dotenv  # ✅ New

# ✅ Load environment variables
load_dotenv()

# ✅ Azure Read API credentials (from .env)
AZURE_OCR_KEY = os.getenv("AZURE_OCR_KEY")
AZURE_OCR_ENDPOINT = os.getenv("AZURE_OCR_ENDPOINT")
READ_API_URL = AZURE_OCR_ENDPOINT + "vision/v3.2/read/analyze"

# Image size constraint
MAX_DIMENSION = 4200


def preprocess_image(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found at path: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.medianBlur(resized, 3)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, sharpen_kernel)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, morph_kernel)

    return cleaned


def resize_if_needed(image_np):
    h, w = image_np.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        resized = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"🔧 Resized image to: {resized.shape[1]}x{resized.shape[0]}")
        return resized
    return image_np


def image_to_text(image_path, debug=False):
    processed = preprocess_image(image_path, debug=debug)
    processed = resize_if_needed(processed)

    _, encoded_img = cv2.imencode('.jpg', processed)
    img_bytes = encoded_img.tobytes()

    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_OCR_KEY,
        'Content-Type': 'application/octet-stream'
    }

    # Send image to Read API
    response = requests.post(READ_API_URL, headers=headers, data=img_bytes)
    if response.status_code != 202:
        raise RuntimeError(f"❌ Read API failed: {response.status_code} - {response.text}")

    operation_url = response.headers.get("Operation-Location")
    if not operation_url:
        raise RuntimeError("❌ Missing Operation-Location header from Azure response.")

    # Poll for result
    for _ in range(20):
        result_resp = requests.get(operation_url, headers=headers)
        result_json = result_resp.json()
        status = result_json.get("status", "")
        if status == "succeeded":
            break
        elif status == "failed":
            raise RuntimeError("❌ Azure Read API operation failed.")
        time.sleep(1)

    # Extract text
    lines = []
    for read_result in result_json.get("analyzeResult", {}).get("readResults", []):
        for line in read_result.get("lines", []):
            lines.append(line["text"])

    final_text = " ".join(lines)
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    if not final_text:
        print("⚠️ No text extracted. Using fallback placeholder.")
        return "This is a placeholder narration because OCR failed to extract content."

    return final_text
