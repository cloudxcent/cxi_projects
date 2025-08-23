from app.ocr.ocr_engine import image_to_text
from app.ocr.cleaner import clean_text
from app.narration.narrator import generate_narration, save_narration

def process_image_to_narration(image_path):
    print("🔍 Performing OCR...")
    raw_text = image_to_text(image_path)

    print("🧹 Cleaning text...")
    cleaned = clean_text(raw_text)

    with open("ocr_output_clean.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)

    print("🎙️ Generating narration...")
    narration = generate_narration(cleaned)
    save_narration(narration)

    print("✅ Done.")

if __name__ == "__main__":
    img_path = "assets/uploads/sample_page.jpg"
    process_image_to_narration(img_path)
