from app.ocr.ocr_engine import image_to_text
image_path = r"C:\Users\Savita\Desktop\project2\ai-story-narrator\test_data\test_page.jpg"

lang = "en"  # or 'hi', 'mr' etc.

text = image_to_text(image_path, lang=lang, debug=False)
print("📝 Extracted OCR Text:")
print(text)
