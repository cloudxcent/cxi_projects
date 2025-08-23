import os
from openai import AzureOpenAI  # ✅ Only works if openai >=1.3.9
from dotenv import load_dotenv  # ✅ New

# ✅ Load environment variables
load_dotenv()

# ✅ Azure OpenAI credentials (from .env)
api_key = os.getenv("AZURE_GPT4_API_KEY")
azure_endpoint = os.getenv("AZURE_GPT4_ENDPOINT")
deployment_name = os.getenv("AZURE_GPT4_DEPLOYMENT_NAME")

# ✅ AzureOpenAI client initialization
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-15-preview",
    azure_endpoint=azure_endpoint,
)

def load_cleaned_text(page_num):
    """Load cleaned OCR text for a specific page."""
    path = f"assets/ocr_output_clean_{page_num}.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ OCR file not found: {path}")
        return ""

def generate_narration(text, language="en"):
    """Generate spoken narration from text without altering content."""
    if not text.strip():
        return "No content to narrate."

    prompt = (
        f"You are a narrator. Read aloud the following content in {language}. "
        f"Do not add or change anything. Just convert the text into a natural spoken narration format in {language}.\n\n{text}"
    )

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ GPT Error: {e}")
        return "Narration could not be generated due to content filter or API error."

def save_narration(narration, page_num):
    """Save the narration output for a specific page."""
    output_path = f"assets/narration_output_{page_num}.txt"
    os.makedirs("assets", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(narration)
    print(f"✅ Narration saved to {output_path}")

def generate_and_save_narration_for_pages(page_count, language="en"):
    """Main loop to generate narration per page."""
    for page_num in range(1, page_count + 1):
        print(f"\n📖 Processing Page {page_num}")
        text = load_cleaned_text(page_num)
        narration = generate_narration(text, language)
        save_narration(narration, page_num)

if __name__ == "__main__":
    total_pages = 3
    selected_language = "mr"  # 'mr' = Marathi, 'hi' = Hindi, 'en' = English
    generate_and_save_narration_for_pages(total_pages, selected_language)
