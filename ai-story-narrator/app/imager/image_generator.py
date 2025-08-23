import os
import sys
import time
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv  # ✅ New

# ✅ Load environment variables
load_dotenv()

# ✅ Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.imager.prompt_builder import build_prompts_from_narration
from app.utils.prompt_rewriter import rewrite_prompt_if_blocked

# ✅ Azure DALL·E 3 credentials (loaded from .env)
api_key = os.getenv("AZURE_DALLE_API_KEY")
azure_endpoint = os.getenv("AZURE_DALLE_ENDPOINT")
image_deployment_name = os.getenv("AZURE_DALLE_DEPLOYMENT_NAME")

# ✅ Azure OpenAI Client Initialization
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-15-preview",
    azure_endpoint=azure_endpoint
)

# ✅ Output folder
output_dir = "assets/generated_images"
os.makedirs(output_dir, exist_ok=True)


def generate_images_from_narration(narration_text: str, character_memory: str = ""):
    """
    Generates images from full narration, ensuring continuity and filtering.
    Automatically rewrites any blocked prompt.
    """
    print("\n🚀 Generating images from narration with continuity...")

    # ✅ Build prompts per scene
    prompts = build_prompts_from_narration(narration_text, soften=True)

    if character_memory:
        prompts = [f"{character_memory}. {p}" for p in prompts]

    print(f"🖼️ Total prompts to generate: {len(prompts)}")
    saved_paths = []

    for idx, prompt in enumerate(prompts):
        print(f"\n🧠 Original Prompt {idx + 1}:\n{prompt}")
        attempt = 0
        max_attempts = 5

        while attempt < max_attempts:
            try:
                response = client.images.generate(
                    model=image_deployment_name,
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                break  # Success
            except Exception as e:
                if "content_filter" in str(e).lower() or "blocked" in str(e).lower():
                    print("⛔ Prompt blocked. Rewriting...")
                    prompt = rewrite_prompt_if_blocked(prompt)
                    print(f"🔁 Rewritten Prompt Attempt {attempt + 1}:\n{prompt}")
                    attempt += 1
                else:
                    print(f"⚠️ Unexpected error: {e}")
                    break
        else:
            print(f"❌ Failed to generate image after {max_attempts} attempts.")
            continue

        # ✅ Save image
        try:
            image_url = response.data[0].url
            print(f"🌐 Image URL: {image_url}")

            image_path_out = os.path.join(output_dir, f"scene_{idx + 1}.png")
            img_data = requests.get(image_url).content
            with open(image_path_out, 'wb') as f:
                f.write(img_data)

            saved_paths.append(image_path_out)
            print(f"✅ Saved: {image_path_out}")
        except Exception as e:
            print(f"❌ Failed to download or save image: {e}")

        time.sleep(1)  # Respect API rate limits

    print("\n🎬 Image generation completed.")
    return saved_paths
