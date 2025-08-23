import os
import time
import requests
from openai import AzureOpenAI

# ✅ Azure Credentials
api_key = "58rqcUmGKiaEZNKEb1rV7bD0NIKJHlgDCWbzs8pn0vqEX8aMzmP3JQQJ99BGACfhMk5XJ3w3AAAAACOGrCG8"
azure_endpoint = "https://chand-mdmtmfqf-swedencentral.services.ai.azure.com/"
image_deployment_name = "dall-e-3"
gpt_deployment_name = "gpt-4"  # Change if your GPT deployment has a different name

# ✅ Azure Client Initialization
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-03-01-preview",
    azure_endpoint=azure_endpoint
)

def split_into_scenes(narration_text, max_len=250):
    sentences = narration_text.split(".")
    scenes = []
    scene = ""

    for s in sentences:
        if len(scene) + len(s) < max_len:
            scene += s.strip() + ". "
        else:
            scenes.append(scene.strip())
            scene = s.strip() + ". "
    if scene:
        scenes.append(scene.strip())
    return scenes

# ✅ Rewrites blocked prompts using GPT
def rewrite_prompt(scene_text):
    print("🔁 Rewriting prompt to bypass content filters...")
    try:
        revision = client.chat.completions.create(
            model=gpt_deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites descriptions to comply with Azure OpenAI content policy."},
                {"role": "user", "content": f"Rewrite this scene to avoid content filter issues: {scene_text}"}
            ]
        )
        return revision.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Failed to rewrite prompt: {e}")
        return scene_text

def generate_images_from_narration(
    narration_path=r"C:\Users\Sankalp Sancheti\OneDrive\Desktop\ai-story-narrator\narration_output.txt",
    output_dir="generated_images"
):
    print("🚀 Script started...")

    print(f"📁 Checking narration file at: {narration_path}")
    if not os.path.exists(narration_path):
        print(f"❌ narration_output.txt not found at: {narration_path}")
        return

    with open(narration_path, "r", encoding="utf-8") as f:
        narration = f.read()

    if not narration.strip():
        print("❌ narration_output.txt is empty. Cannot generate images.")
        return

    print("📖 Narration loaded:")
    print(narration[:300] + "..." if len(narration) > 300 else narration)
    print("📚 Splitting into scenes...")

    scenes = split_into_scenes(narration)
    print(f"🖼️ Total scenes: {len(scenes)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, scene in enumerate(scenes):
        print(f"\n🧠 Prompt {idx + 1}: {scene}")

        image_url = None

        try:
            response = client.images.generate(
                model=image_deployment_name,
                prompt=scene,
                n=1,
                size="1024x1024"
            )
            image_url = response.data[0].url
            print(f"🌐 Image URL: {image_url}")

        except Exception as e:
            print(f"⚠️ Error generating image {idx + 1} (original): {e}")

            # ✅ If blocked, retry with safer prompt
            if "content_filter" in str(e).lower() or "blocked" in str(e).lower():
                revised_scene = rewrite_prompt(scene)
                print(f"🔁 Retrying with revised prompt: {revised_scene}")

                try:
                    response = client.images.generate(
                        model=image_deployment_name,
                        prompt=revised_scene,
                        n=1,
                        size="1024x1024"
                    )
                    image_url = response.data[0].url
                    print(f"✅ Revised Image URL: {image_url}")
                except Exception as inner_e:
                    print(f"❌ Failed again after rewriting: {inner_e}")
                    continue
            else:
                continue

        # Save the image if URL is available
        if image_url:
            image_path = os.path.join(output_dir, f"scene_{idx + 1}.png")
            img_data = requests.get(image_url).content
            with open(image_path, 'wb') as handler:
                handler.write(img_data)
            print(f"✅ Saved: {image_path}")
            time.sleep(1)

    print("\n🎬 All images generated.")

# ✅ Run script
if __name__ == "__main__":
    generate_images_from_narration()
