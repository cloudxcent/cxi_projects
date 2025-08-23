import os
from openai import AzureOpenAI
from dotenv import load_dotenv  # ✅ New

# ✅ Load environment variables
load_dotenv()

# ✅ Azure OpenAI GPT config (from .env)
api_key = os.getenv("AZURE_GPT4_API_KEY")
azure_endpoint = os.getenv("AZURE_GPT4_ENDPOINT")
gpt_deployment = os.getenv("AZURE_GPT4_DEPLOYMENT_NAME")

# ✅ GPT client
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-03-01-preview",
    azure_endpoint=azure_endpoint
)


def rewrite_prompt_if_blocked(prompt: str) -> str:
    """
    Rewrite prompt using GPT to bypass filters without losing visual intent.
    Applies to all prompts dynamically.
    """
    print("🛡️ Rewriting blocked prompt using GPT...")

    system_msg = (
        "You are a helpful AI assistant that rewrites image generation prompts which may contain blocked or filtered words. "
        "Your goal is to preserve the original meaning and visual imagery, but remove or soften any terms that may trigger safety filters. "
        "Avoid references to violence, war, blood, weapons, abuse, or historical conflict. Use cinematic, peaceful, or symbolic alternatives. "
        "Rewritten prompts should be descriptive, imaginative, and safe for all audiences."
    )

    user_msg = f"Rewrite this prompt to preserve its meaning but make it safe for image generation systems:\n\n{prompt}"

    try:
        response = client.chat.completions.create(
            model=gpt_deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"✅ Rewritten Prompt: {rewritten}")
        return rewritten
    except Exception as e:
        print(f"⚠️ GPT error during rewrite: {e}")
        return prompt  # fallback to original prompt
