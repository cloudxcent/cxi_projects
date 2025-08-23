from typing import List
import re
import langdetect

# ========== Sensitive Word Replacements (Softening) ==========
SENSITIVE_WORDS_EN = {
    r"\bbetrayed\b": "replaced",
    r"\bconquered\b": "expanded into",
    r"\bdefeated\b": "overcame challenges from",
    r"\barmy\b": "military forces",
    r"\bseized\b": "assumed control of",
    r"\bkilled\b": "neutralized",
    r"\bwar\b": "military campaign",
    r"\battack(ed)?\b": "advanced toward",
    r"\bbattle\b": "conflict",
    r"\bfort\b": "defensive structure"
}

SENSITIVE_WORDS_HI = {
    r"\bयुद्ध\b": "सैनिक अभियान",
    r"\bहमला\b": "आगे बढ़े",
    r"\bमार दिया\b": "खत्म किया",
    r"\bसेना\b": "सैन्य बल",
}

SENSITIVE_WORDS_MR = {
    r"\bयुद्ध\b": "लष्करी मोहिम",
    r"\bहल्ला\b": "पुढे सरसावले",
    r"\bठार मारले\b": "धोका दूर केला",
    r"\bसेना\b": "लष्करी पथक",
}

# ========== Language Detection ==========
def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except:
        return "en"

# ========== Softening Engine ==========
def soften_content(text: str) -> str:
    lang = detect_language(text)
    if lang == "hi":
        dictionary = SENSITIVE_WORDS_HI
    elif lang == "mr":
        dictionary = SENSITIVE_WORDS_MR
    else:
        dictionary = SENSITIVE_WORDS_EN

    for pattern, replacement in dictionary.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ========== Scene Splitter ==========
def split_into_scenes(text: str, max_len: int = 250) -> List[str]:
    sentences = re.split(r'(?<=[।!?\.])\s+', text.strip())
    scenes = []
    buffer = ""

    for sent in sentences:
        if len(buffer) + len(sent) < max_len:
            buffer += sent.strip() + " "
        else:
            scenes.append(buffer.strip())
            buffer = sent.strip() + " "
    if buffer:
        scenes.append(buffer.strip())

    return scenes or [text.strip()]

# ========== Final Prompt Builder ==========
def build_prompts_from_narration(narration_text: str, max_len: int = 250, soften: bool = False) -> List[str]:
    """
    Build scene-by-scene prompts from narration text,
    keeping story continuity and character memory.
    """
    if soften:
        narration_text = soften_content(narration_text)

    scenes = split_into_scenes(narration_text, max_len)
    prompts = []

    memory_context = ""
    character_memory = set()

    for idx, scene in enumerate(scenes):
        scene_clean = scene.strip()

        # Extract new characters from scene (naive title-case words)
        names = re.findall(r'\b([A-Z][a-z]{2,})\b', scene_clean)
        character_memory.update(names)

        # Memory description
        char_context = f" Characters involved: {', '.join(sorted(character_memory))}." if character_memory else ""

        # Prompt format
        if idx == 0:
            prompt = (
                f"Scene {idx + 1}: {scene_clean}. "
                f"This is the beginning of the story. Set the environment and introduce the characters clearly."
            )
        else:
            prompt = (
                f"Scene {idx + 1}: {scene_clean}. "
                f"Continue from earlier: {memory_context.strip()} {char_context}"
            )

        # Add to output
        prompts.append(prompt.strip())

        # Update memory
        memory_context = scene_clean

    return prompts
