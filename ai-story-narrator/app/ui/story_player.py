import os
import streamlit as st
from PIL import Image
import time

# === Configuration ===
IMAGE_DIR = "generated_images"
AUDIO_PATH = "output_audio.wav"
NARRATION_PATH = "narration_output.txt"

def split_into_scenes(narration_text, max_len=250):
    """Split narration into scenes (same as image generator)."""
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

def load_scenes():
    """Load narration scenes and corresponding images."""
    if not os.path.exists(NARRATION_PATH):
        st.error("Narration text not found.")
        return [], []

    with open(NARRATION_PATH, "r", encoding="utf-8") as f:
        narration = f.read()

    scenes = split_into_scenes(narration)

    images = []
    for idx in range(1, len(scenes) + 1):
        img_path = os.path.join(IMAGE_DIR, f"scene_{idx}.png")
        if os.path.exists(img_path):
            images.append(img_path)
        else:
            images.append(None)

    return scenes, images

# === Streamlit App ===
st.set_page_config(layout="centered", page_title="AI Story Playback", page_icon="🎥")

st.title("🎬 AI Story Playback")
st.markdown("Watch your AI-narrated story unfold with visuals and audio.")

# Play full narration audio
if os.path.exists(AUDIO_PATH):
    st.audio(AUDIO_PATH)
else:
    st.warning("Narration audio not found.")

scenes, images = load_scenes()

if not scenes:
    st.stop()

# Scene-by-scene playback
st.markdown("## Scenes")
for i, (scene_text, img_path) in enumerate(zip(scenes, images)):
    st.markdown(f"### Scene {i + 1}")
    if img_path:
        st.image(Image.open(img_path), caption=f"Scene {i + 1}", use_column_width=True)
    else:
        st.warning("Image not found.")
    st.markdown(f"*{scene_text}*")
    st.markdown("---")
    time.sleep(0.5)
