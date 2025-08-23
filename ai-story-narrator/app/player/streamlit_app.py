# app/player/streamlit_app.py

import streamlit as st
from PIL import Image
import os

# Paths
IMAGE_DIR = "assets/images"
AUDIO_DIR = "assets/audio"
STORY_FILE = "assets/story.json"

# Load story from JSON (simulated for now)
import json
def load_story():
    with open(STORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenes"]

def show_story():
    scenes = load_story()
    total = len(scenes)

    st.title("📖 AI Story Playback")
    st.markdown("Enjoy your narrated story with visuals!")

    # Scene Navigation
    if "scene" not in st.session_state:
        st.session_state.scene = 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⏮️ Previous") and st.session_state.scene > 0:
            st.session_state.scene -= 1
    with col3:
        if st.button("Next ⏭️") and st.session_state.scene < total - 1:
            st.session_state.scene += 1

    scene = scenes[st.session_state.scene]
    scene_number = st.session_state.scene + 1

    st.subheader(f"Scene {scene_number}")
    st.text(scene["text"])

    # Show image
    image_path = os.path.join(IMAGE_DIR, f"scene_{scene_number}.png")
    if os.path.exists(image_path):
        st.image(Image.open(image_path), use_column_width=True)
    else:
        st.warning("Image not found.")

    # Play audio
    audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_number}.mp3")
    if os.path.exists(audio_path):
        st.audio(audio_path, format="audio/mp3")
    else:
        st.warning("Audio not found.")

if __name__ == "__main__":
    show_story()
