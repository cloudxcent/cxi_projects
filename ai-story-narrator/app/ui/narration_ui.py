import streamlit as st
import sys
import os
from PIL import Image
import base64
import glob

# Append app directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.ocr.ocr_engine import image_to_text
from app.ocr.cleaner import clean_text
from app.narration.narrator import generate_narration
from app.tts.speech_synthesizer import synthesize_speech_ssml
from app.imager.image_generator import generate_images_from_narration
from app.video.video_generator import create_video_with_audio

# Output directories
BASE_DIRS = {
    "ocr": "assets/ocr_output",
    "clean": "assets/ocr_cleaned",
    "narration": "assets/narration",
    "audio": "assets/audio",
    "images": "assets/generated_images",
    "video": "assets/video",
    "uploads": "assets/uploads"
}
for path in BASE_DIRS.values():
    os.makedirs(path, exist_ok=True)

# 🧹 Cleanup old outputs on every run
for key in ["images", "audio", "video", "narration", "ocr", "clean"]:
    folder = BASE_DIRS[key]
    for file in glob.glob(os.path.join(folder, "*.*")):
        try:
            os.remove(file)
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")

# Language options for TTS
LANGUAGE_OPTIONS = {
    "English": {"voice": "en-IN-NeerjaNeural"},
    "Hindi":   {"voice": "hi-IN-SwaraNeural"},
    "Marathi": {"voice": "mr-IN-AarohiNeural"}
}

# Streamlit UI setup
st.set_page_config(page_title="AI Story Narrator", layout="wide")
st.title("📖 AI Story Narrator")

uploaded_images = st.file_uploader("📤 Upload book page images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
language_display = st.selectbox("🌐 Select narration language", list(LANGUAGE_OPTIONS.keys()))
voice = LANGUAGE_OPTIONS[language_display]["voice"]

character_memory = st.text_input(
    "🧠 Describe main character(s) for visual continuity (optional)", 
    placeholder="e.g., A young girl named Meera wearing a red dress"
)

if uploaded_images:
    st.success(f"✅ {len(uploaded_images)} page(s) uploaded successfully!")
    audio_paths = []
    narration_texts = []

    with st.status("⏳ Processing your story...", expanded=True) as status:
        for i, img_file in enumerate(uploaded_images):
            page_num = i + 1
            st.write(f"📄 Page {page_num} processing...")

            # Save uploaded image
            image_path = os.path.join(BASE_DIRS["uploads"], f"page_{page_num}.png")
            img = Image.open(img_file)
            img.save(image_path)

            # OCR
            st.write("🔍 Performing OCR...")
            try:
                raw_text = image_to_text(image_path)  # Auto language detection
            except Exception as e:
                st.error(f"❌ OCR failed on Page {page_num}: {e}")
                continue

            # Save raw OCR
            with open(os.path.join(BASE_DIRS["ocr"], f"page_{page_num}.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)

            # Clean
            st.write("🧹 Cleaning text...")
            cleaned_text = clean_text(raw_text)
            narration_texts.append(cleaned_text)
            with open(os.path.join(BASE_DIRS["clean"], f"page_{page_num}.txt"), "w", encoding="utf-8") as f:
                f.write(cleaned_text)

        # Merge narration
        full_text = "\n".join(narration_texts)
        st.write("🗣️ Generating narration...")
        narration = generate_narration(full_text, language_display)
        narration_path = os.path.join(BASE_DIRS["narration"], "full_story.txt")
        with open(narration_path, "w", encoding="utf-8") as f:
            f.write(narration)

        # TTS
        st.write("🎙️ Synthesizing audio...")
        audio_path = os.path.join(BASE_DIRS["audio"], "story.wav")
        synthesize_speech_ssml(narration, voice, audio_path)

        # Image Generation
        st.write("🎨 Generating story visuals with continuity...")
        image_paths = generate_images_from_narration(narration, character_memory=character_memory)

        # Final video
        st.write("🎞️ Creating final story video...")
        video_output = os.path.join(BASE_DIRS["video"], "final_story_video.mp4")
        try:
            create_video_with_audio(image_paths, [audio_path], [narration_path], video_output)
            status.update(label="✅ Story video ready!", state="complete")
        except Exception as e:
            st.error(f"❌ Failed to create video: {e}")
            status.update(label="⚠️ Video creation failed.", state="error")

    # Show video
    if os.path.exists(video_output):
        st.subheader("🎬 Final Story Video")
        with open(video_output, "rb") as f:
            video_bytes = f.read()
            b64_video = base64.b64encode(video_bytes).decode()
            st.video(f"data:video/mp4;base64,{b64_video}", format="video/mp4")
    else:
        st.error("⚠️ Final video not found.")
