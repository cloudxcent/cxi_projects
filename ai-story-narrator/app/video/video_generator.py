import os
import glob
from moviepy.editor import ImageSequenceClip, AudioFileClip


def create_video_with_audio(image_files, audio_files, narration_files, output_path):
    if not image_files:
        raise ValueError("❌ Missing image files for video generation.")
    if not audio_files:
        raise ValueError("❌ Missing audio files for video generation.")
    if not narration_files:
        raise ValueError("❌ Missing narration files for video generation.")

    print("🎞️ Creating video with", len(image_files), "images and", len(audio_files), "audio files...")

    # Total duration from all audio files
    total_duration = 0
    audio_clips = []
    for path in audio_files:
        clip = AudioFileClip(path)
        total_duration += clip.duration
        audio_clips.append(clip)

    # Calculate per-image duration
    duration_per_image = total_duration / len(image_files)

    # Prepare final audio (combine multiple audio clips)
    final_audio = audio_clips[0]
    for clip in audio_clips[1:]:
        final_audio = final_audio.append(clip)

    # Sort image files for order
    image_files_sorted = sorted(image_files)

    # Create video clip with synchronized durations
    clip = ImageSequenceClip(image_files_sorted, durations=[duration_per_image] * len(image_files_sorted)).set_audio(final_audio)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export final video
    clip.write_videofile(output_path, fps=24)
    print(f"✅ Final video saved at: {output_path}")


# ✅ For manual test
if __name__ == "__main__":
    image_files = sorted(glob.glob("assets/generated_images/page_1_scene*.png"))
    audio_files = [os.path.join("assets", "audio", "page_1.wav")]
    narration_files = [os.path.join("assets", "narration", "page_1.txt")]
    output_path = os.path.join("assets", "video", "video_output_1.mp4")

    create_video_with_audio(image_files, audio_files, narration_files, output_path)
