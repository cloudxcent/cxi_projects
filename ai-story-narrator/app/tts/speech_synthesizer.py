import azure.cognitiveservices.speech as speechsdk
import re
import os
from dotenv import load_dotenv  # ✅ New

# ✅ Load environment variables
load_dotenv()

# Recommended voice mapping (for use in UI or elsewhere)
voice_map = {
    "English (Female)": "en-US-JennyNeural",
    "English (Male)": "en-US-GuyNeural",
    "Hindi (Female)": "hi-IN-SwaraNeural",
    "Hindi (Male)": "hi-IN-MadhurNeural",
    "Marathi (Female)": "mr-IN-AarohiNeural",
    "Marathi (Male)": "mr-IN-ManoharNeural"
}

def synthesize_speech_ssml(
    text,
    voice="en-US-JennyNeural",
    output_audio="narration_audio.wav",
    style="narration-professional"
):
    """
    Synthesizes speech from the given text using Azure TTS with SSML formatting,
    expressive voice style, and sentence-level pacing.
    """
    # ✅ Azure credentials (from .env)
    speech_key = os.getenv("AZURE_TTS_API_KEY")
    service_region = os.getenv("AZURE_TTS_REGION")

    # Extract language tag (e.g., en-US, hi-IN, mr-IN)
    lang_tag = "-".join(voice.split("-")[:2])

    # Split text into sentences for pacing
    sentences = re.split(r'(?<=[।.?!])\s+', text.strip())
    ssml_body = "\n".join(f"<s>{sentence.strip()}</s>" for sentence in sentences if sentence.strip())

    # Build SSML with expressive style
    ssml = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
           xmlns:mstts='http://www.w3.org/2001/mstts'
           xml:lang='{lang_tag}'>
      <voice name='{voice}'>
        <mstts:express-as style='{style}'>
          {ssml_body}
        </mstts:express-as>
      </voice>
    </speak>
    """

    # Configure Azure Speech SDK
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_audio)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Perform synthesis
    result = synthesizer.speak_ssml_async(ssml).get()

    # Result status
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ Audio saved successfully:", output_audio)
    else:
        print("❌ Error in speech synthesis:", result.reason)
