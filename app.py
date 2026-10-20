import whisper
import speech_recognition as sr
import numpy as np
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    recognizer = sr.Recognizer()
    # Make recognizer less sensitive so it captures more speech
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8

    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("Adjusting for ambient noise... (Please be quiet for a second)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("\nReady! Speak into your microphone.")
            print("Press Ctrl+C at any time to stop.\n")

            while True:
                try:
                    print("Listening...")
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)

                    print("Transcribing...")

                    # Convert raw audio to numpy float32 array (no ffmpeg needed)
                    raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                    audio_float = raw_data.astype(np.float32) / 32768.0

                    # Transcribe directly using numpy array
                    result = model.transcribe(audio_float, fp16=False)
                    text = result["text"].strip()

                    if text:
                        print(f"\n=> {text}\n")
                    else:
                        print("(No speech detected)\n")

                except KeyboardInterrupt:
                    print("\nExiting program...")
                    break
                except Exception as e:
                    print(f"Error: {e}")

    except OSError as e:
        print(f"Microphone error: {e}")
        print("Please check that your microphone is connected and accessible.")
        sys.exit(1)

if __name__ == "__main__":
    main()
