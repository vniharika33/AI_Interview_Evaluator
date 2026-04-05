import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path

# Load Whisper model
model = whisper.load_model("small")

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "audio/answers"


def record_audio(filename, duration=15, fs=44100):
    """
    Records audio from microphone and saves as WAV file
    """
    print("Recording... Speak now")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("Recording saved:", filename)


def transcribe_audio(audio_filename):
    """
    Converts audio file to text using Whisper
    """
    audio_path = AUDIO_DIR / audio_filename

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = model.transcribe(str(audio_path))
    return result["text"]


if __name__ == "__main__":
    # STEP 1: Record audio
    audio_file = AUDIO_DIR / "sample_answer.wav"
    record_audio(str(audio_file))

    # STEP 2: Transcribe audio
    print("\nTranscribing audio...")
    text = transcribe_audio("sample_answer.wav")

    print("\nTranscribed Text:")
    print(text)
