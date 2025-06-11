import argparse
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydub import AudioSegment

# Prioritize bundled ffmpeg

def get_ffmpeg_path() -> str:
    base = os.path.join(os.path.dirname(__file__), 'ffmpeg')
    if sys.platform.startswith('win'):
        return os.path.join(base, 'windows', 'ffmpeg.exe')
    elif sys.platform == 'darwin':
        return os.path.join(base, 'mac', 'ffmpeg')
    else:
        return os.path.join(base, 'linux', 'ffmpeg')

FFMPEG_BINARY = get_ffmpeg_path()
if os.path.exists(FFMPEG_BINARY):
    path_dir = os.path.dirname(FFMPEG_BINARY)
    os.environ['PATH'] = os.pathsep.join([path_dir, os.environ.get('PATH', '')])
    if sys.platform.startswith('win'):
        os.environ['FFMPEG_BINARY'] = FFMPEG_BINARY

load_dotenv()

def transcribe(path: str, model: str, use_gpu: bool = False):
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError('HF_TOKEN not provided')
    client = InferenceClient(model, token=token)
    audio = AudioSegment.from_file(path)
    # Convert to wav bytes
    wav_io = audio.export(format='wav')
    wav_bytes = wav_io.read()
    return client.audio_to_text(wav_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Simple CLI for speech recognition')
    parser.add_argument('file', help='audio file path')
    parser.add_argument('--model', default='openai/whisper-small', help='model name')
    parser.add_argument('--use-gpu', action='store_true', help='enable GPU if available')
    args = parser.parse_args(argv)
    text = transcribe(args.file, args.model, args.use_gpu)
    print(text)

if __name__ == '__main__':
    main()
