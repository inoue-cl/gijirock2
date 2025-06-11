import argparse
import os
import sys
from dotenv import load_dotenv
from pydub import AudioSegment
from transformers import pipeline

# ─── ffmpeg パス設定（OS別）
def get_ffmpeg_path() -> str:
    base = os.path.join(os.path.dirname(__file__), 'ffmpeg')
    if sys.platform.startswith('win'):
        return os.path.join(base, 'windows', 'ffmpeg.exe')
    elif sys.platform == 'darwin':
        return os.path.join(base, 'mac', 'ffmpeg')
    else:
        return os.path.join(base, 'linux', 'ffmpeg')

# ─── ffmpeg を PATH に追加
FFMPEG_BINARY = get_ffmpeg_path()
if os.path.exists(FFMPEG_BINARY):
    path_dir = os.path.dirname(FFMPEG_BINARY)
    os.environ['PATH'] = os.pathsep.join([path_dir, os.environ.get('PATH', '')])
    if sys.platform.startswith('win'):
        os.environ['FFMPEG_BINARY'] = FFMPEG_BINARY

# ─── .env からトークンを読み込む（オプション扱い）
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # 今後は不要になるがログイン済チェック用に保持

def transcribe(path: str, model: str, use_gpu: bool = False):
    device = 0 if use_gpu else -1

    # 入力音声を一時 WAV に変換
    audio = AudioSegment.from_file(path)
    temp_path = "temp_audio.wav"
    audio.export(temp_path, format="wav")

    # Whisper パイプライン（トークンは既に認証済みなので渡さない）
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device
    )

    result = asr(temp_path)
    os.remove(temp_path)

    return result["text"]

def main(argv=None):
    parser = argparse.ArgumentParser(description='Simple CLI for speech recognition')
    parser.add_argument('file', help='audio file path')
    parser.add_argument('--model', default='openai/whisper-small', help='HuggingFace model name')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration')
    args = parser.parse_args(argv)

    print("🔍 モデル読み込み中（これは初回は時間かかるよ）…")
    text = transcribe(args.file, args.model, args.use_gpu)
    
    print("\n✅ --- 認識結果 --- ✅\n")
    print(text)

if __name__ == '__main__':
    main()
