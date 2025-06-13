import argparse
import os
import sys
from dotenv import load_dotenv
from pydub import AudioSegment
from transformers import pipeline
from typing import List, Dict, Optional
import time

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
HF_TOKEN = os.getenv("HF_TOKEN")  # 使用可能なら話者分離に利用


def _load_diarization_pipeline(threshold: float, min_cluster_size: int,
                               min_on: float, min_off: float) -> Optional["Pipeline"]:
    """Load pyannote's speaker diarization pipeline with tuned parameters."""
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except Exception:
        return None

    token = os.getenv("HF_TOKEN")
    if not token:
        return None

    try:
        pipe = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        pipe.instantiate({
            "clustering": {
                "threshold": threshold,
                "min_cluster_size": min_cluster_size,
            },
            "segmentation": {
                "min_duration_on": min_on,
                "min_duration_off": min_off,
            },
        })
        return pipe
    except Exception:
        return None


def _preprocess_audio(path: str, highpass: int, lowpass: int, nf: int) -> AudioSegment:
    """Run FFmpeg-based filtering and return processed audio."""
    filters = f"highpass=f={highpass},lowpass=f={lowpass},loudnorm,afftdn=nf={nf}"
    tmp = path + ".filtered.wav"
    AudioSegment.from_file(path).export(
        tmp, format="wav", parameters=["-af", filters]
    )
    processed = AudioSegment.from_file(tmp)
    os.remove(tmp)
    return processed


def _merge_segments(segments: List[Dict], short_dur: float = 3.0, gap: float = 0.5) -> List[Dict]:
    """Merge short consecutive segments from the same speaker."""
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        same = seg["speaker"] == last["speaker"]
        short = (seg["end"] - seg["start"] <= short_dur) or (
            last["end"] - last["start"] <= short_dur
        )
        if same and (seg["start"] - last["end"] <= gap or short):
            last["end"] = seg["end"]
            last["text"] += " " + seg["text"]
        else:
            merged.append(seg.copy())
    return merged


def transcribe(path: str, model: str, use_gpu: bool = False,
               threshold: float = 0.5, min_cluster_size: int = 15,
               min_on: float = 0.5, min_off: float = 0.5,
               hp: int = 80, lp: int = 8000, nf: int = -25) -> List[Dict]:
    """Transcribe the given audio file and return list of segments."""
    device = 0 if use_gpu else -1

    audio = _preprocess_audio(path, hp, lp, nf)
    temp_path = "temp_audio.wav"
    audio.export(temp_path, format="wav")

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        return_timestamps=True,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )

    diarizer = _load_diarization_pipeline(threshold, min_cluster_size, min_on, min_off)

    diarization_result = None
    if diarizer is not None:
        try:
            diarization_result = diarizer(temp_path)
        except Exception:
            diarization_result = None

    result = asr(temp_path)
    os.remove(temp_path)

    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        speaker = None
        if diarization_result is not None:
            for turn, _, spk in diarization_result.itertracks(yield_label=True):
                if start >= turn.start and end <= turn.end:
                    speaker = spk
                    break
        segments.append({
            "start": float(start),
            "end": float(end),
            "speaker": speaker,
            "text": text,
        })

    segments = _merge_segments(segments)
    return segments


def format_segments(segments: List[Dict]) -> str:
    """Return formatted transcription string."""
    lines = []
    speaker_map = {}
    next_idx = 0
    for seg in segments:
        start = time.strftime("%H:%M:%S", time.gmtime(seg["start"]))
        end = time.strftime("%H:%M:%S", time.gmtime(seg["end"]))
        spk = seg.get("speaker")
        if spk not in speaker_map:
            label = chr(ord('A') + next_idx)
            speaker_map[spk] = f"話者{label}"
            next_idx += 1
        speaker = speaker_map[spk]
        lines.append(f"[{start} - {end}] {speaker}: {seg['text']}")
    return "\n".join(lines)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Simple CLI for speech recognition')
    parser.add_argument('file', help='audio file path')
    parser.add_argument('--model', default='openai/whisper-small', help='HuggingFace model name')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--threshold', type=float, default=0.5, help='Diarization clustering threshold')
    parser.add_argument('--cluster-size', type=int, default=15, help='Minimum cluster size')
    parser.add_argument('--min-duration-on', type=float, default=0.5, help='Minimum speech segment length')
    parser.add_argument('--min-duration-off', type=float, default=0.5, help='Minimum silence length')
    parser.add_argument('--highpass', type=int, default=80, help='High-pass filter frequency')
    parser.add_argument('--lowpass', type=int, default=8000, help='Low-pass filter frequency')
    parser.add_argument('--noise-floor', type=int, default=-25, help='Noise reduction level')
    args = parser.parse_args(argv)

    print("🔍 モデル読み込み中（これは初回は時間かかるよ）…")
    segments = transcribe(
        args.file,
        args.model,
        args.use_gpu,
        threshold=args.threshold,
        min_cluster_size=args.cluster_size,
        min_on=args.min_duration_on,
        min_off=args.min_duration_off,
        hp=args.highpass,
        lp=args.lowpass,
        nf=args.noise_floor,
    )

    print("\n✅ --- 認識結果 --- ✅\n")
    print(format_segments(segments))

if __name__ == '__main__':
    main()
