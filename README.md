# Speech Transcriber

This project provides a cross-platform CLI/GUI tool for running speech recognition models hosted on Hugging Face.

## Requirements
- Python 3.10+
- ffmpeg (bundled binaries provided)

## Setup

The tool runs on Python 3.10 and has been tested on macOS (M1/M2) using Conda.
Follow these three steps to get started:

```bash
conda create -n transcriber python=3.10
conda activate transcriber
pip install -r requirements.txt
```

For convenience an `environment.yml` is provided. You can alternatively run
`conda env create -f environment.yml`.

Create a `.env` file with your HuggingFace token:
```
HF_TOKEN=your_token_here
```

## Usage

### CLI
```
python main.py path/to/file.wav --model openai/whisper-small \
  --threshold 0.5 --min-duration-on 0.5 --min-duration-off 0.5 \
  --highpass 80 --lowpass 8000 --noise-floor -25
```

### GUI
```
python gui.py
```
The GUI supports drag-and-drop of audio files, a progress bar during
transcription and a **Save** button to export the result to a text file.
Speaker diarization results are displayed as "話者A", "話者B" and so on for
clarity when multiple speakers are detected.
Audio is preprocessed with FFmpeg filters (`highpass`, `lowpass`, `loudnorm`,
`afftdn`) before transcription. The GUI lets you tune these filter values or
choose from presets such as **会議室** or **電話** to match your recording
environment. Diarization thresholds can also be adjusted for noisy audio.

## Building executables
Run the helper script which uses PyInstaller on Windows and directory packaging on macOS:
```
python build.py
```

## GitHub Actions
The workflow `build.yml` builds the project on macOS and Windows and uploads the artifacts (`tar.gz` for macOS and `exe` for Windows).

## Troubleshooting
If installation fails with build errors (e.g. `hmmlearn` or `pybind11`), make sure
you are using the provided versions of `torch`, `torchaudio` and
`pyannote.audio` from the `requirements.txt`. Creating the Conda environment via
`environment.yml` will ensure compatible versions are installed.