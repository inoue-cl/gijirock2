# Speech Transcriber

This project provides a cross-platform CLI/GUI tool for running speech recognition models hosted on Hugging Face.

## Requirements
- Python 3.10+
- ffmpeg (bundled binaries provided)

## Setup

### Using `venv`
```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: install via Conda for `pyannote.audio`
```bash
conda create -n transcriber python=3.10
conda activate transcriber
pip install -r requirements.txt
```

Create a `.env` file with your HuggingFace token:
```
HF_TOKEN=your_token_here
```

## Usage

### CLI
```
python main.py path/to/file.wav --model openai/whisper-small
```

### GUI
```
python gui.py
```

## Building executables
Run the helper script which uses PyInstaller on Windows and directory packaging on macOS:
```
python build.py
```

## GitHub Actions
The workflow `build.yml` builds the project on macOS and Windows and uploads the artifacts (`tar.gz` for macOS and `exe` for Windows).

## Troubleshooting
If `pyannote.audio` fails to install, ensure you are using a compatible Python version and that `torch` is installed. Using a Conda environment is recommended when installation errors occur.
