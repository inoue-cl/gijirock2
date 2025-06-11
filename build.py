import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / 'dist'

SPEC = ROOT / 'app.spec'


def clean():
    if DIST.exists():
        shutil.rmtree(DIST)
    build = ROOT / 'build'
    if build.exists():
        shutil.rmtree(build)


def build_windows():
    subprocess.check_call([
        sys.executable, '-m', 'PyInstaller', str(SPEC)
    ])


def build_macos():
    target = DIST / 'app'
    target.mkdir(parents=True, exist_ok=True)
    for f in ['main.py', 'gui.py', 'requirements.txt']:
        shutil.copy(ROOT / f, target)
    shutil.make_archive(str(DIST / 'transcriber-macos'), 'gztar', root_dir=target)


def main():
    clean()
    if sys.platform.startswith('win'):
        build_windows()
    else:
        build_macos()

if __name__ == '__main__':
    main()
