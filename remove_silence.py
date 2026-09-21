#!/usr/bin/env python3
"""Close-mic tools for m4a recordings.

Originals are left untouched.
"""

from __future__ import annotations

import shutil
import sys


def die(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        die("ffmpeg and ffprobe are required (install ffmpeg)")


def main() -> None:
    require_ffmpeg()


if __name__ == "__main__":
    main()
