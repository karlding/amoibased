#!/usr/bin/env python3
"""Print the duration of each m4a recording.

Pass files on the command line, or run it in a directory of .m4a files.
Originals are left untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def die(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        die("ffmpeg and ffprobe are required (install ffmpeg)")


def stream_info(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,bit_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        die(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    duration = fmt.get("duration")
    if duration is None:
        die(f"could not read duration for {path.name}")
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration": float(duration),
        "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else 192_000,
    }


def format_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def collect_files(paths: list[Path]) -> list[Path]:
    if paths:
        files = []
        for path in paths:
            if not path.is_file():
                die(f"not a file: {path}")
            files.append(path)
        return files
    files = sorted(Path(".").glob("*.m4a"))
    if not files:
        die("no .m4a files found")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="m4a files to inspect (default: all .m4a in the current directory)",
    )
    return parser.parse_args()


def process_file(path: Path) -> None:
    info = stream_info(path)
    print(f"{path.name}")
    print(f"  original  : {format_hms(info['duration'])}")


def main() -> None:
    args = parse_args()
    require_ffmpeg()
    for path in collect_files(args.files):
        process_file(path)


if __name__ == "__main__":
    main()
