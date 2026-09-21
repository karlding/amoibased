#!/usr/bin/env python3
"""Remove silence from each m4a file, with a per-track noise gate.

Each track is processed independently. A short analysis pass finds the
noise floor and close-mic peak, then gates a little above the floor so
quiet speech is kept. Speech regions are concatenated
with short fades so joins do not click. Originals are left untouched;
stripped files go to trimmed/.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ANALYSIS_RATE = 16_000
FRAME_MS = 20
BYTES_PER_S16 = 2
FADE_S = 0.02


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


def read_exact(stream, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def rms_db_frames(path: Path) -> list[float]:
    """Decode to 16 kHz mono and return dBFS for each analysis frame."""
    frame_samples = ANALYSIS_RATE * FRAME_MS // 1000
    frame_bytes = frame_samples * BYTES_PER_S16
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(ANALYSIS_RATE),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    frames: list[float] = []
    try:
        while True:
            data = read_exact(proc.stdout, frame_bytes)
            if not data:
                break
            samples = array.array("h")
            samples.frombytes(data[: len(data) - (len(data) % BYTES_PER_S16)])
            n = len(samples)
            if n == 0:
                continue
            acc = 0
            for sample in samples:
                acc += sample * sample
            rms = math.sqrt(acc / n)
            frames.append(20.0 * math.log10(max(rms, 1e-9) / 32768.0))
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        die(f"ffmpeg decode failed for {path.name}")
    if not frames:
        die(f"no audio decoded from {path.name}")
    return frames


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = min(
        len(sorted_vals) - 1,
        max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))),
    )
    return sorted_vals[idx]


def level_histogram(
    dbs: list[float], lo: float = -80.0, hi: float = 0.0, step: float = 1.0
) -> tuple[list[float], list[float]]:
    """Return (bin centers, smoothed counts) for frames above digital silence."""
    n_bins = max(2, int((hi - lo) / step))
    hist = [0.0] * n_bins
    for db in dbs:
        if db < lo:
            continue
        i = int((db - lo) / step)
        hist[max(0, min(n_bins - 1, i))] += 1
    smoothed = [0.0] * n_bins
    for i in range(n_bins):
        window = hist[max(0, i - 2) : i + 3]
        smoothed[i] = sum(window) / len(window)
    centers = [lo + (i + 0.5) * step for i in range(n_bins)]
    return centers, smoothed


def find_peaks(
    centers: list[float], values: list[float], min_height: float
) -> list[tuple[float, int, float]]:
    peaks: list[tuple[float, int, float]] = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1] and values[i] >= min_height:
            peaks.append((centers[i], i, values[i]))
    return peaks


def identify_main_speaker(dbs: list[float]) -> dict[str, float]:
    """Find the close-mic peak, then gate just above the noise floor.

    Quiet syllables from the main speaker sit well below the close-mic peak,
    so the cut is placed a small margin above room noise rather than in the
    valley under that peak.
    """
    audible_levels = [db for db in dbs if db >= -80.0]
    noise = percentile(sorted(audible_levels), 10) if audible_levels else -60.0

    centers, smoothed = level_histogram(dbs)
    audible = len(audible_levels)
    min_height = max(audible * 0.02, 1.0)
    peaks = find_peaks(centers, smoothed, min_height)
    speech_peaks = [peak for peak in peaks if peak[0] >= -50.0]
    if not speech_peaks:
        speech_peaks = peaks
    if not speech_peaks:
        speech = percentile(sorted(dbs), 95)
        return {
            "main_speaker": speech,
            "bleed": float("nan"),
            "noise": noise,
            "threshold": min(noise + 10.0, speech - 12.0),
        }

    main_db, _, _ = max(speech_peaks, key=lambda peak: peak[0])
    quieter = [peak[0] for peak in speech_peaks if peak[0] < main_db - 4.0]
    bleed = max(quieter) if quieter else float("nan")
    threshold = min(noise + 10.0, main_db - 12.0)
    return {
        "main_speaker": main_db,
        "bleed": bleed,
        "noise": noise,
        "threshold": threshold,
    }


def keep_intervals(
    dbs: list[float],
    threshold_db: float,
    min_silence: float,
    pad: float,
) -> list[tuple[float, float]]:
    """Return [start, end) times in seconds that should be kept."""
    n = len(dbs)
    if n == 0:
        return []
    frame_s = FRAME_MS / 1000.0
    is_speech = [db >= threshold_db for db in dbs]

    pad_frames = max(1, int(round(pad / frame_s)))
    filled = is_speech[:]
    last_speech = -10**9
    for i, speech in enumerate(is_speech):
        if speech:
            last_speech = i
        elif i - last_speech <= pad_frames:
            filled[i] = True
    next_speech = 10**9
    for i in range(n - 1, -1, -1):
        if is_speech[i]:
            next_speech = i
        elif next_speech - i <= pad_frames:
            filled[i] = True

    intervals: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not filled[i]:
            i += 1
            continue
        j = i + 1
        while j < n and filled[j]:
            j += 1
        intervals.append((i * frame_s, j * frame_s))
        i = j

    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_silence:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    duration = n * frame_s
    return [(max(0.0, s), min(duration, e)) for s, e in merged if e - s > 0.05]


def cosine_gain(i: int, n: int) -> float:
    if n <= 1:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * i / n)


def fade_chunk(
    data: bytes,
    channels: int,
    frame_offset: int,
    region_frames: int,
    fade_in: int,
    fade_out: int,
) -> bytes:
    n_frames = len(data) // (BYTES_PER_S16 * channels)
    start = frame_offset
    end = frame_offset + n_frames
    needs_in = start < fade_in
    needs_out = end > region_frames - fade_out
    if not needs_in and not needs_out:
        return data

    samples = array.array("h")
    samples.frombytes(data)
    for i in range(n_frames):
        abs_i = frame_offset + i
        gain = 1.0
        if abs_i < fade_in:
            gain = cosine_gain(abs_i, fade_in)
        tail = region_frames - 1 - abs_i
        if tail < fade_out:
            gain *= cosine_gain(tail, fade_out)
        if gain != 1.0:
            for ch in range(channels):
                idx = i * channels + ch
                samples[idx] = int(samples[idx] * gain)
    return samples.tobytes()


def render_stripped(
    src: Path,
    dst: Path,
    info: dict,
    intervals: list[tuple[float, float]],
    fade_s: float,
) -> None:
    sr = info["sample_rate"]
    ch = info["channels"]
    bitrate = max(96_000, min(256_000, info["bit_rate"]))
    bytes_per_frame = BYTES_PER_S16 * ch
    keep_samples = [(int(start * sr), int(end * sr)) for start, end in intervals]

    decode = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(ch),
            "-ar",
            str(sr),
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    encode = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(ch),
            "-ar",
            str(sr),
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            str(bitrate),
            str(dst),
        ],
        stdin=subprocess.PIPE,
    )
    assert decode.stdout is not None and encode.stdin is not None

    def skip(n_frames: int) -> None:
        remaining = n_frames
        while remaining > 0:
            n = min(remaining, 65_536)
            data = read_exact(decode.stdout, n * bytes_per_frame)
            if not data:
                return
            remaining -= len(data) // bytes_per_frame

    def copy_faded(n_frames: int) -> None:
        fade = min(int(fade_s * sr), max(0, n_frames // 2))
        remaining = n_frames
        pos = 0
        while remaining > 0:
            n = min(remaining, 65_536)
            data = read_exact(decode.stdout, n * bytes_per_frame)
            if not data:
                return
            got = len(data) // bytes_per_frame
            data = fade_chunk(data, ch, pos, n_frames, fade, fade)
            encode.stdin.write(data)
            remaining -= got
            pos += got

    try:
        pos = 0
        for start, end in keep_samples:
            if start > pos:
                skip(start - pos)
                pos = start
            if end > pos:
                copy_faded(end - pos)
                pos = end
    finally:
        decode.stdout.close()
        encode.stdin.close()
        decode.wait()
        encode.wait()

    if encode.returncode not in (0, None):
        die(f"ffmpeg encode failed for {dst.name}")


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


def parse_threshold(value: str) -> float | None:
    if value.lower() == "auto":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threshold must be 'auto' or a dBFS number like -35"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="m4a files to process (default: all .m4a in the current directory)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("trimmed"),
        help="directory for stripped files (default: trimmed/)",
    )
    parser.add_argument(
        "--threshold",
        type=parse_threshold,
        default=None,
        help="silence threshold in dBFS, or 'auto' to measure each track (default: auto)",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.5,
        help="only drop quiet stretches at least this long, in seconds",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.2,
        help="keep this many seconds of audio around each speech region",
    )
    parser.add_argument(
        "--fade",
        type=float,
        default=FADE_S,
        help="fade in/out at each join, in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure gates and print timings without writing files",
    )
    return parser.parse_args()


def process_file(path: Path, args: argparse.Namespace) -> None:
    dest = args.output_dir / path.name
    info = stream_info(path)
    original = info["duration"]
    print(f"{path.name}")
    print(f"  original  : {format_hms(original)}")

    print("  measuring : noise floor and close-mic peak")
    dbs = rms_db_frames(path)
    if args.threshold is None:
        stats = identify_main_speaker(dbs)
        threshold = stats["threshold"]
        print(f"  noise     : {stats['noise']:.1f} dBFS")
        print(f"  main spk  : {stats['main_speaker']:.1f} dBFS")
        if stats["bleed"] == stats["bleed"]:
            print(f"  bleed     : {stats['bleed']:.1f} dBFS")
        else:
            print("  bleed     : none detected")
    else:
        threshold = args.threshold

    intervals = keep_intervals(dbs, threshold, args.min_silence, args.pad)
    kept = sum(end - start for start, end in intervals)
    print(f"  threshold : {threshold:.1f} dBFS")
    print(f"  fade      : {args.fade * 1000:.0f} ms at each join")
    print(f"  regions   : {len(intervals)} speech segment(s)")
    print(f"  kept      : {format_hms(kept)}  ({100.0 * kept / original:.1f}%)")
    print(f"  removed   : {format_hms(original - kept)}")
    if args.dry_run or not intervals:
        if not intervals:
            print("  skipped   : no speech detected")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  writing   : {dest}")
    render_stripped(path, dest, info, intervals, args.fade)


def main() -> None:
    args = parse_args()
    require_ffmpeg()
    for path in collect_files(args.files):
        process_file(path, args)


if __name__ == "__main__":
    main()
