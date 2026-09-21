#!/usr/bin/env python3
"""Remove silence from each m4a file, with bleed ducking and a noise gate.

When two tracks are present (speaker + translator), keep intervals open on
each close-mic's own speech, but refuse to open while the other mic is the
clear source (bleed exclusion). Once a hold is open, trail/hang finish even
if the other speaker starts, so words are not chopped. Soft duck attenuates
any residual below-trail bleed inside kept audio. Speech regions are
concatenated with short fades so joins do not click. Originals are left
untouched; stripped files go to trimmed/.
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

# 16 kHz is enough for speech (voice energy is almost all below 8 kHz) and
# matches what whisper.cpp expects after we downsample for analysis.
ANALYSIS_RATE = 16_000
# 20 ms is a standard VAD hop: short enough to catch onsets, long enough
# for a stable RMS estimate (320 samples at 16 kHz).
FRAME_MS = 20
BYTES_PER_S16 = 2
# Peak of signed 16-bit PCM; dBFS = 20 * log10(rms / this).
FULL_SCALE_S16 = 32768.0
# Floor so silent frames are not log(0). 1e-9 is ~-180 dBFS, below any mic.
RMS_FLOOR = 1e-9
# 20 * log10(1e-4) ≈ -80 dBFS: below this we treat a frame as digital silence.
INAUDIBLE_DB = -80.0

# Loudest 10% of audible frames ≈ close-mic speech, not one-off plosives (p99).
SPEECH_PERCENTILE = 90
# Quietest 10% of audible frames ≈ room tone on that mic.
NOISE_PERCENTILE = 10
# Used only if a file has no audible frames.
FALLBACK_NOISE_DB = -60.0
FALLBACK_SPEECH_DB = -25.0
# Single-file gate: sit 10 dB above the noise floor, but at least 12 dB
# below speech so quiet syllables are not cut.
NOISE_GATE_ABOVE_FLOOR_DB = 10.0
NOISE_GATE_BELOW_SPEECH_DB = 12.0

# On-mic if within this many dB of that track's speech peak. 8 dB sits
# above pause/room on the hotter mic (~10–15 dB down) and still includes
# normal speech dynamics.
ON_MIC_BELOW_SPEECH_DB = 8.0
# After energy drops below the trail band, keep this long through true
# quiet so the last consonant is not stripped. Trail already covers the
# fading syllable; 350 ms is about one extra phone.
HANG_MS = 350.0
# Still count as this speaker (without a new confirmation) down to this
# many dB below the speech peak. Fading syllables live here; room during a
# pause is typically ~30–40 dB down on these close-mics.
TRAIL_BELOW_SPEECH_DB = 18.0
# Reach this far back from confirmed speech so the first words are at
# full gain. Longer than CONFIRM_MS; shorter than a pause between turns.
PREROLL_MS = 400.0
# Require this much contiguous speech-like energy before opening. Two
# analysis frames, so a single 20 ms blip cannot start a region.
CONFIRM_MS = 40.0
# One-tap NLMS attributing this-mic energy to the other close-mic. Detection
# only — audio is never subtracted.
# If this mic is at least this many dB more on-mic than the other, treat as
# local even when the other is active (overlap / shared room).
LOCALNESS_DOMINANCE_DB = 4.0
# Keep/duck treat a frame as local speech when at least this fraction of
# this-mic energy is not explained by the other mic. Bleed is predictable
# (score near 0); independent speech is not.
LOCALNESS_MIN = 0.40
# Soft duck envelope: ~40 ms attack, ~120 ms release so bleed fades without
# clicks. Keep boundaries come from the this-mic VAD, not from duck.
DUCK_ATTACK_FRAMES = 2
DUCK_RELEASE_FRAMES = 6
# Dummy dB values so keep_intervals can reuse its energy-gate path:
# 0 is well above THRESHOLD, INAUDIBLE_DB is well below.
KEEP_PROXY_DB = 0.0
DROP_PROXY_THRESHOLD_DB = -40.0

# Default join fade: 60 ms raised-cosine across splices.
FADE_S = 0.06
# Within-sentence pauses are typically < 0.5 s; longer gaps are turn-taking.
MIN_SILENCE_S = 0.5
# Extra keep on either side of a region, on top of preroll/hang. 250 ms is
# about one unstressed syllable and is not a turn-taking wait.
PAD_S = 0.25
# Pair mode: hang + preroll already extend regions. An extra keep_intervals
# pad re-inserts room and merges phrase gaps into dead air.
PAIR_PAD_S = 0.0
# Pair mode: merge only short within-phrase gaps. Larger gaps are turns.
PAIR_MIN_SILENCE_S = 0.25
# Pair mode: drop keep scraps shorter than this. 80 ms ≈ one short phone.
PAIR_MIN_REGION_S = 0.08
# Drop islands shorter than this (50 ms is less than a speech sound).
MIN_REGION_S = 0.05

# 64 Ki frames per PCM read: ~1.5 s at 44.1 kHz, a balance of syscalls vs RAM.
PCM_CHUNK_FRAMES = 65_536
# Fallback AAC bitrate when ffprobe omits bit_rate; clamp keeps 96–256 kb/s.
DEFAULT_AAC_BITRATE = 192_000
MIN_AAC_BITRATE = 96_000
MAX_AAC_BITRATE = 256_000
DEFAULT_OUTPUT_DIR = Path("trimmed")


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
        "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else DEFAULT_AAC_BITRATE,
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


def speech_level_db(dbs: list[float]) -> float:
    """Typical close-mic level: SPEECH_PERCENTILE of audible frames."""
    audible = [db for db in dbs if db >= INAUDIBLE_DB]
    return percentile(sorted(audible or dbs), SPEECH_PERCENTILE)


def ms_to_frames(ms: float) -> int:
    return max(1, int(round(ms / FRAME_MS)))


def smooth_gain(values: list[float], attack_frames: int, release_frames: int) -> list[float]:
    out: list[float] = []
    current = 0.0
    attack = 1.0 if attack_frames <= 1 else 1.0 - math.exp(-1.0 / attack_frames)
    release = 1.0 if release_frames <= 1 else 1.0 - math.exp(-1.0 / release_frames)
    for target in values:
        coeff = attack if target > current else release
        current += coeff * (target - current)
        out.append(current)
    return out


def localness_scores(
    this_dbs: list[float],
    other_dbs: list[float],
    this_speech: float,
    other_speech: float,
    on_mic_db: float = ON_MIC_BELOW_SPEECH_DB,
    trail_db: float = TRAIL_BELOW_SPEECH_DB,
    dominance_db: float = LOCALNESS_DOMINANCE_DB,
) -> list[float]:
    """Per-frame score in [0, 1]: how local this mic's energy is.

    Uses the other close-mic as a reference:
    - Other below trail, or this dominates by dominance_db → local.
    - Other on-mic and louder by dominance_db → bleed (quieter energy on
      this mic is the other speaker's pickup).
    - Otherwise one-tap NLMS predicts this amplitude from the other; high
      residual means local, low residual means bleed.

    Detection only — audio is never subtracted.
    """
    n = min(len(this_dbs), len(other_dbs))
    open_db = -on_mic_db
    trail_open_db = -trail_db
    scores: list[float] = []
    eps = 1e-4
    w = 0.0
    mu = 0.35
    w_max = 1.0
    for i in range(n):
        this_rel = this_dbs[i] - this_speech
        other_rel = other_dbs[i] - other_speech
        if other_rel < trail_open_db or this_rel >= other_rel + dominance_db:
            scores.append(1.0)
            w *= 0.85
            continue
        if other_rel >= open_db and this_rel <= other_rel - dominance_db:
            # Other is clearly the on-mic source; this quieter copy is bleed.
            scores.append(0.0)
            t_a = 10.0 ** (this_rel / 20.0)
            o_a = 10.0 ** (other_rel / 20.0)
            pred = w * o_a
            err = t_a - pred
            w += mu * err * o_a / (o_a * o_a + eps)
            w = max(0.0, min(w, w_max))
            continue
        t_a = 10.0 ** (this_rel / 20.0)
        o_a = 10.0 ** (other_rel / 20.0)
        pred = w * o_a
        err = t_a - pred
        w += mu * err * o_a / (o_a * o_a + eps)
        w = max(0.0, min(w, w_max))
        pred = w * o_a
        explained = min(1.0, (pred * pred) / (t_a * t_a + 1e-8))
        scores.append(1.0 - explained)
    return scores



def this_keep_flags(
    this_dbs: list[float],
    this_speech: float,
    other_dbs: list[float] | None = None,
    other_speech: float | None = None,
    on_mic_db: float = ON_MIC_BELOW_SPEECH_DB,
    hang_ms: float = HANG_MS,
    preroll_ms: float = PREROLL_MS,
    trail_db: float = TRAIL_BELOW_SPEECH_DB,
    localness_min: float = LOCALNESS_MIN,
) -> list[bool]:
    """Decide which frames belong on this track's timeline.

    One rule: keep while this mic has speech-like energy that is local to
    this mic (not explained as bleed from the other close-mic). Speech-like
    includes the trail band. Localness comes from short-window prediction
    of this envelope from the other mic. Hang after the last owned frame
    covers a short handoff; hang only rearms from on-mic/dominant local
    speech or when the other has not been recently active, so trail-band
    bleed flickers and brief other-quiet dips cannot keep re-arming through
    their turn. Ducking is separate and never decides keep.
    """
    n = len(this_dbs)
    if other_dbs is None:
        other_dbs = [INAUDIBLE_DB] * n
        other_speech = this_speech
    else:
        n = min(n, len(other_dbs))
        if other_speech is None:
            other_speech = this_speech
    hang_frames = ms_to_frames(hang_ms)
    preroll_frames = ms_to_frames(preroll_ms)
    confirm = ms_to_frames(CONFIRM_MS)
    open_db = -on_mic_db
    trail_open_db = -trail_db
    local = localness_scores(
        this_dbs, other_dbs, this_speech, other_speech, trail_db=trail_db
    )

    hold = False
    hang_left = 0
    want_run = 0
    other_active_left = 0
    flags: list[bool] = []
    for i in range(n):
        this_rel = this_dbs[i] - this_speech
        other_rel = other_dbs[i] - other_speech
        if other_rel >= trail_open_db:
            other_active_left = hang_frames
        elif other_active_left > 0:
            other_active_left -= 1
        speech_like = this_rel >= trail_open_db
        # While the other was recently active, trail-band energy on this mic
        # cannot claim want (hang may still cover a short handoff). Instant
        # other-quiet dips must not look like this-mic dominance.
        if other_active_left > 0 and this_rel < open_db:
            want = False
        else:
            want = speech_like and local[i] >= localness_min
        can_rearm_hang = other_active_left == 0 or this_rel >= open_db
        if want:
            want_run += 1
        else:
            want_run = 0
        if want_run >= confirm or (hold and want):
            hold = True
            if can_rearm_hang:
                hang_left = hang_frames
        elif hang_left > 0:
            hang_left -= 1
            hold = True
        else:
            hold = False
        flags.append(hold)

    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < n and flags[i]:
            i += 1
        j = start - 1
        steps = 0
        while j >= 0 and steps < preroll_frames:
            this_rel_j = this_dbs[j] - this_speech
            if this_rel_j < trail_open_db:
                break
            if local[j] < localness_min:
                break
            flags[j] = True
            j -= 1
            steps += 1
    return flags


def bleed_duck_gains(
    this_dbs: list[float],
    other_dbs: list[float],
    this_speech: float,
    other_speech: float,
    on_mic_db: float = ON_MIC_BELOW_SPEECH_DB,
    trail_db: float = TRAIL_BELOW_SPEECH_DB,
    localness_min: float = LOCALNESS_MIN,
) -> list[float]:
    """Soft attenuation for bleed; never used to decide keep intervals.

    Full gain while this energy is local. Fade when this mic is quiet under
    on-mic other speech, or when speech-like energy is explained as bleed.
    Local speech including soft trail stays up, so cross-bleed on the other
    mic cannot pump mid-phrase.
    """
    n = min(len(this_dbs), len(other_dbs))
    open_db = -on_mic_db
    trail_open_db = -trail_db
    local = localness_scores(
        this_dbs, other_dbs, this_speech, other_speech, trail_db=trail_db
    )
    targets: list[float] = []
    for i in range(n):
        this_rel = this_dbs[i] - this_speech
        other_rel = other_dbs[i] - other_speech
        if this_rel < trail_open_db:
            target = 0.0 if other_rel >= open_db else 1.0
        elif local[i] >= localness_min:
            target = 1.0
        else:
            target = 0.0
        targets.append(target)
    return smooth_gain(
        targets,
        attack_frames=DUCK_ATTACK_FRAMES,
        release_frames=DUCK_RELEASE_FRAMES,
    )


def apply_frame_gains(
    data: bytes,
    channels: int,
    sample_offset: int,
    sample_rate: int,
    gains: list[float],
) -> bytes:
    n_frames = len(data) // (BYTES_PER_S16 * channels)
    if n_frames == 0 or not gains:
        return data
    hop = max(1, int(round(sample_rate * FRAME_MS / 1000.0)))
    samples = array.array("h")
    samples.frombytes(data)
    last = len(gains) - 1
    for i in range(n_frames):
        gi = min(last, (sample_offset + i) // hop)
        gain = gains[gi]
        if gain == 1.0:
            continue
        for ch in range(channels):
            idx = i * channels + ch
            samples[idx] = int(samples[idx] * gain)
    return samples.tobytes()


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
            frames.append(20.0 * math.log10(max(rms, RMS_FLOOR) / FULL_SCALE_S16))
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


def identify_main_speaker(dbs: list[float]) -> dict[str, float]:
    """Estimate noise floor and close-mic speech from percentiles."""
    audible_levels = sorted(db for db in dbs if db >= INAUDIBLE_DB)
    noise = percentile(audible_levels, NOISE_PERCENTILE) if audible_levels else FALLBACK_NOISE_DB
    speech = percentile(audible_levels, SPEECH_PERCENTILE) if audible_levels else FALLBACK_SPEECH_DB
    return {
        "main_speaker": speech,
        "bleed": float("nan"),
        "noise": noise,
        "threshold": min(
            noise + NOISE_GATE_ABOVE_FLOOR_DB,
            speech - NOISE_GATE_BELOW_SPEECH_DB,
        ),
    }


def keep_intervals(
    dbs: list[float],
    threshold_db: float,
    min_silence: float,
    pad: float,
    min_region: float = MIN_REGION_S,
) -> list[tuple[float, float]]:
    """Return [start, end) times in seconds that should be kept."""
    n = len(dbs)
    if n == 0:
        return []
    frame_s = FRAME_MS / 1000.0
    is_speech = [db >= threshold_db for db in dbs]

    pad_frames = max(1, int(round(pad / frame_s))) if pad > 0 else 0
    filled = is_speech[:]
    # Sentinel: no speech seen yet (smaller than any real frame index).
    last_speech = -1
    for i, speech in enumerate(is_speech):
        if speech:
            last_speech = i
        elif last_speech >= 0 and i - last_speech <= pad_frames:
            filled[i] = True
    next_speech = n
    for i in range(n - 1, -1, -1):
        if is_speech[i]:
            next_speech = i
        elif next_speech < n and next_speech - i <= pad_frames:
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
    return [(max(0.0, s), min(duration, e)) for s, e in merged if e - s > min_region]


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
    gains: list[float] | None = None,
) -> None:
    sr = info["sample_rate"]
    ch = info["channels"]
    bitrate = max(MIN_AAC_BITRATE, min(MAX_AAC_BITRATE, info["bit_rate"]))
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
            n = min(remaining, PCM_CHUNK_FRAMES)
            data = read_exact(decode.stdout, n * bytes_per_frame)
            if not data:
                return
            remaining -= len(data) // bytes_per_frame

    def copy_faded(n_frames: int, abs_pos: int) -> None:
        fade = min(int(fade_s * sr), max(0, n_frames // 2))
        remaining = n_frames
        region_pos = 0
        while remaining > 0:
            n = min(remaining, PCM_CHUNK_FRAMES)
            data = read_exact(decode.stdout, n * bytes_per_frame)
            if not data:
                return
            got = len(data) // bytes_per_frame
            if gains:
                data = apply_frame_gains(data, ch, abs_pos + region_pos, sr, gains)
            data = fade_chunk(data, ch, region_pos, n_frames, fade, fade)
            encode.stdin.write(data)
            remaining -= got
            region_pos += got

    try:
        pos = 0
        for start, end in keep_samples:
            if start > pos:
                skip(start - pos)
                pos = start
            if end > pos:
                copy_faded(end - pos, pos)
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
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for stripped files (default: {DEFAULT_OUTPUT_DIR}/)",
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
        default=MIN_SILENCE_S,
        help="only drop quiet stretches at least this long, in seconds",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=PAD_S,
        help="keep this many seconds of audio around each speech region",
    )
    parser.add_argument(
        "--fade",
        type=float,
        default=FADE_S,
        help="fade in/out at each join, in seconds",
    )
    parser.add_argument(
        "--speech-margin",
        type=float,
        default=ON_MIC_BELOW_SPEECH_DB,
        help=(
            "dB below a track's own speech peak to count as that person on-mic "
            f"(default: {ON_MIC_BELOW_SPEECH_DB:g})"
        ),
    )
    parser.add_argument(
        "--no-cancel",
        action="store_true",
        help="skip cross-track bleed cancellation even when two files are given",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure gates and print timings without writing files",
    )
    return parser.parse_args()


def process_file(path: Path, args: argparse.Namespace, source: Path | None = None) -> None:
    src = source or path
    dest = args.output_dir / path.name
    info = stream_info(src)
    original = info["duration"]
    print(f"{path.name}")
    print(f"  original  : {format_hms(original)}")
    if source is not None:
        print("  cleaned   : other-track bleed cancelled")

    print("  measuring : noise floor and close-mic peak")
    dbs = rms_db_frames(src)
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
    render_stripped(src, dest, info, intervals, args.fade)


def process_pair(files: list[Path], args: argparse.Namespace) -> None:
    """Gate each close-mic on its own energy; soft-duck bleed from the other."""
    print("Analyzing both tracks (this-mic VAD + bleed duck)…")
    dbs_list: list[list[float]] = []
    speech_list: list[float] = []
    infos = [stream_info(path) for path in files]
    for path in files:
        print(f"  scanning  : {path.name}")
        dbs = rms_db_frames(path)
        dbs_list.append(dbs)
        speech = speech_level_db(dbs)
        speech_list.append(speech)
        print(f"    speech  : {speech:.1f} dBFS")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(files):
        dest = args.output_dir / path.name
        original = infos[i]["duration"]
        flags = this_keep_flags(
            dbs_list[i],
            speech_list[i],
            dbs_list[1 - i],
            speech_list[1 - i],
            on_mic_db=args.speech_margin,
        )
        gains = bleed_duck_gains(
            dbs_list[i],
            dbs_list[1 - i],
            speech_list[i],
            speech_list[1 - i],
            on_mic_db=args.speech_margin,
        )
        on_mic = sum(1 for f in flags if f)
        proxy = [KEEP_PROXY_DB if f else INAUDIBLE_DB for f in flags]
        intervals = keep_intervals(
            proxy,
            DROP_PROXY_THRESHOLD_DB,
            PAIR_MIN_SILENCE_S,
            PAIR_PAD_S,
            min_region=PAIR_MIN_REGION_S,
        )
        kept = sum(end - start for start, end in intervals)
        print(f"{path.name}")
        print(f"  original  : {format_hms(original)}")
        print(f"  speech    : {speech_list[i]:.1f} dBFS")
        print(f"  on-mic    : {100.0 * on_mic / max(len(flags), 1):.1f}% of frames")
        print(f"  fade      : {args.fade * 1000:.0f} ms at each join")
        print(f"  regions   : {len(intervals)} speech segment(s)")
        print(f"  kept      : {format_hms(kept)}  ({100.0 * kept / original:.1f}%)")
        print(f"  removed   : {format_hms(original - kept)}")
        if args.dry_run or not intervals:
            if not intervals:
                print("  skipped   : no speech detected")
            continue
        print(f"  writing   : {dest}")
        render_stripped(path, dest, infos[i], intervals, args.fade, gains=gains)


def main() -> None:
    args = parse_args()
    require_ffmpeg()
    files = collect_files(args.files)
    if len(files) == 2 and not args.no_cancel:
        process_pair(files, args)
        return
    if len(files) != 2 and not args.no_cancel:
        print("Bleed ducking needs exactly two tracks; gating only.")
    for path in files:
        process_file(path, args)


if __name__ == "__main__":
    main()
