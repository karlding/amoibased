# amoibased

Strip silence from close-mic `.m4a` recordings. Originals are left in place; trimmed AAC files are written to `trimmed/`.

With one file, or with more than two, each track is gated on its own. A short analysis pass estimates the noise floor and the close-mic speech level, then keeps regions a little above the floor so quiet syllables survive. Kept regions are joined with a short fade.

With exactly two files (corresponding to two tracks of speakers taking turns), each track stays open on its own speech and stays closed while the other mic is the clear source. A hold that is already open finishes its trail, so the other person starting does not chop the last syllable. Residual bleed inside kept audio is ducked, not subtracted.

## Requirements

- Python 3.13 or newer
- [ffmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on `PATH`)

## Usage

Process every `.m4a` in the current directory. Two files use pair mode; any other count uses the per-track gate.

```bash
python3 remove_silence.py
```

Pass files explicitly:

```bash
python3 remove_silence.py speaker_a.m4a speaker_b.m4a
```

Preview timings without writing audio:

```bash
python3 remove_silence.py --dry-run speaker_a.m4a speaker_b.m4a
```

Gate each file on its own even when two files are given:

```bash
python3 remove_silence.py --no-cancel speaker_a.m4a speaker_b.m4a
```

```bash
python3 remove_silence.py --help 
```

## Tests

Clip audio used by the fixture tests stays on disk and is not committed. Those tests skip when the local files are absent.

After obtaining the source test files and unpacking them into the appropriate directory:

```bash
uv run pytest
```
