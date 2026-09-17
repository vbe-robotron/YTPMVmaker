---
name: ytpmv-maker
description: Use the YTPMVmaker CLI to generate multi-material MIDI-based 音MAD audio from素材, MIDI, BGM, and saved project settings.
---

# YTPMVmaker CLI

Use the shared `ytpmv_core` API through `cli.py` for repeatable audio generation. GUI and CLI project settings use the same JSON schema. Prefer a saved GUI project when the user has already configured track assignments, per-track volume, per-material base notes, regions, and playback modes.

```bash
uv run python cli.py -m vocal.wav -m effect.wav --midi melody.mid --bgm bgm.wav --output-dir out
uv run python cli.py --project otomad_project.json
```

The CLI writes `otomad_result.wav` and `otomad_result.mp3`, and prints progress. Verify output duration when the MIDI contains long trailing events.

## Installation

Copy this entire `skills/ytpmv-maker` directory into the agent's skills directory. The directory must retain `SKILL.md` and `agents/openai.yaml`. The skill expects the repository's `cli.py` and `ytpmv_core.py` to be available in the working directory.
