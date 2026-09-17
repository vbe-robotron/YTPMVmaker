"""YTPMVmaker の非対話CLI。"""
import argparse
import json
from pathlib import Path

from ytpmv_core import make_song


def main():
    parser = argparse.ArgumentParser(description="複数素材からMIDI音MADを生成")
    parser.add_argument("--material", "-m", action="append", help="音声素材（複数指定可）")
    parser.add_argument("--midi", help="MIDIファイル")
    parser.add_argument("--bgm", help="BGMファイル")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--base-note", type=int, default=60)
    parser.add_argument("--project", help="GUIで保存したプロジェクトJSON")
    args = parser.parse_args()

    data = json.loads(Path(args.project).read_text(encoding="utf-8")) if args.project else {}
    materials = data.get("materials", args.material or [])
    if not materials:
        parser.error("--material を1つ以上指定するか、素材入りの --project を指定してください")
    midi = data.get("midi") or args.midi
    bgm = data.get("bgm") or args.bgm
    outdir = Path(data.get("output_dir", args.output_dir))
    outdir.mkdir(parents=True, exist_ok=True)
    base_notes = data.get("material_base_notes", [args.base_note] * len(materials))
    regions = data.get("material_regions", [{"start": 0, "end": 0, "mode": "synth"} for _ in materials])
    assignments = {int(k): int(v) for k, v in data.get("track_assignments", {}).items()}
    volumes = {int(k): float(v) for k, v in data.get("track_volumes", {}).items()}
    bpm = int(data.get("bpm", args.bpm))
    base = int(data.get("base_note", args.base_note))
    wav = outdir / "otomad_result.wav"
    mp3 = outdir / "otomad_result.mp3"

    def progress(value, message):
        print(f"\r{value * 100:5.1f}% {message}", end="", flush=True)

    count, duration = make_song(materials, midi, bgm, bpm, base, assignments,
                                base_notes, volumes, regions, str(wav), str(mp3), progress)
    print(f"\n完了: {count}ノート / {duration / 1000:.1f}秒")
    print(f"WAV: {wav}\nMP3: {mp3}")


if __name__ == "__main__":
    main()
