"""YTPMVmaker の非対話CLI。"""
import argparse
import json
from pathlib import Path

import mido

from ytpmv_core import (
    analyze_materials,
    auto_assign_materials,
    auto_track_volumes,
    build_video_timeline,
    is_video_source,
    make_song,
    midi_to_notes,
    render_video,
)


def main():
    parser = argparse.ArgumentParser(description="複数素材からMIDI音MADを生成")
    parser.add_argument("--material", "-m", action="append", help="音声素材（複数指定可）")
    parser.add_argument("--midi", help="MIDIファイル")
    parser.add_argument("--bgm", help="BGMファイル")
    parser.add_argument("--bgm-volume", type=float, default=None,
                        help="BGM音量(dB)。BGMの基準音量-24dBFSに対する相対値。既定は0")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--base-note", type=int, default=None,
                        help="素材の基準音（MIDI番号）。未指定なら自動判定します")
    parser.add_argument("--project", help="GUIで保存したプロジェクトJSON")
    parser.add_argument("--solo-track", type=int, help="指定したMIDIトラック単体だけを書き出す（0始まり）")
    parser.add_argument("--video", action="append", dest="videos",
                        help="映像素材（画像/動画）。複数指定可。音声素材と同じ順に割り当てられます")
    parser.add_argument("--render-video", action="store_true", help="音声に合わせてMP4を書き出す")
    parser.add_argument("--video-size", default=None, help="動画サイズ（例: 1280x720）")
    parser.add_argument("--fps", type=int, default=None, help="動画のフレームレート（既定30）")
    parser.add_argument("--gap-mode", choices=["hold", "black"], default="hold",
                        help="ノート間の無音区間の埋め方（hold: 直前の映像を継続 / black: 黒画面）")
    args = parser.parse_args()

    data = json.loads(Path(args.project).read_text(encoding="utf-8")) if args.project else {}
    materials = data.get("materials", args.material or [])
    if not materials:
        parser.error("--material を1つ以上指定するか、素材入りの --project を指定してください")
    midi = data.get("midi") or args.midi
    bgm = data.get("bgm") or args.bgm
    outdir = Path(data.get("output_dir", args.output_dir))
    outdir.mkdir(parents=True, exist_ok=True)
    regions = data.get("material_regions", [{"start": 0, "end": 0, "mode": "auto"} for _ in materials])
    assignments = {int(k): int(v) for k, v in data.get("track_assignments", {}).items()}
    volumes = {int(k): float(v) for k, v in data.get("track_volumes", {}).items()}
    bpm = int(data.get("bpm", args.bpm))
    base = int(data.get("base_note") or args.base_note or 60)
    bgm_volume = data.get("bgm_volume")
    if bgm_volume is None or bgm_volume == "":
        bgm_volume = args.bgm_volume if args.bgm_volume is not None else 0.0
    try:
        bgm_volume = float(bgm_volume)
    except (TypeError, ValueError):
        print(f"BGM音量の値が不正なため0dBで続行します: {bgm_volume!r}")
        bgm_volume = 0.0

    videos = data.get("videos") or args.videos or []
    video_assignments = {int(k): int(v) for k, v in data.get("video_assignments", {}).items()}
    video_size = data.get("video_size") or args.video_size or "1280x720"
    try:
        width, height = [int(v) for v in str(video_size).lower().replace("×", "x").replace("*", "x").split("x")]
    except ValueError:
        print(f"動画サイズの値が不正なため1280x720で続行します: {video_size!r}")
        width, height = 1280, 720
    fps = int(data.get("video_fps") or args.fps or 30)

    base_notes = data.get("material_base_notes")
    if base_notes is None:
        base_notes = [args.base_note] * len(materials)
    base_notes = [int(v) if v is not None else None for v in base_notes]
    base_notes += [None] * (len(materials) - len(base_notes))

    # 基準音が未指定の素材、または再生モードが「自動判定」の素材は解析する。
    # 自動割り当ても「明るさ」を使うため、必要なときだけ解析する。
    # 解析は重いので、対象を集めてからまとめて並列に実行する。
    profiles = [None] * len(materials)
    modes = [((regions[i].get("mode") if i < len(regions) else None) or "auto") for i in range(len(materials))]
    for i, path in enumerate(materials):
        if base_notes[i] is not None and assignments and modes[i] != "auto":
            # 解析不要な素材はパスと基準音だけ持たせる。
            profiles[i] = {"path": path, "base_note": base_notes[i], "brightness": 440.0}

    targets = [i for i, profile in enumerate(profiles) if profile is None]
    for i, profile in zip(targets, analyze_materials([materials[i] for i in targets])):
        path = materials[i]
        if isinstance(profile, Exception):
            print(f"素材解析エラー: {Path(path).name}: {type(profile).__name__}: {profile}")
            profile = {"base_note": base, "brightness": 440.0}
        profile["path"] = path
        if base_notes[i] is None:
            base_notes[i] = int(profile["base_note"])
            print(f"基準音を自動判定: {Path(path).name} → MIDI {base_notes[i]}")
        else:
            profile["base_note"] = base_notes[i]
        if modes[i] == "auto":
            label = {"synth": "シンセ化", "raw": "そのまま"}.get(profile.get("recommended_mode"), "シンセ化")
            print(f"再生モードを自動判定: {Path(path).name} → {label}"
                  f"（{profile.get('recommended_reason', '')}）")
        profiles[i] = profile

    wav = outdir / "otomad_result.wav"
    mp3 = outdir / "otomad_result.mp3"
    if args.solo_track is not None:
        wav = outdir / f"track_{args.solo_track + 1}.wav"
        mp3 = None

    # 割り当てが未指定なら、素材の基準音・明るさから自動で決めて音量バランスも整える。
    if midi and not assignments:
        try:
            notes, midi_bpm, tpq = midi_to_notes(midi)
            track_names = {i: (track.name or "") for i, track in enumerate(mido.MidiFile(midi).tracks)}
            assignments = auto_assign_materials(notes, profiles, track_names)
            if not volumes:
                volumes = auto_track_volumes(notes, bpm if bpm > 0 else midi_bpm, tpq)
            print("自動割り当て:")
            for track in sorted(assignments):
                name = track_names.get(track) or f"トラック {track + 1}"
                index = assignments[track]
                print(f"  {name} → {Path(materials[index]).name}"
                      f" (基準音 {profiles[index].get('base_note', base)} / 音量 {volumes.get(track, 0):+.1f}dB)")
        except Exception as ex:
            parser.exit(1, f"\n自動割り当てエラー: {type(ex).__name__}: {ex}\n")

    if bgm:
        print(f"BGM音量: {bgm_volume:+.1f}dB")

    def progress(value, message):
        print(f"\r{value * 100:5.1f}% {message}", end="", flush=True)

    try:
        count, duration = make_song(materials, midi, bgm, bpm, base, assignments,
                                    base_notes, volumes, regions, str(wav),
                                    str(mp3) if mp3 else None, progress,
                                    solo_track=args.solo_track,
                                    bgm_gain_db=bgm_volume,
                                    material_profiles=profiles)
    except Exception as ex:
        parser.exit(1, f"\nエラー: {type(ex).__name__}: {ex}\n")
    print(f"\n完了: {count}ノート / {duration / 1000:.1f}秒")
    print(f"WAV: {wav}")
    if mp3:
        print(f"MP3: {mp3}")

    if not args.render_video:
        return
    if not videos:
        parser.exit(1, "動画生成には --video で映像素材を指定するか、映像入りの --project を使ってください。\n")
    if not midi:
        parser.exit(1, "動画生成には --midi が必要です。\n")

    print("映像素材:")
    for i, video in enumerate(videos):
        print(f"  [{i}] {Path(video).name}（{'動画' if is_video_source(video) else '画像'}）")
    if not video_assignments:
        # 指定が無い場合はファイル名一致→順番で素材へ割り当てる。
        for i, path in enumerate(materials):
            stem = Path(path).stem.lower()
            video_assignments[i] = next(
                (j for j, video in enumerate(videos)
                 if Path(video).stem.lower() in stem or stem in Path(video).stem.lower()),
                i % len(videos),
            )

    timeline = str(outdir / "video_timeline.json")
    mp4 = str(outdir / "otomad_result.mp4")
    try:
        clip_count = build_video_timeline(materials, midi, bpm, base, assignments, base_notes,
                                          regions, videos, video_assignments, timeline,
                                          width, height, fps, profiles)
        print(f"タイムライン: {clip_count}クリップ → {timeline}")
        segments = render_video(timeline, str(wav), mp4, width, height, fps, args.gap_mode, progress)
    except Exception as ex:
        parser.exit(1, f"\n動画生成エラー: {type(ex).__name__}: {ex}\n")
    print(f"\n動画: {segments}セグメント / {width}x{height} {fps}fps")
    print(f"MP4: {mp4}")


if __name__ == "__main__":
    main()
