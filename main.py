import asyncio
import json
import math
import os
import tempfile
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import flet as ft
from pydub import AudioSegment, silence
import mido
import numpy as np


@dataclass
class Clip:
    source: str
    start_ms: int
    duration_ms: int
    midi_note: int
    pitch_shift: int
    gain_db: float
    pan: float


@dataclass
class Note:
    note: int
    start_tick: int
    duration_tick: int
    velocity: int
    track: int = 0


def midi_to_notes(path: str) -> tuple[list[Note], int, int]:
    mid = mido.MidiFile(path)
    tpq = mid.ticks_per_beat
    tempo = 500000
    notes = []

    # 全トラックを絶対tickへ展開し、note_on/note_offをペアリング
    active = {}
    events = []
    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))

    events.sort(key=lambda x: x[0])
    for tick, msg in events:
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (tick, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            if msg.type == "note_off" or msg.velocity == 0:
                key = (msg.channel, msg.note)
                if key in active:
                    start, vel = active.pop(key)
                    notes.append(Note(msg.note, start, max(1, tick - start), vel, track_index))

    notes.sort(key=lambda n: (n.start_tick, n.note))
    bpm = round(60000000 / tempo) if tempo else 120
    return notes, bpm, tpq


def detect_chops(audio: AudioSegment, min_silence=80, keep_ms=35) -> list[tuple[int, int]]:
    """無音で区切り、極端に短い区間は除外。"""
    if len(audio) == 0:
        return []

    threshold = max(-45, audio.dBFS - 16)
    ranges = silence.detect_nonsilent(
        audio, min_silence_len=min_silence, silence_thresh=threshold
    )
    result = []
    for start, end in ranges:
        start = max(0, start - keep_ms)
        end = min(len(audio), end + keep_ms)
        if end - start >= 40:
            result.append((start, end))
    if not result:
        result = [(0, len(audio))]
    return result


def pitch_shift(audio: AudioSegment, semitones: float) -> AudioSegment:
    """速度を変えずに近似的にピッチ変更する簡易実装。
    厳密なタイムストレッチではなく、サンプルレート変更→元レートへ戻す方式。"""
    if abs(semitones) < 0.001:
        return audio
    factor = 2 ** (semitones / 12.0)
    new_rate = max(1000, int(audio.frame_rate * factor))
    shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
    return shifted.set_frame_rate(audio.frame_rate)


def normalize_clip(audio: AudioSegment, target_dbfs=-18.0) -> AudioSegment:
    if audio.rms == 0:
        return audio
    return audio.apply_gain(target_dbfs - audio.dBFS)


def estimate_base_note(path: str) -> int:
    """音声の代表的な基音を推定し、MIDIノート番号で返す。"""
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(44100)
    if len(audio) > 3000:
        audio = audio[:3000]
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if not len(samples) or np.max(np.abs(samples)) == 0:
        return 60
    samples *= np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / audio.frame_rate)
    valid = (freqs >= 65) & (freqs <= 1200)
    if not np.any(valid):
        return 60
    frequency = freqs[valid][np.argmax(spectrum[valid])]
    return int(round(69 + 12 * math.log2(frequency / 440)))


def make_song(
    material_paths: list[str],
    midi_path: str | None,
    bgm_path: str | None,
    bpm: int,
    base_note: int,
    track_assignments: dict[int, int] | None,
    material_base_notes: list[int] | None,
    track_volumes: dict[int, float] | None,
    material_regions: list[dict] | None,
    output_wav: str,
    output_mp3: str | None,
    progress_cb=None,
):
    if progress_cb:
        progress_cb(0.02, "音声素材を読み込み中…")
    libraries = []
    for path in material_paths:
        audio = AudioSegment.from_file(path).set_channels(2).set_frame_rate(44100)
        region = (material_regions or [{}])[len(libraries)]
        start, end = int(region.get("start", 0)), int(region.get("end", len(audio)))
        selected = audio[max(0, start):min(len(audio), end)]
        chunks = [(0, len(selected))] if region.get("mode") == "raw" else detect_chops(selected)
        libraries.append((selected, chunks))

    notes = []
    tpq = 480
    if midi_path:
        if progress_cb:
            progress_cb(0.08, "MIDIを解析中…")
        notes, midi_bpm, tpq = midi_to_notes(midi_path)
        if bpm <= 0:
            bpm = midi_bpm

    if not notes:
        beat_ms = 60000 / max(1, bpm)
        # MIDIなしなら8分音符グリッドへ
        count = max(1, int(len(libraries[0][0]) / (beat_ms / 2)))
        for i in range(count):
            notes.append(Note(base_note, i * 240, 120, 100))
        tpq = 480

    max_end_tick = max(n.start_tick + n.duration_tick for n in notes)
    total_ms = int((max_end_tick / tpq) * (60000 / max(1, bpm))) + 500
    # overlay() をノートごとに呼ぶと、長いミックス全体を毎回コピーするため遅い。
    # NumPy のバッファへ直接加算して、最後に AudioSegment を一度だけ作る。
    mix_frames = math.ceil(total_ms * 44100 / 1000)
    mix_buffer = np.zeros((mix_frames, 2), dtype=np.float32)
    clip_cache = {}

    for i, n in enumerate(notes):
        material_index = (track_assignments or {}).get(n.track, n.track % len(libraries))
        material_index = material_index % len(libraries)
        material, chunks = libraries[material_index]
        selected_region = (material_regions or [{}])[material_index]
        src_start, src_end = chunks[i % len(chunks)]
        clip = material[src_start:src_end]
        # ノート長へ収める
        note_ms = max(35, int((n.duration_tick / tpq) * (60000 / max(1, bpm))))
        source_base = (material_base_notes or [base_note])[material_index]
        shift = n.note - source_base if selected_region.get("mode") != "raw" else 0
        cache_key = (material_index, i % len(chunks), shift, note_ms)
        if cache_key not in clip_cache:
            clip = pitch_shift(clip, shift)
            if len(clip) > note_ms:
                clip = clip[:note_ms]
            elif len(clip) < note_ms:
                reps = math.ceil(note_ms / max(1, len(clip)))
                clip = (clip * reps)[:note_ms]
            clip_cache[cache_key] = clip
        clip = clip_cache[cache_key]

        # ベロシティを音量へ、左右に軽く振る
        track_volume = (track_volumes or {}).get(n.track, 0.0)
        # 初期値は控えめにして、必要ならトラック音量で上げられるようにする。
        gain = -24 + (n.velocity / 127.0) * 4 + track_volume
        clip = clip.apply_gain(gain).set_frame_rate(44100).set_channels(2)
        pan = math.sin(i * 0.7) * 0.35
        clip = clip.pan(pan)

        start_ms = int((n.start_tick / tpq) * (60000 / max(1, bpm)))
        start_frame = int(start_ms * 44100 / 1000)
        samples = np.frombuffer(clip.raw_data, dtype=np.int16).reshape(-1, 2).astype(np.float32)
        end_frame = min(mix_frames, start_frame + len(samples))
        if start_frame < mix_frames and end_frame > start_frame:
            mix_buffer[start_frame:end_frame] += samples[:end_frame - start_frame]

        if progress_cb:
            progress_cb(0.1 + (i + 1) / len(notes) * 0.7, f"ノートを配置中… ({i + 1}/{len(notes)})")

    mix = AudioSegment(
        np.clip(mix_buffer, -32768, 32767).astype(np.int16).tobytes(),
        frame_rate=44100, sample_width=2, channels=2,
    )

    if bgm_path:
        if progress_cb:
            progress_cb(0.85, "BGMを合成中…")
        bgm = AudioSegment.from_file(bgm_path).set_channels(2)
        if len(bgm) < len(mix):
            reps = math.ceil(len(mix) / len(bgm))
            bgm = (bgm * reps)[:len(mix)]
        else:
            bgm = bgm[:len(mix)]
        bgm = normalize_clip(bgm, -24)
        mix = bgm.overlay(mix)

    mix = normalize_clip(mix, -1.0)
    if progress_cb:
        progress_cb(0.92, "音声ファイルを書き出し中…")
    mix.export(output_wav, format="wav")
    if output_mp3:
        mix.export(output_mp3, format="mp3", bitrate="192k")

    return len(notes), len(mix)


def build_timeline(material_path, midi_path, bpm, base_note, out_json):
    notes, midi_bpm, tpq = midi_to_notes(midi_path)
    if bpm <= 0:
        bpm = midi_bpm
    audio = AudioSegment.from_file(material_path)
    chunks = detect_chops(audio)

    items = []
    for i, n in enumerate(notes):
        s, e = chunks[i % len(chunks)]
        start_ms = int((n.start_tick / tpq) * (60000 / bpm))
        duration_ms = max(35, int((n.duration_tick / tpq) * (60000 / bpm)))
        items.append({
            "type": "video_clip",
            "source": os.path.abspath(material_path),
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "source_start_ms": s,
            "source_end_ms": e,
            "midi_note": n.note,
            "pitch_shift_semitones": n.note - base_note,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "bpm": bpm,
            "base_note": base_note,
            "clips": items,
            "template": {
                "transition": "cut",
                "scale_mode": "fit",
                "audio_linked": True,
            }
        }, f, ensure_ascii=False, indent=2)
    return len(items)


class App:
    def __init__(self, page: ft.Page):
        self.page = page
        page.title = "音MAD Auto Maker"
        page.window.width = 1050
        page.window.height = 760
        page.padding = 18

        self.materials = []
        self.material_base_notes = []
        self.material_regions = []
        self.material_list = ft.Column()
        self.track_assignments = {}
        self.track_volumes = {}
        self.assignment_list = ft.Column()
        self.midi = ft.TextField(label="MIDI (任意)", expand=True)
        self.bgm = ft.TextField(label="BGM (任意)", expand=True)
        self.output_dir = ft.TextField(label="出力フォルダ", value=str(Path.cwd()), expand=True)
        self.bpm = ft.TextField(label="BPM", value="120", width=130)
        self.base_note = ft.TextField(label="素材の基準音 (MIDI番号)", value="60", width=180)
        self.log = ft.TextField(label="ログ", multiline=True, min_lines=12, max_lines=12, read_only=True, expand=True)

        self.progress = ft.ProgressBar(value=0, expand=True)
        self.status = ft.Text("待機中", color=ft.Colors.GREY_700)

        self.file_picker = ft.FilePicker()
        self.dir_picker = ft.FilePicker()

    def refresh_materials(self):
        self.material_list.controls = [
            ft.Row([
                ft.Text(Path(path).name, expand=True),
                ft.TextField(label="基準音", value=str(self.material_base_notes[i]), width=100,
                             on_change=lambda e, n=i: self.set_material_base_note(n, e.control.value)),
                ft.Button("自動判定", on_click=lambda e, n=i: self.auto_base_note(n)),
                ft.TextField(label="開始ms", value=str(self.material_regions[i]["start"]), width=95,
                             on_change=lambda e, n=i: self.set_region(n, "start", e.control.value)),
                ft.TextField(label="終了ms", value=str(self.material_regions[i]["end"]), width=95,
                             on_change=lambda e, n=i: self.set_region(n, "end", e.control.value)),
                ft.Button("削除", on_click=lambda e, p=path: self.remove_material(p)),
                ft.Button("範囲を生成", on_click=lambda e, n=i: self.page.run_task(self.generate_region, n)),
                ft.Button("試聴", on_click=lambda e, n=i: self.page.run_task(self.preview_region, n)),
                ft.Button("別素材として登録", on_click=lambda e, n=i: self.page.run_task(self.register_region, n)),
            ])
            for i, path in enumerate(self.materials)
        ]
        for i, row in enumerate(self.material_list.controls):
            mode = ft.Dropdown(label="再生", value=self.material_regions[i]["mode"], width=145,
                options=[ft.dropdown.Option("synth", "シンセ化"), ft.dropdown.Option("raw", "そのまま")])
            mode.on_change = lambda e, n=i: self.set_region(n, "mode", e.control.value)
            row.controls.insert(5, mode)
        self.page.update()

    def set_material_base_note(self, index, value):
        try:
            self.material_base_notes[index] = int(value)
        except ValueError:
            pass

    def auto_base_note(self, index):
        try:
            self.material_base_notes[index] = estimate_base_note(self.materials[index])
            self.refresh_materials()
            self.add_log(f"基準音を自動判定: {Path(self.materials[index]).name} → MIDI {self.material_base_notes[index]}")
        except Exception as ex:
            self.add_log(f"基準音判定エラー: {type(ex).__name__}: {ex}")

    def set_region(self, index, key, value):
        if key == "mode":
            self.material_regions[index][key] = value
        else:
            try:
                self.material_regions[index][key] = int(value)
            except ValueError:
                pass

    async def generate_region(self, index):
        out = await self.export_region(index)
        if out:
            self.add_log(f"選択範囲を書き出しました: {out}")

    async def export_region(self, index):
        try:
            path = self.materials[index]
            region = self.material_regions[index]
            audio = await asyncio.to_thread(AudioSegment.from_file, path)
            start = max(0, int(region["start"]))
            end = min(len(audio), int(region["end"]))
            if end <= start:
                raise ValueError("切り出し範囲が不正です")
            outdir = Path(self.output_dir.value.strip() or ".")
            outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / f"{Path(path).stem}_selected_{index}.wav"
            await asyncio.to_thread(audio[start:end].export, str(out), format="wav")
            return out
        except Exception as ex:
            self.add_log(f"範囲生成エラー: {type(ex).__name__}: {ex}")
            return None

    async def preview_region(self, index):
        out = await self.export_region(index)
        if not out:
            return
        try:
            # Flet 1.x には Audio コントロールがないため、利用可能なOSプレイヤーを起動する。
            player = next((p for p in ("ffplay", "paplay", "aplay") if shutil.which(p)), None)
            if not player:
                raise RuntimeError("ffplay / paplay / aplay が見つかりません")
            if hasattr(self, "preview_process") and self.preview_process.poll() is None:
                self.preview_process.terminate()
            command = ([player, "-nodisp", "-autoexit", "-loglevel", "quiet", "-af", "volume=0.25", str(out)]
                       if player == "ffplay" else [player, str(out)])
            self.preview_process = subprocess.Popen(command)
            self.add_log(f"試聴中（ffplay時は25%）: {out}")
        except Exception as ex:
            self.add_log(f"試聴エラー: {type(ex).__name__}: {ex}")

    async def register_region(self, index):
        out = await self.export_region(index)
        if not out:
            return
        self.materials.append(str(out))
        self.material_base_notes.append(self.material_base_notes[index])
        self.material_regions.append({"start": 0, "end": len(AudioSegment.from_file(out)), "mode": "synth"})
        self.refresh_materials()
        self.refresh_assignments()
        self.add_log(f"切り出した範囲を別素材として登録: {out.name}")

    def remove_material(self, path):
        index = self.materials.index(path)
        self.materials.remove(path)
        self.material_base_notes.pop(index)
        self.material_regions.pop(index)
        self.refresh_assignments()
        self.refresh_materials()

    def refresh_assignments(self):
        if not self.midi.value.strip() or not self.materials:
            self.assignment_list.controls = []
            return
        try:
            mid = mido.MidiFile(self.midi.value.strip())
            tracks = [(i, (track.name or f"トラック {i + 1}")) for i, track in enumerate(mid.tracks)]
            controls = []
            for track_index, name in tracks:
                current = self.track_assignments.get(track_index, track_index % len(self.materials))
                current = min(current, len(self.materials) - 1)
                dropdown = ft.Dropdown(
                    value=str(current),
                    options=[ft.dropdown.Option(str(i), Path(path).name) for i, path in enumerate(self.materials)],
                    expand=True,
                )
                dropdown.on_change = lambda e, t=track_index: self.set_track_assignment(t, e.control.value)
                volume = ft.TextField(label="音量dB", value=str(self.track_volumes.get(track_index, 0)), width=100,
                    on_change=lambda e, t=track_index: self.set_track_volume(t, e.control.value))
                controls.append(ft.Row([ft.Text(name, width=180), dropdown, volume]))
            self.assignment_list.controls = controls
        except Exception:
            self.assignment_list.controls = []

    def set_track_assignment(self, track, material_index):
        self.track_assignments[track] = int(material_index)

    def set_track_volume(self, track, value):
        try:
            self.track_volumes[track] = float(value)
        except ValueError:
            pass

    def project_data(self):
        return {"materials": self.materials, "material_base_notes": self.material_base_notes,
                "material_regions": self.material_regions, "track_assignments": self.track_assignments,
                "track_volumes": self.track_volumes, "midi": self.midi.value, "bgm": self.bgm.value,
                "bpm": self.bpm.value, "base_note": self.base_note.value, "output_dir": self.output_dir.value}

    def save_project(self, e):
        path = Path(self.output_dir.value.strip() or ".") / "otomad_project.json"
        path.write_text(json.dumps(self.project_data(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.add_log(f"プロジェクト保存: {path}")

    def load_project(self, e):
        path = Path(self.output_dir.value.strip() or ".") / "otomad_project.json"
        if not path.exists():
            self.add_log(f"プロジェクトがありません: {path}")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.materials = data.get("materials", [])
        self.material_base_notes = data.get("material_base_notes", [60] * len(self.materials))
        self.material_regions = data.get("material_regions", [{"start": 0, "end": 0, "mode": "synth"} for _ in self.materials])
        self.track_assignments = {int(k): int(v) for k, v in data.get("track_assignments", {}).items()}
        self.track_volumes = {int(k): float(v) for k, v in data.get("track_volumes", {}).items()}
        for field in (self.midi, self.bgm, self.bpm, self.base_note, self.output_dir):
            key = {self.midi: "midi", self.bgm: "bgm", self.bpm: "bpm", self.base_note: "base_note", self.output_dir: "output_dir"}[field]
            field.value = data.get(key, field.value)
        self.refresh_materials(); self.refresh_assignments()
        self.add_log(f"プロジェクト読み込み: {path}")

    def auto_assign_tracks(self):
        if not self.midi.value.strip() or not self.materials:
            return
        mid = mido.MidiFile(self.midi.value.strip())
        for i, track in enumerate(mid.tracks):
            name = (track.name or "").lower()
            for j, path in enumerate(self.materials):
                if Path(path).stem.lower() in name:
                    self.track_assignments[i] = j
                    break
        self.refresh_assignments(); self.add_log("トラックを素材名で自動割り当てしました")

    async def add_material(self, e):
        await self.pick_file(None, ["wav", "mp3", "flac", "ogg"])

    async def pick_material(self):
        try:
            result = await self.file_picker.pick_files(allow_multiple=True, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["wav", "mp3", "flac", "ogg"])
            if result:
                self.materials.extend(item.path or item.name for item in result)
                self.material_base_notes.extend([60] * len(result))
                self.material_regions.extend([{"start": 0, "end": 0, "mode": "synth"} for _ in result])
                for i, item in enumerate(result):
                    try:
                        duration = len(AudioSegment.from_file(item.path or item.name))
                    except Exception:
                        duration = 0
                    self.material_regions[-len(result) + i]["end"] = duration
                self.refresh_assignments()
                self.refresh_materials()
        except Exception as ex:
            self.add_log(f"素材選択エラー: {type(ex).__name__}: {ex}")

    def add_log(self, s):
        self.log.value = (self.log.value + "\n" + s).strip()
        self.page.update()

    async def pick_file(self, field, extensions):
        try:
            result = await self.file_picker.pick_files(
                allow_multiple=False,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=extensions,
            )

            if result:
                field.value = result[0].path or result[0].name
                if field is self.midi:
                    self.track_assignments = {}
                    self.refresh_assignments()
                self.page.update()

        except Exception as ex:
            self.add_log(f"ファイル選択エラー: {type(ex).__name__}: {ex}")

    async def pick_dir(self):
        try:
            result = await self.dir_picker.get_directory_path()

            if result:
                self.output_dir.value = result
                self.page.update()

        except Exception as ex:
            self.add_log(f"出力先選択エラー: {type(ex).__name__}: {ex}")

    async def run_audio(self, e):
        try:
            if not self.materials:
                raise ValueError("音声素材を指定してください。")
            midi = self.midi.value.strip() or None
            bgm = self.bgm.value.strip() or None
            bpm = int(float(self.bpm.value))
            base_note = int(self.base_note.value)
            assignments = dict(self.track_assignments)
            outdir = Path(self.output_dir.value.strip() or ".")
            outdir.mkdir(parents=True, exist_ok=True)
            wav = str(outdir / "otomad_result.wav")
            mp3 = str(outdir / "otomad_result.mp3")

            self.progress.value = 0
            self.add_log("制作開始…")
            self.page.update()

            def cb(v, message="制作中…"):
                self.progress.value = v
                self.status.value = f"{message} {v * 100:.0f}%"
                self.page.update()

            count, duration = await asyncio.to_thread(
                make_song, self.materials, midi, bgm, bpm, base_note, assignments,
                self.material_base_notes, self.track_volumes, self.material_regions,
                wav, mp3, cb
            )
            self.progress.value = 1
            self.status.value = "制作完了"
            self.add_log(f"完了: {count}ノート / {duration/1000:.1f}秒")
            self.add_log(f"WAV: {wav}")
            self.add_log(f"MP3: {mp3}")
        except Exception as ex:
            self.add_log(f"エラー: {type(ex).__name__}: {ex}")

    async def run_video_template(self, e):
        try:
            if not self.materials or not self.midi.value.strip():
                raise ValueError("動画テンプレートには素材とMIDIが必要です。")
            outdir = Path(self.output_dir.value.strip() or ".")
            outdir.mkdir(parents=True, exist_ok=True)
            bpm = int(float(self.bpm.value))
            base = int(self.base_note.value)
            out = str(outdir / "video_timeline.json")
            count = await asyncio.to_thread(
                build_timeline, self.materials[0], self.midi.value.strip(),
                bpm, base, out
            )
            self.add_log(f"動画テンプレート生成: {count}クリップ")
            self.add_log(f"JSON: {out}")
        except Exception as ex:
            self.add_log(f"エラー: {type(ex).__name__}: {ex}")

    def build(self):
        pick_midi = ft.Button("MIDIを選択", on_click=lambda e: self.page.run_task(self.pick_file, self.midi, ["mid", "midi"]))
        pick_bgm = ft.Button("BGMを選択", on_click=lambda e: self.page.run_task(self.pick_file, self.bgm, ["wav", "mp3", "flac", "ogg"]))
        pick_dir = ft.Button("出力先", on_click=lambda e: self.page.run_task(self.pick_dir))

        return ft.Column([
            ft.Text("🎵 音MAD Auto Maker", size=30, weight=ft.FontWeight.BOLD),
            ft.Text("MIDIの音階に合わせて音声素材をチョップ・ピッチ調整し、自動演奏します。"),
            ft.Text("素材一覧", size=18, weight=ft.FontWeight.BOLD),
            self.material_list,
            ft.Button("＋ 音声素材を追加", on_click=lambda e: self.page.run_task(self.pick_material)),
            ft.Row([self.midi, pick_midi]),
            ft.Text("素材の割り当て", size=18, weight=ft.FontWeight.BOLD),
            self.assignment_list,
            ft.Row([ft.Button("自動割り当て", on_click=self.auto_assign_tracks),
                    ft.Button("プロジェクト保存", on_click=self.save_project),
                    ft.Button("プロジェクト読み込み", on_click=self.load_project)]),
            ft.Row([self.bgm, pick_bgm]),
            ft.Row([self.bpm, self.base_note]),
            ft.Row([self.output_dir, pick_dir]),
            ft.Divider(),
            ft.Row([
                ft.Button("▶ 自動制作 (WAV/MP3)", on_click=lambda e: self.page.run_task(self.run_audio, e)),
                ft.Button("🎬 動画テンプレJSON", on_click=lambda e: self.page.run_task(self.run_video_template, e)),
            ]),
            self.progress,
            self.status,
            self.log,
        ], expand=True, scroll=ft.ScrollMode.AUTO)


def main(page: ft.Page):
    page.add(App(page).build())


if __name__ == "__main__":
    ft.run(main)
