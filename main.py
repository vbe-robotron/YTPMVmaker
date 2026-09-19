import asyncio
import json
import math
import os
import sys
import tempfile
import shutil
import subprocess
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import flet as ft
from pydub import AudioSegment, silence
import mido
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MATERIALS_DIR = PROJECT_ROOT / "materials"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_project_directories() -> tuple[Path, Path]:
    """素材と成果物の標準フォルダを作成して返す。"""
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return MATERIALS_DIR, OUTPUTS_DIR


def timestamp_label() -> str:
    """同一秒内の複数生成も区別できるタイムスタンプを返す。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _worker_count(env_name: str, default: int, cap: int | None = None) -> int:
    """並列ワーカー数を返す。環境変数で上書きでき、1を指定すると逐次実行になる。"""
    count = max(1, int(default))
    raw = os.environ.get(env_name, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            count = value
    if cap is not None:
        count = min(count, max(1, int(cap)))
    return max(1, count)


def _audio_workers() -> int:
    """音声デコード・解析用のワーカー数。

    大きな素材を同時にデコードしすぎてメモリを使い切らないよう上限を設ける。
    """
    return _worker_count("YTPMV_WORKERS", os.cpu_count() or 2, cap=4)


def _render_workers() -> int:
    """ノート描画（ピッチシフト・ミックス）用のワーカー数。"""
    return _worker_count("YTPMV_WORKERS", os.cpu_count() or 2)


def _video_workers() -> int:
    """映像セグメントの並列エンコード数。

    x264 は1プロセスでも複数スレッドを使うため、同時実行数を増やしすぎると
    逆に遅くなる。CPUコア数の半分（最大8）を既定とする。
    """
    default = max(2, min(8, (os.cpu_count() or 2) // 2))
    return _worker_count("YTPMV_VIDEO_WORKERS", default)


def _parallel_map(func, items, workers: int | None = None) -> list:
    """独立な処理をスレッドで並列実行し、入力順の結果リストを返す。

    pydub/audioop・NumPy・ffmpeg などの処理は GIL を解放するため、
    スレッドでも十分に並列化できる。workers が1以下なら逐次実行する。
    """
    items = list(items)
    if len(items) <= 1:
        return [func(item) for item in items]
    count = _render_workers() if workers is None else max(1, int(workers))
    count = min(count, len(items))
    if count <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(func, items))


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
    order = 0
    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, order, track_index, msg))
            order += 1

    events.sort(key=lambda x: (x[0], x[1]))
    for tick, _, track_index, msg in events:
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(track_index, msg.channel, msg.note)] = (tick, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            if msg.type == "note_off" or msg.velocity == 0:
                key = (track_index, msg.channel, msg.note)
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


def _load_material_library(path: str, region: dict, normalize: bool) -> tuple[AudioSegment, list[tuple[int, int]]]:
    """素材を読み込み、使用区間とチョップ位置を返す。

    normalize=True のときは音声生成と同じ 44.1kHz/ステレオへ揃える。
    """
    audio = AudioSegment.from_file(path)
    if normalize:
        audio = audio.set_channels(2).set_frame_rate(44100)
    start, end = int(region.get("start", 0)), int(region.get("end", 0))
    if end <= start:
        # 終了位置が未設定（0など）の場合は素材全体を使う。
        end = len(audio)
    selected = audio[max(0, start):min(len(audio), end)]
    chunks = [(0, len(selected))] if region.get("mode") == "raw" else detect_chops(selected)
    return selected, chunks


def _wsola_shift(samples: np.ndarray, factor: float, frame_rate: int) -> np.ndarray:
    """WSOLA (Waveform Similarity Overlap-Add) によるタイムストレッチ。

    factor > 1 で再生速度を上げる（後でリサンプルで音程も上げる）。
    音の長さを変えるためのコアで、最終的には元の長さに戻してピッチシフトに使う。
    """
    # ウィンドウ・ホップサイズを音程補正向けに調整（短い素材でも動くよう小さめに）
    win_ms = 40         # 分析ウィンドウ 40ms
    hop_ms = 10         # 合成ホップ 10ms
    win_size = max(64, int(frame_rate * win_ms / 1000))
    hop_syn = max(16, int(frame_rate * hop_ms / 1000))
    hop_ana = max(16, int(round(hop_syn * factor)))
    search = max(8, win_size // 4)  # 類似位置の探索幅

    n = len(samples)
    if n < win_size:
        # 極端に短い素材はリサンプルで代用
        return samples

    # ハニング窓（正規化済み）
    window = np.hanning(win_size).astype(np.float32)

    # 出力バッファ（元と同じ長さを目標に）
    target_len = n  # 後段で元長さに合わせるので概算でよい
    out = np.zeros(target_len + win_size * 2, dtype=np.float64)
    norm = np.zeros_like(out)

    pos_ana = 0
    pos_syn = 0
    prev_ana = 0

    while pos_syn + win_size <= len(out):
        if pos_ana + win_size > n:
            break

        # 最初のフレーム以外は前フレームとの類似位置を探す
        if pos_ana == 0:
            best = pos_ana
        else:
            lo = max(0, pos_ana - search)
            hi = min(n - win_size, pos_ana + search)
            if lo >= hi:
                best = pos_ana
            else:
                # 前フレームの後半と次フレーム候補の前半の相関で最良位置を選ぶ
                ref = samples[prev_ana: prev_ana + win_size] * window
                best_corr = -np.inf
                best = pos_ana
                step = max(1, (hi - lo) // 16)  # 探索を粗くして高速化
                for start in range(lo, hi, step):
                    candidate = samples[start: start + win_size] * window
                    corr = float(np.dot(ref, candidate))
                    if corr > best_corr:
                        best_corr = corr
                        best = start

        frame = (samples[best: best + win_size] * window).astype(np.float64)
        out[pos_syn: pos_syn + win_size] += frame
        norm[pos_syn: pos_syn + win_size] += window.astype(np.float64) ** 2

        prev_ana = best
        pos_ana = best + hop_ana
        pos_syn += hop_syn

    # 正規化（ゼロ除算を防ぐ）
    safe = norm > 1e-8
    out[safe] /= norm[safe]
    # 合成長を元の長さにクリップ
    out = out[:target_len]
    return out.astype(np.float32)


def pitch_shift(audio: AudioSegment, semitones: float) -> AudioSegment:
    """WSOLA + リサンプルによる高品質ピッチシフト。

    長さを保ちながら音程だけを変える。
    - |semitones| < 0.5 : 誤差が無視できるため高速パスを使う
    - それ以外 : WSOLA でタイムストレッチ → リサンプルで音程補正
    """
    if abs(semitones) < 0.5:
        if abs(semitones) < 0.01:
            return audio
        # 小さいずれはレート操作のみ（速度変化は ±3% 未満で聴感上無視できる）
        factor = 2 ** (semitones / 12.0)
        new_rate = max(1000, int(audio.frame_rate * factor))
        return audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(audio.frame_rate)

    factor = 2 ** (semitones / 12.0)   # > 1 で高音化
    frame_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width

    raw = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)

    if channels == 2:
        raw = raw.reshape(-1, 2)
        out_ch = []
        for ch in range(2):
            # WSOLA で 1/factor 倍に時間伸縮（これをリサンプルで元の長さに戻すと音程が変わる）
            stretched = _wsola_shift(raw[:, ch], 1.0 / factor, frame_rate)
            # リサンプルで元の長さに戻す → 音程が factor 倍になる
            n_orig = len(raw)
            n_str = len(stretched)
            if n_str > 0 and n_orig > 0:
                indices = np.linspace(0, n_str - 1, n_orig)
                i0 = np.floor(indices).astype(np.int64)
                i1 = np.clip(i0 + 1, 0, n_str - 1)
                frac = (indices - i0).astype(np.float32)
                resampled = stretched[i0] * (1 - frac) + stretched[i1] * frac
            else:
                resampled = np.zeros(n_orig, dtype=np.float32)
            out_ch.append(resampled)
        result = np.stack(out_ch, axis=1).flatten()
    else:
        stretched = _wsola_shift(raw, 1.0 / factor, frame_rate)
        n_orig = len(raw)
        n_str = len(stretched)
        if n_str > 0 and n_orig > 0:
            indices = np.linspace(0, n_str - 1, n_orig)
            i0 = np.floor(indices).astype(np.int64)
            i1 = np.clip(i0 + 1, 0, n_str - 1)
            frac = (indices - i0).astype(np.float32)
            result = stretched[i0] * (1 - frac) + stretched[i1] * frac
        else:
            result = np.zeros(n_orig, dtype=np.float32)

    clipped = np.clip(result, -32768, 32767).astype(np.int16)
    return audio._spawn(clipped.tobytes(), overrides={
        "frame_rate": frame_rate, "channels": channels, "sample_width": sample_width,
    })


def normalize_clip(audio: AudioSegment, target_dbfs=-18.0) -> AudioSegment:
    if audio.rms == 0:
        return audio
    return audio.apply_gain(target_dbfs - audio.dBFS)



def _estimate_base_note_hps(samples: np.ndarray, frame_rate: int) -> int:
    """HPS（高調波積スペクトル）＋マルチフレーム投票による基音推定。

    単純な FFT ピークは倍音を誤検出しやすい。HPS は各倍音を間引いて積算することで
    基音成分だけを強調できる。複数フレームの投票により雑音に頑健になる。

    返値は MIDI ノート番号（21〜108）。
    """
    frame_size = 4096   # 周波数分解能を確保（44100Hz で約10.7Hz/bin）
    hop = frame_size // 2
    n_harmonics = 5     # HPS の次数

    f_lo, f_hi = 55.0, 1200.0  # 探索周波数帯域 (Hz)

    window = np.hanning(frame_size).astype(np.float32)
    vote: dict[int, float] = {}

    n = len(samples)
    frame_starts = list(range(0, n - frame_size + 1, hop))
    if not frame_starts:
        return 60

    frame_rms = []
    for start in frame_starts:
        seg = samples[start: start + frame_size]
        frame_rms.append(float(np.sqrt(np.mean(seg ** 2))))

    peak_rms = max(frame_rms) if frame_rms else 0.0
    threshold_rms = peak_rms * 0.30

    for fi, start in enumerate(frame_starts):
        if frame_rms[fi] < threshold_rms:
            continue

        seg = samples[start: start + frame_size].copy() * window
        spectrum = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(frame_size, 1.0 / frame_rate)

        # HPS: 各倍音の間引き積算
        hps = spectrum.copy().astype(np.float64)
        for h in range(2, n_harmonics + 1):
            decimated_len = len(spectrum) // h
            if decimated_len < 2:
                break
            src_idx = np.arange(decimated_len) * h
            src_idx = np.clip(src_idx, 0, len(spectrum) - 1)
            hps[:decimated_len] *= spectrum[src_idx]
            hps[decimated_len:] = 0.0

        valid = (freqs >= f_lo) & (freqs <= f_hi) & (np.arange(len(freqs)) < len(hps))
        if not np.any(valid):
            continue

        f0_hz = float(freqs[valid][np.argmax(hps[valid])])
        if f0_hz <= 0:
            continue

        midi_note = int(round(69 + 12 * math.log2(f0_hz / 440.0)))
        midi_note = max(21, min(108, midi_note))
        vote[midi_note] = vote.get(midi_note, 0.0) + frame_rms[fi]

    if not vote:
        return 60

    # 隣接半音への票の拡散で安定化してから最多票を採用
    smoothed: dict[int, float] = {}
    for note, weight in vote.items():
        for offset, decay in ((0, 1.0), (-1, 0.3), (1, 0.3)):
            n2 = note + offset
            if 21 <= n2 <= 108:
                smoothed[n2] = smoothed.get(n2, 0.0) + weight * decay
    return max(smoothed, key=smoothed.__getitem__)


def analyze_material(path: str) -> dict:
    """素材の長さ・基準音・明るさ・音の種類をまとめて解析する。

    戻り値は {"duration_ms", "base_note", "brightness", "dbfs",
              "flatness", "harmonicity", "attack_ms", "crest",
              "recommended_mode", "recommended_reason"}。
    brightness はスペクトル重心(Hz)で、高いほど明るい（倍音の多い）音。
    base_note は HPS＋マルチフレーム投票で推定し、旧実装より倍音誤検出が少ない。
    """
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(44100)
    result = {
        "duration_ms": len(audio), "base_note": 60, "brightness": 440.0, "dbfs": -60.0,
        "flatness": 1.0, "harmonicity": 0.0, "attack_ms": 0.0, "crest": 1.0, "sustain_ratio": 1.0,
    }
    # アタック〜安定部を含む最大 5 秒を解析する
    segment = audio[:5000] if len(audio) > 5000 else audio
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if not len(samples) or np.max(np.abs(samples)) == 0:
        result["recommended_mode"], result["recommended_reason"] = recommend_play_mode(result)
        return result
    try:
        result["dbfs"] = round(float(segment.dBFS), 2)
    except (ValueError, ZeroDivisionError):
        pass

    result["flatness"], result["harmonicity"] = _spectral_metrics(samples, audio.frame_rate)
    result["attack_ms"], result["crest"], result["sustain_ratio"] = _envelope_metrics(samples, audio.frame_rate)

    # 基準音: HPS＋マルチフレーム投票（旧: 単一フレームの FFT ピーク）
    result["base_note"] = _estimate_base_note_hps(samples, audio.frame_rate)

    # 明るさ（スペクトル重心）: 全帯域の平均スペクトルから計算
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), 1 / audio.frame_rate)
    valid_b = (freqs >= 65) & (freqs <= 8000)
    if np.any(valid_b):
        band = spectrum[valid_b]
        total = float(band.sum())
        if total > 0:
            result["brightness"] = float((freqs[valid_b] * band).sum() / total)
    result["recommended_mode"], result["recommended_reason"] = recommend_play_mode(result)
    return result




def analyze_materials(paths: list[str]) -> list[dict | Exception]:
    """複数素材を並列に解析する（入力順）。

    解析に失敗した素材は例外オブジェクトをそのまま返すので、
    呼び出し側でログ表示などを行える。
    """
    def run(path):
        try:
            return analyze_material(path)
        except Exception as ex:
            return ex

    return _parallel_map(run, paths, workers=_audio_workers())


def _spectral_metrics(samples, frame_rate):
    """スペクトル平坦度とハーモニック成分比を求める。

    flatness: 0に近いほど tonal（音程感がある）、1に近いほどノイズ的。
    harmonicity: 基音の倍音に含まれるエネルギー比率（1に近いほど音程が明瞭）。
    """
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), 1 / frame_rate)
    band = (freqs >= 65) & (freqs <= 8000)
    power = spectrum[band] ** 2 + 1e-12
    if not np.any(band) or power.size == 0:
        return 1.0, 0.0
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    flatness = max(0.0, min(1.0, flatness))

    low = (freqs >= 65) & (freqs <= 1200)
    if not np.any(low):
        return flatness, 0.0
    f0 = float(freqs[low][np.argmax(spectrum[low])])
    total = float(power.sum())
    if total <= 0 or f0 <= 0:
        return flatness, 0.0
    harmonic = 0.0
    for k in range(1, 17):
        target = f0 * k
        if target > 8000:
            break
        # ±3%を倍音成分として合算する
        mask = (freqs >= target * 0.97) & (freqs <= target * 1.03) & band
        harmonic += float((spectrum[mask] ** 2).sum())
    return flatness, max(0.0, min(1.0, harmonic / total))


def _envelope_metrics(samples, frame_rate):
    """アタック時間(ms)・ピーク/平均のRMS比・持続比率を求める。

    打楽器や効果音はアタックが鋭く、減衰が速い（持続比率が小さい）。
    """
    frame = max(1, int(frame_rate * 0.01))  # 10ms
    usable = len(samples) - len(samples) % frame
    if usable < frame:
        return 0.0, 1.0, 1.0
    frames = samples[:usable].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return 0.0, 1.0, 1.0
    attack_ms = float(int(np.argmax(rms)) * 10)
    crest = peak / max(1e-6, float(rms.mean()))
    # ピークの20%以上を保っている最後の位置を持続時間とみなす
    above = np.nonzero(rms >= peak * 0.2)[0]
    sustain_frames = int(above[-1] - above[0] + 1) if len(above) else 1
    sustain_ratio = min(1.0, (sustain_frames * frame) / max(1, len(samples)))
    return attack_ms, crest, sustain_ratio


def recommend_play_mode(profile: dict) -> tuple[str, str]:
    """解析結果から再生モード（synth / raw）を推定する。

    - 音程感があり、ある程度持続する音 → synth（ピッチを変えて演奏できる）
    - アタックが鋭く減衰が速い一発音（打楽器・効果音）→ raw（原音のまま鳴らす）
    """
    harmonicity = float(profile.get("harmonicity", 0.0))
    flatness = float(profile.get("flatness", 1.0))
    attack_ms = float(profile.get("attack_ms", 0.0))
    crest = float(profile.get("crest", 1.0))
    sustain = float(profile.get("sustain_ratio", 1.0))

    if harmonicity >= 0.45 and flatness <= 0.35 and sustain >= 0.35:
        return "synth", (f"音程感があり持続する（倍音成分 {harmonicity:.0%} / "
                         f"平坦度 {flatness:.2f} / 持続 {sustain:.0%}）")
    if attack_ms <= 25 and (crest >= 3.0 or sustain < 0.2):
        return "raw", (f"打楽器・効果音らしい一発音（アタック {attack_ms:.0f}ms / "
                       f"ピーク比 {crest:.1f} / 持続 {sustain:.0%}）")
    if harmonicity >= 0.30 and flatness <= 0.5:
        return "synth", f"音程感がある（倍音成分 {harmonicity:.0%} / 持続 {sustain:.0%}）"
    return "raw", (f"音程感が弱くノイズ的（倍音成分 {harmonicity:.0%} / "
                   f"平坦度 {flatness:.2f} / 持続 {sustain:.0%}）")


def resolve_region_modes(material_paths, material_regions, material_profiles=None) -> list[dict]:
    """regionのmodeが"auto"（未指定含む）の素材を解析して synth/raw を確定させる。"""
    resolved = []
    pending = []  # (index, path) 解析が必要な素材
    for index, path in enumerate(material_paths):
        region = dict((material_regions or [{}])[index]) if material_regions and index < len(material_regions) else {}
        mode = region.get("mode") or "auto"
        if mode == "auto":
            profile = material_profiles[index] if material_profiles and index < len(material_profiles) else None
            if profile and "recommended_mode" in profile:
                mode = profile.get("recommended_mode") or "synth"
            else:
                pending.append((index, path))
        region["mode"] = mode
        resolved.append(region)

    if pending:
        # 解析は重いので、必要な素材だけまとめて並列に実行する。
        analyzed = analyze_materials([path for _, path in pending])
        for (index, _), profile in zip(pending, analyzed):
            if isinstance(profile, Exception):
                profile = {}
            resolved[index]["mode"] = profile.get("recommended_mode") or "synth"
    return resolved


def estimate_base_note(path: str) -> int:
    """音声の代表的な基音を推定し、MIDIノート番号で返す。"""
    return analyze_material(path)["base_note"]


def auto_assign_materials(
    notes: list["Note"],
    material_profiles: list[dict],
    track_names: dict[int, str] | None = None,
) -> dict[int, int]:
    """トラックごとに「一番自然に鳴る」素材を選んで割り当てる。

    評価基準（スコアが小さいほど良い）:
    - トラック名に素材名が含まれる場合は、その素材だけを候補にする（ユーザーの指定を優先）
    - 素材の基準音とトラックの平均音高の差が小さいほど良い
    - 1オクターブを超えるピッチ変更は大きく減点
    - 素材の明るさ（スペクトル重心）とトラックの音域の順位が近いほど良い
    - 同じ素材の重複使用は軽く減点
    """
    if not notes or not material_profiles:
        return {}

    by_track: dict[int, list[Note]] = {}
    for note in notes:
        by_track.setdefault(note.track, []).append(note)

    mean_notes = {track: sum(n.note for n in ns) / len(ns) for track, ns in by_track.items()}
    # 音数の多いトラック（主旋律）から先に決めると、良い素材を優先的に取れる。
    tracks = sorted(by_track, key=lambda track: -len(by_track[track]))

    def normalize(values: dict) -> dict:
        if not values:
            return {}
        low, high = min(values.values()), max(values.values())
        if high == low:
            return {key: 0.5 for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}

    brightness = normalize(
        {i: float(profile.get("brightness", 440.0)) for i, profile in enumerate(material_profiles)}
    )
    pitch = normalize(mean_notes)
    stems = [Path(str(profile.get("path") or "")).stem.lower() for profile in material_profiles]

    used: dict[int, int] = {}
    result: dict[int, int] = {}
    for track in tracks:
        name = ((track_names or {}).get(track) or "").lower()
        matched = {i for i, stem in enumerate(stems) if stem and stem in name}
        # トラック名に素材名が含まれている場合は、その素材を優先候補にする。
        candidates = sorted(matched) if matched else list(range(len(material_profiles)))
        best_index, best_score = candidates[0], None
        for index in candidates:
            profile = material_profiles[index]
            base_note = int(profile.get("base_note", 60))
            shift = mean_notes[track] - base_note
            score = abs(shift)
            score += max(0.0, abs(shift) - 12) * 2.0
            score += abs(pitch[track] - brightness[index]) * 4.0
            score += used.get(index, 0) * 2.0
            if best_score is None or score < best_score:
                best_index, best_score = index, score
        result[track] = best_index
        used[best_index] = used.get(best_index, 0) + 1
    return result


def auto_track_volumes(
    notes: list["Note"], bpm: int, tpq: int, bgm_gain_db: float = 0.0,
    has_bgm: bool = False,
) -> dict[int, float]:
    """トラックごとの音量バランスを自動調整してdBで返す。

    - 音数が多い（密な）トラックは合計エネルギーが大きくなりすぎないよう適度に抑える
    - 主旋律トラックは埋もれないよう適度に引き上げる
    - 低音・高音の聴感特性を補正
    - MIDIベロシティの相対差を適正範囲（±1.0dB）で反映し、極端な格差を防止
    - 全体として極端に大きすぎる・小さすぎるトラックが出ないよう適正範囲に抑制
    """
    if not notes:
        return {}

    by_track: dict[int, list[Note]] = {}
    for note in notes:
        by_track.setdefault(note.track, []).append(note)

    beat_ms = 60000 / max(1, bpm)
    densities: dict[int, float] = {}
    mean_notes: dict[int, float] = {}
    mean_velocities: dict[int, float] = {}
    for track, ns in by_track.items():
        end_tick = max(n.start_tick + n.duration_tick for n in ns)
        seconds = max(0.1, end_tick / max(1, tpq) * beat_ms / 1000)
        densities[track] = len(ns) / seconds
        mean_notes[track] = sum(n.note for n in ns) / len(ns)
        mean_velocities[track] = sum(n.velocity for n in ns) / len(ns)

    sorted_densities = sorted(densities.values())
    reference = sorted_densities[len(sorted_densities) // 2] or 1.0
    main_track = max(by_track, key=lambda track: len(by_track[track]))

    # BGMがある場合のベースリフト（BGM音量に比例して適度に持ち上げる）
    bgm_lift = 0.0
    if has_bgm:
        bgm_lift = 0.5 + max(-1.5, min(2.0, float(bgm_gain_db) * 0.4))

    # 全トラックのベロシティ平均の正規化（極端な格差を出さないよう滑らかに）
    all_velocities = list(mean_velocities.values())
    vel_min = min(all_velocities)
    vel_max = max(all_velocities)
    vel_range = max(1.0, vel_max - vel_min)

    volumes: dict[int, float] = {}
    for track in by_track:
        # 音数密度の補正（急激な倍率にならないよう制限）
        ratio = max(0.1, densities[track]) / max(0.1, reference)
        density_comp = -1.0 * math.log2(ratio)
        volume = max(-2.0, min(2.0, density_comp))
        
        if track == main_track:
            volume += 1.2
            
        # 音域補正（極端にならないよう穏やかに）
        volume += max(-1.0, min(1.0, (72 - mean_notes[track]) * 0.03))
        
        # ベロシティ補正（最大±1.0dB）
        vel_norm = (mean_velocities[track] - vel_min) / vel_range
        volume += max(-1.0, min(1.0, (vel_norm - 0.5) * 2.0))
        
        # BGM補正
        volume += bgm_lift
        
        # トラック全体音量は±3.5dBの安定レンジにクランプ
        volumes[track] = round(max(-3.5, min(3.5, volume)), 1)
    return volumes


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
    solo_track: int | None = None,
    bgm_gain_db: float = 0.0,
    material_profiles: list[dict] | None = None,
):
    """MIDIノートに沿って素材を配置し、WAV/MP3を書き出す。

    solo_track にトラック番号を渡すと、そのトラックのノートだけを描画し、
    BGMを合成せずにトラック単体の音声を生成する（トラック別試聴用）。
    bgm_gain_db は BGM の基準音量(-24dBFS)に対する相対ゲイン(dB)。
    再生モードが "auto" の素材は解析して synth / raw を自動判定する。
    """
    if progress_cb:
        progress_cb(0.02, "音声素材を読み込み中…")
    # "auto" の素材はここで synth / raw を確定させる。
    regions = resolve_region_modes(material_paths, material_regions, material_profiles)
    # 素材のデコードは重いので並列に読み込む（入力順は維持）。
    libraries = _parallel_map(
        lambda index: _load_material_library(material_paths[index], regions[index], True),
        range(len(material_paths)),
        workers=_audio_workers(),
    )

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

    if solo_track is not None:
        # トラック単体の試聴用に、対象トラックのノートだけへ絞り込む。
        notes = [n for n in notes if n.track == solo_track]
        if not notes:
            raise ValueError(f"トラック {solo_track + 1} にはノートがありません。")

    max_end_tick = max(n.start_tick + n.duration_tick for n in notes)
    total_ms = int((max_end_tick / tpq) * (60000 / max(1, bpm))) + 500
    # overlay() をノートごとに呼ぶと、長いミックス全体を毎回コピーするため遅い。
    # NumPy のバッファへ直接加算して、最後に AudioSegment を一度だけ作る。
    mix_frames = math.ceil(total_ms * 44100 / 1000)
    mix_buffer = np.zeros((mix_frames, 2), dtype=np.float32)
    clip_cache = {}
    cache_lock = threading.Lock()

    # 各素材の音量を基準(-18 dBFS)へ均一化し、素材ごとの録音音量差による極端な大小を解消
    normalized_libraries = []
    for material, chunks in libraries:
        norm_mat = normalize_clip(material, target_dbfs=-18.0)
        normalized_libraries.append((norm_mat, chunks))
    libraries = normalized_libraries

    # 各素材のチョップ粒を事前に解析して基準音を推定しておく（ピッチ最小選択用）
    # 粒数が多い場合は重いので、最大 32 粒まで解析する
    _chunk_base_notes: dict[int, list[int]] = {}
    for mi, (material, chunks) in enumerate(libraries):
        if regions[mi].get("mode") == "raw":
            _chunk_base_notes[mi] = [0] * len(chunks)
            continue
        src_base = (material_base_notes or [base_note])[mi]
        notes_per_chunk: list[int] = []
        limit = min(len(chunks), 32)
        for ci in range(limit):
            src_start, src_end = chunks[ci]
            seg = material[src_start:src_end].set_channels(1).set_frame_rate(44100)
            seg_samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
            # 粒が十分長い場合のみ HPS で基音推定、短い場合は素材基準音を使う
            if len(seg_samples) >= 4096 * 2:
                chunk_note = _estimate_base_note_hps(seg_samples, 44100)
            else:
                chunk_note = src_base
            notes_per_chunk.append(chunk_note)
        # 粒が 32 未満の場合は末尾の値で埋める
        if len(chunks) > limit:
            notes_per_chunk.extend([notes_per_chunk[-1]] * (len(chunks) - limit))
        _chunk_base_notes[mi] = notes_per_chunk

    # ノートごとのパラメータ計算（軽い）は逐次で行い、
    # ピッチシフトやゲイン・パンなどの音声処理（重い）は並列で行う。
    jobs = []
    for i, n in enumerate(notes):
        material_index = (track_assignments or {}).get(n.track, n.track % len(libraries))
        material_index = material_index % len(libraries)
        selected_region = regions[material_index]
        chunks = libraries[material_index][1]

        if selected_region.get("mode") == "raw":
            # rawモードは音程を使わないので単純な循環インデックス
            chunk_index = i % len(chunks)
            shift = 0
        else:
            # ピッチシフト量が最小になるチョップ粒を選ぶ
            chunk_base_notes = _chunk_base_notes[material_index]
            best_ci = 0
            best_shift_abs = float("inf")
            # 全粒から最小シフト量のものを選ぶ（粒が1つの場合は即決）
            for ci in range(len(chunks)):
                s = n.note - chunk_base_notes[ci]
                if abs(s) < best_shift_abs:
                    best_shift_abs = abs(s)
                    best_ci = ci
            chunk_index = best_ci
            shift = n.note - chunk_base_notes[best_ci]

        # ノート長へ収める
        note_ms = max(35, int((n.duration_tick / tpq) * (60000 / max(1, bpm))))
        # ベロシティを音量へ、左右に軽く振る
        track_volume = (track_volumes or {}).get(n.track, 0.0)
        # 初期値は控えめにして、必要ならトラック音量で上げられるようにする。
        # MIDIベロシティを十分なダイナミックレンジで反映する。
        # 低ベロシティは控えめ、高ベロシティは明確に前へ出す。
        # ベロシティのダイナミックレンジを圧縮し、小さすぎる音や突き抜ける爆音を防止
        velocity = max(1, min(127, n.velocity)) / 127.0
        velocity_gain = (velocity ** 0.5) * 6.0 - 6.0  # 最大6dBの範囲で自然に追従
        gain = -12.0 + velocity_gain + track_volume
        pan = math.sin(i * 0.7) * 0.35
        start_ms = int((n.start_tick / tpq) * (60000 / max(1, bpm)))
        jobs.append((
            i, material_index, chunk_index, note_ms, shift, gain, pan,
            int(start_ms * 44100 / 1000),
        ))

    def render_note(job):
        """1ノート分のクリップを作り、float32のステレオ波形を返す。"""
        _i, material_index, chunk_index, note_ms, shift, gain, pan, _start_frame = job
        cache_key = (material_index, chunk_index, shift, note_ms)
        with cache_lock:
            clip = clip_cache.get(cache_key)
        if clip is None:
            material, chunks = libraries[material_index]
            src_start, src_end = chunks[chunk_index]
            clip = pitch_shift(material[src_start:src_end], shift)
            if len(clip) > note_ms:
                clip = clip[:note_ms]
            elif len(clip) < note_ms and len(clip) > 0:
                # ループ繰り返しでノート長へ伸ばす。継ぎ目をクロスフェードで滑らかにする
                fade_ms = min(20, len(clip) // 4)
                reps = math.ceil(note_ms / max(1, len(clip)))
                looped = clip * reps
                # クロスフェードで継ぎ目のクリックを抑制
                if fade_ms > 0 and len(looped) > fade_ms * 2:
                    looped = looped.fade_in(fade_ms).fade_out(fade_ms)
                clip = looped[:note_ms]
            with cache_lock:
                clip = clip_cache.setdefault(cache_key, clip)
        # apply_gain/set_frame_rate/set_channels は新しいSegmentを返すため、
        # キャッシュを共有していても pan() の破壊的変更は影響しない。
        clip = clip.apply_gain(gain).set_frame_rate(44100).set_channels(2)
        clip = clip.pan(pan)
        return np.frombuffer(clip.raw_data, dtype=np.int16).reshape(-1, 2).astype(np.float32)

    total_notes = len(jobs)
    workers = min(_render_workers(), max(1, total_notes))
    # 結果を溜め込みすぎないよう、ワーカー数の数倍ずつ準備して順番にミックスする。
    batch_size = max(16, workers * 8)
    progress_span = 0.7 / max(1, total_notes)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch_start in range(0, total_notes, batch_size):
            batch = jobs[batch_start:batch_start + batch_size]
            rendered = pool.map(render_note, batch) if pool else [render_note(job) for job in batch]
            for job, samples in zip(batch, rendered):
                i, _material_index, _chunk_index, _note_ms, _shift, _gain, _pan, start_frame = job
                end_frame = min(mix_frames, start_frame + len(samples))
                if start_frame < mix_frames and end_frame > start_frame:
                    mix_buffer[start_frame:end_frame] += samples[:end_frame - start_frame]

                if progress_cb:
                    progress_cb(0.1 + (i + 1) * progress_span, f"ノートを配置中… ({i + 1}/{total_notes})")
    finally:
        if pool is not None:
            pool.shutdown()

    mix = AudioSegment(
        np.clip(mix_buffer, -32768, 32767).astype(np.int16).tobytes(),
        frame_rate=44100, sample_width=2, channels=2,
    )

    if bgm_path and solo_track is None:
        if progress_cb:
            progress_cb(0.85, "BGMを合成中…")
        bgm = AudioSegment.from_file(bgm_path).set_channels(2)
        if len(bgm) < len(mix):
            reps = math.ceil(len(mix) / len(bgm))
            bgm = (bgm * reps)[:len(mix)]
        else:
            bgm = bgm[:len(mix)]
        bgm = normalize_clip(bgm, -24)
        if bgm_gain_db:
            bgm = bgm.apply_gain(bgm_gain_db)
        mix = bgm.overlay(mix)

    mix = normalize_clip(mix, -1.0)
    if progress_cb:
        progress_cb(0.92, "音声ファイルを書き出し中…")
    mix.export(output_wav, format="wav")
    if output_mp3:
        mix.export(output_mp3, format="mp3", bitrate="192k")

    return len(notes), len(mix)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif"}
MODE_LABELS = {"auto": "自動判定", "synth": "シンセ化", "raw": "そのまま"}


def is_video_source(path: str) -> bool:
    """動画（アニメーションGIF含む）かどうかを拡張子で判定する。"""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def build_video_timeline(
    material_paths: list[str],
    midi_path: str | None,
    bpm: int,
    base_note: int,
    track_assignments: dict[int, int] | None,
    material_base_notes: list[int] | None,
    material_regions: list[dict] | None,
    video_paths: list[str] | None,
    video_assignments: dict[int, int] | None,
    out_json: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    material_profiles: list[dict] | None = None,
) -> int:
    """音声と同じノート配置で、映像クリップ付きのタイムラインJSON(v2)を書き出す。

    各クリップは「どの映像を・どの位置から・どの長さで出すか」を持ち、
    render_video() がそのままMP4へ焼き込める形式にする。
    再生モードが "auto" の素材は解析して synth / raw を自動判定する。
    """
    # 音声生成と同じチョップ位置にするため、ここでもモードを確定させる。
    regions = resolve_region_modes(material_paths, material_regions, material_profiles)
    # 素材のデコードは重いので並列に読み込む（入力順は維持）。
    libraries = _parallel_map(
        lambda index: _load_material_library(material_paths[index], regions[index], False),
        range(len(material_paths)),
        workers=_audio_workers(),
    )
    if not libraries:
        raise ValueError("音声素材がありません。")

    notes = []
    tpq = 480
    if midi_path:
        notes, midi_bpm, tpq = midi_to_notes(midi_path)
        if bpm <= 0:
            bpm = midi_bpm
    if not notes:
        beat_ms = 60000 / max(1, bpm)
        count = max(1, int(len(libraries[0][0]) / (beat_ms / 2)))
        notes = [Note(base_note, i * 240, 120, 100) for i in range(count)]
        tpq = 480

    videos = [str(path) for path in (video_paths or [])]
    items = []
    for i, n in enumerate(notes):
        material_index = (track_assignments or {}).get(n.track, n.track % len(libraries))
        material_index = material_index % len(libraries)
        _, chunks = libraries[material_index]
        src_start, src_end = chunks[i % len(chunks)]
        video_source = None
        video_start_ms = 0
        if videos:
            video_index = (video_assignments or {}).get(material_index)
            if video_index is None:
                video_index = material_index % len(videos)
            video_index = int(video_index) % len(videos)
            video_source = os.path.abspath(videos[video_index])
            # 動画素材は音声と同じチョップ位置から見せる（画像は常に先頭から）。
            video_start_ms = src_start if is_video_source(video_source) else 0

        start_ms = int((n.start_tick / tpq) * (60000 / max(1, bpm)))
        duration_ms = max(35, int((n.duration_tick / tpq) * (60000 / max(1, bpm))))
        source_base = (material_base_notes or [base_note])[material_index]
        items.append({
            "type": "video_clip",
            "source": os.path.abspath(material_paths[material_index]),
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "source_start_ms": src_start,
            "source_end_ms": src_end,
            "video_source": video_source,
            "video_start_ms": video_start_ms,
            "midi_note": n.note,
            "pitch_shift_semitones": n.note - source_base,
            "track": n.track,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "version": 2,
            "bpm": bpm,
            "base_note": base_note,
            "clips": items,
            "template": {
                "transition": "cut",
                "scale_mode": "fit",
                "audio_linked": True,
                "width": width,
                "height": height,
                "fps": fps,
            },
        }, f, ensure_ascii=False, indent=2)
    return len(items)


def probe_duration_ms(path: str) -> int:
    """ffprobeでメディアの長さ(ms)を取得する。取得できない場合は0。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return max(0, int(float(result.stdout.strip()) * 1000))
    except Exception:
        return 0


def build_video_segments(clips: list[dict], total_ms: int, gap_mode: str = "hold") -> list[dict]:
    """クリップ列を、隙間のない映像セグメント列へ変換する。

    ノートとノートの間（無音区間）は gap_mode に従って埋める。
    - "hold": 直前の映像を続けて表示（先頭の隙間は最初の映像を使う）
    - "black": 黒画面
    """
    ordered = sorted((c for c in clips if c.get("video_source")), key=lambda c: int(c["start_ms"]))
    if not ordered:
        return []

    segments = []
    cursor = 0

    def fill_gap(length_ms, next_clip=None):
        if length_ms <= 0:
            return
        if gap_mode == "hold" and segments:
            # 直前の映像を、その続きから表示する。
            previous = segments[-1]
            segments.append({
                **previous,
                "source_start_ms": int(previous.get("source_start_ms", 0)) + int(previous.get("duration_ms", 0)),
                "duration_ms": length_ms,
            })
        elif gap_mode == "hold" and next_clip is not None:
            segments.append({
                "source": next_clip["video_source"],
                "source_start_ms": int(next_clip.get("video_start_ms", 0)),
                "duration_ms": length_ms,
                "is_image": not is_video_source(next_clip["video_source"]),
            })
        else:
            segments.append({"source": None, "source_start_ms": 0, "duration_ms": length_ms, "is_image": False})

    for clip in ordered:
        start_ms = max(0, int(clip["start_ms"]))
        duration_ms = max(1, int(clip["duration_ms"]))
        end_ms = min(total_ms, start_ms + duration_ms)
        if end_ms <= cursor:
            continue
        if start_ms > cursor:
            fill_gap(start_ms - cursor, clip)
            cursor = start_ms
        segments.append({
            "source": clip["video_source"],
            "source_start_ms": int(clip.get("video_start_ms", 0)),
            "duration_ms": end_ms - cursor,
            "is_image": not is_video_source(clip["video_source"]),
        })
        cursor = end_ms
        if cursor >= total_ms:
            break

    if cursor < total_ms:
        fill_gap(total_ms - cursor)
    return segments


def render_video(
    timeline_path: str,
    audio_path: str,
    out_mp4: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    gap_mode: str = "hold",
    progress_cb=None,
) -> int:
    """video_timeline.json と音声から、音声付きMP4を書き出す。

    各区間を h264/mpegts で書き出して concat し、音声と合成する。
    戻り値は書き出したセグメント数。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません。FFmpegをインストールしてPATHに追加してください。")

    data = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    clips = [c for c in data.get("clips", []) if c.get("video_source")]
    if not clips:
        raise ValueError("映像が割り当てられたクリップがありません。映像素材を登録・割り当てしてください。")

    audio_ms = probe_duration_ms(audio_path)
    if audio_ms <= 0:
        audio_ms = max(int(c["start_ms"]) + int(c["duration_ms"]) for c in clips) + 500
    segments = build_video_segments(clips, audio_ms, gap_mode)
    if not segments:
        raise ValueError("映像セグメントを作成できませんでした。")

    width, height, fps = max(16, int(width)), max(16, int(height)), max(1, int(fps))
    scale_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},format=yuv420p"
    )
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ytpmv_video_") as tmpdir:
        tmp = Path(tmpdir)
        total = len(segments)
        segment_paths = [tmp / f"seg_{i:05d}.ts" for i in range(total)]
        list_lines = [f"file '{path.as_posix()}'" for path in segment_paths]

        def encode(index: int) -> int:
            _encode_video_segment(ffmpeg, segments[index], segment_paths[index],
                                  scale_filter, width, height, fps)
            return index

        # 各区間は独立なので、ffmpegを複数プロセスで並列に走らせる。
        workers = min(_video_workers(), max(1, total))
        completed = 0
        lock = threading.Lock()
        if workers <= 1:
            for index in range(total):
                encode(index)
                completed += 1
                if progress_cb:
                    progress_cb(0.05 + 0.88 * completed / total,
                                f"映像を書き出し中… ({completed}/{total})")
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = [pool.submit(encode, i) for i in range(total)]
                for future in as_completed(futures):
                    future.result()
                    with lock:
                        completed += 1
                        value = 0.05 + 0.88 * completed / total
                    if progress_cb:
                        progress_cb(value, f"映像を書き出し中… ({completed}/{total})")
            except BaseException:
                # 失敗時は未実行ぶんのエンコードを打ち切って、すぐエラーを返す。
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown()

        list_path = tmp / "concat.txt"
        list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")
        if progress_cb:
            progress_cb(0.95, "音声と合成中…")
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
             "-movflags", "+faststart", str(out_mp4)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpegの合成に失敗しました: {result.stderr.strip()[-500:]}")

    if progress_cb:
        progress_cb(1.0, "動画の書き出し完了")
    return len(segments)


def _encode_video_segment(ffmpeg, segment, out_path, scale_filter, width, height, fps):
    """1区間分の映像（音声なし）を mpegts で書き出す。"""
    duration = max(0.03, segment["duration_ms"] / 1000)
    output = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-r", str(fps), "-f", "mpegts", str(out_path)]
    if segment.get("source") is None:
        # 黒画面（映像が無い区間）
        command = [ffmpeg, "-y", "-f", "lavfi",
                   "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration:.3f}",
                   "-vf", "format=yuv420p", "-t", f"{duration:.3f}"] + output
    elif segment["is_image"]:
        command = [ffmpeg, "-y", "-loop", "1", "-i", str(segment["source"]),
                   "-t", f"{duration:.3f}", "-vf", scale_filter] + output
    else:
        command = [ffmpeg, "-y", "-stream_loop", "-1",
                   "-ss", f"{segment['source_start_ms'] / 1000:.3f}",
                   "-i", str(segment["source"]),
                   "-t", f"{duration:.3f}", "-vf", scale_filter] + output
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        name = Path(str(segment.get("source"))).name
        raise RuntimeError(f"映像の書き出しに失敗しました({name}): {result.stderr.strip()[-300:]}")


class App:
    def __init__(self, page: ft.Page):
        ensure_project_directories()
        self.page = page
        page.title = "音MAD Auto Maker"
        page.window.width = 1050
        page.window.height = 760
        page.padding = 18

        self.materials = []
        self.material_base_notes = []
        self.material_regions = []
        self.material_profiles = []
        self._skipped_tracks = set()
        self.videos = []
        self.video_assignments = {}
        self.material_list = ft.Column()
        self.video_list = ft.Column([ft.Text("映像素材が未登録です（画像または動画を追加してください）")])
        self.video_assignment_list = ft.Column()
        self.track_assignments = {}
        self.track_volumes = {}
        self.assignment_list = ft.Column()
        self.midi = ft.TextField(label="MIDI (任意)", expand=True)
        self.bgm = ft.TextField(label="BGM (任意)", expand=True,
                                on_change=lambda e: self.rebalance_volumes())
        self.bgm_volume = ft.TextField(label="BGM音量dB", value="0", width=130,
                                       on_change=lambda e: self.rebalance_volumes())
        self.video_size = ft.TextField(label="動画サイズ", value="1280x720", width=150)
        self.video_fps = ft.TextField(label="FPS", value="30", width=90)
        self.output_dir = ft.TextField(label="成果物フォルダ", value=str(OUTPUTS_DIR), expand=True)
        self.bpm = ft.TextField(label="BPM", value="120", width=130)
        self.base_note = ft.TextField(label="素材の基準音 (MIDI番号)", value="60", width=180)
        # 並列数（空欄で自動）。値を変えると次の生成から反映される。
        self.workers = ft.TextField(label="並列数", value=str(_render_workers()), width=100,
                                    tooltip="音声の読み込み・解析・ノート描画の並列数（空欄で自動 / 1で逐次）")
        self.video_workers = ft.TextField(label="動画並列数", value=str(_video_workers()), width=120,
                                          tooltip="動画セグメントのエンコード並列数（空欄で自動 / 1で逐次）")
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
            current = self.material_regions[i].get("mode") or "auto"
            label = "再生"
            if current == "auto":
                resolved = self.recommended_mode(i)
                label = f"再生（自動: {MODE_LABELS.get(resolved, resolved)}）"
            mode = ft.Dropdown(
                label=label, value=current, width=190,
                options=[ft.dropdown.Option(key, text) for key, text in MODE_LABELS.items()])
            mode.on_change = lambda e, n=i: self.set_region(n, "mode", e.control.value)
            row.controls.insert(5, mode)
        self.page.update()

    def recommended_mode(self, index):
        """素材の推奨再生モード（synth / raw）を返す。未解析なら解析する。"""
        profile = self.material_profiles[index] if index < len(self.material_profiles) else None
        if not profile or "recommended_mode" not in profile:
            try:
                profile = analyze_material(self.materials[index])
                if index < len(self.material_profiles):
                    self.material_profiles[index] = profile
            except Exception:
                return "synth"
        return profile.get("recommended_mode", "synth")

    def set_material_base_note(self, index, value):
        try:
            self.material_base_notes[index] = int(value)
        except ValueError:
            pass

    def auto_base_note(self, index):
        try:
            profile = analyze_material(self.materials[index])
            self.material_base_notes[index] = profile["base_note"]
            self.material_profiles[index] = profile
            self.refresh_materials()
            self.add_log(
                f"基準音を自動判定: {Path(self.materials[index]).name} → MIDI {profile['base_note']}"
                f"（明るさ {profile['brightness']:.0f}Hz）"
            )
            self.add_log(
                f"  再生モードを自動判定: {MODE_LABELS.get(profile.get('recommended_mode', 'synth'), '')}"
                f" — {profile.get('recommended_reason', '')}"
            )
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
            outdir = Path(self.output_dir.value.strip() or OUTPUTS_DIR)
            outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / f"{timestamp_label()}_{Path(path).stem}_selected_{index}.wav"
            await asyncio.to_thread(audio[start:end].export, str(out), format="wav")
            return out
        except Exception as ex:
            self.add_log(f"範囲生成エラー: {type(ex).__name__}: {ex}")
            return None

    def stop_preview(self):
        """起動中の試聴プレイヤーを停止する。"""
        if hasattr(self, "preview_process") and self.preview_process.poll() is None:
            self.preview_process.terminate()

    def open_player(self, path, volume=0.3):
        """生成した音声をOSのプレイヤーで開く。

        Windows / macOS / Linux の既定アプリを優先し、使えない環境では
        ffplay（ウィンドウ表示）→ paplay / aplay の順にフォールバックする。
        """
        try:
            path = str(path)
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
                self.add_log(f"プレイヤーで開きました: {path}")
                return
            opener = None
            if sys.platform == "darwin":
                opener = "open"
            elif shutil.which("xdg-open") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                opener = "xdg-open"
            if opener:
                self.preview_process = subprocess.Popen([opener, path])
                self.add_log(f"プレイヤーで開きました: {path}")
                return

            # 既定プレイヤーが使えない環境では、ウィンドウを開ける ffplay を使う。
            player = next((p for p in ("ffplay", "paplay", "aplay") if shutil.which(p)), None)
            if not player:
                raise RuntimeError("既定のプレイヤーも ffplay / paplay / aplay も見つかりません")
            self.stop_preview()
            command = ([player, "-autoexit", "-loglevel", "warning", "-af", f"volume={volume}", path]
                       if player == "ffplay" else [player, path])
            self.preview_process = subprocess.Popen(command)
            hint = "（space=一時停止 / q=終了）" if player == "ffplay" else ""
            self.add_log(f"{player} で再生中{hint}: {path}")
        except Exception as ex:
            self.add_log(f"試聴エラー: {type(ex).__name__}: {ex}")

    async def preview_region(self, index):
        out = await self.export_region(index)
        if not out:
            return
        self.open_player(out)

    async def preview_track(self, track_index):
        """指定MIDIトラックだけを生成して試聴する。"""
        try:
            if not self.materials:
                raise ValueError("音声素材を指定してください。")
            self.add_log(f"並列数: {self.apply_worker_settings()}")
            midi = self.midi.value.strip() or None
            outdir = Path(self.output_dir.value.strip() or OUTPUTS_DIR)
            outdir.mkdir(parents=True, exist_ok=True)
            out = str(outdir / f"{timestamp_label()}_track_{track_index + 1}_preview.wav")

            self.progress.value = 0
            self.add_log(f"トラック {track_index + 1} を生成中…")
            self.page.update()

            cb = self.progress_callback("生成中…")

            count, duration = await asyncio.to_thread(
                make_song, self.materials, midi, None,
                int(float(self.bpm.value)), int(self.base_note.value),
                dict(self.track_assignments), list(self.material_base_notes),
                dict(self.track_volumes), list(self.material_regions),
                out, None, cb, track_index, 0.0, list(self.material_profiles),
            )
            self.progress.value = 1
            self.status.value = f"トラック {track_index + 1} 生成完了"
            self.add_log(f"トラック {track_index + 1}: {count}ノート / {duration / 1000:.1f}秒 → {out}")
            self.open_player(out, volume=0.4)
        except Exception as ex:
            self.add_log(f"トラック試聴エラー: {type(ex).__name__}: {ex}")

    async def register_region(self, index):
        out = await self.export_region(index)
        if not out:
            return
        self.materials.append(str(out))
        try:
            profile = await asyncio.to_thread(analyze_material, str(out))
        except Exception as ex:
            self.add_log(f"切り出し範囲の解析エラー: {type(ex).__name__}: {ex}")
            profile = dict(self.material_profiles[index]) if index < len(self.material_profiles) else {}
        self.material_base_notes.append(profile.get("base_note", self.material_base_notes[index]))
        self.material_profiles.append(profile)
        self.material_regions.append({"start": 0, "end": profile.get("duration_ms", 0), "mode": "auto"})
        self.refresh_materials()
        self.refresh_assignments()
        self.add_log(
            f"切り出した範囲を別素材として登録: {out.name}"
            f"（基準音 MIDI {self.material_base_notes[-1]} / "
            f"再生モード {MODE_LABELS.get(profile.get('recommended_mode', 'synth'), '')}）"
        )

    def remove_material(self, path):
        index = self.materials.index(path)
        self.materials.remove(path)
        self.material_base_notes.pop(index)
        self.material_regions.pop(index)
        if index < len(self.material_profiles):
            self.material_profiles.pop(index)
        self.refresh_assignments()
        self.refresh_materials()
        self.refresh_videos()

    async def pick_video(self):
        try:
            extensions = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
            result = await self.file_picker.pick_files(
                allow_multiple=True, file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=extensions)
            if not result:
                return
            for item in result:
                path = item.path or item.name
                self.videos.append(path)
                kind = "動画" if is_video_source(path) else "画像"
                self.add_log(f"映像素材を追加: {Path(path).name}（{kind}）")
            self.refresh_videos()
        except Exception as ex:
            self.add_log(f"映像選択エラー: {type(ex).__name__}: {ex}")

    def remove_video(self, path):
        index = self.videos.index(path)
        self.videos.remove(path)
        # 削除した位置より後の割り当てを1つずらす。
        self.video_assignments = {
            material: (video - 1 if video > index else video)
            for material, video in self.video_assignments.items() if video != index
        }
        self.refresh_videos()

    def set_video_assignment(self, material_index, value):
        try:
            self.video_assignments[material_index] = int(value)
        except (TypeError, ValueError):
            self.video_assignments.pop(material_index, None)

    def resolved_video_assignments(self):
        """素材ごとに使う映像indexを解決する（未設定はファイル名一致→順番）。"""
        resolved = {}
        if not self.videos:
            return resolved
        for i, path in enumerate(self.materials):
            index = self.video_assignments.get(i)
            if index is None:
                stem = Path(path).stem.lower()
                index = next(
                    (j for j, video in enumerate(self.videos)
                     if Path(video).stem.lower() in stem or stem in Path(video).stem.lower()),
                    i % len(self.videos),
                )
            resolved[i] = int(index) % len(self.videos)
        return resolved

    def refresh_videos(self):
        try:
            if not self.videos:
                self.video_list.controls = [ft.Text("映像素材が未登録です（画像または動画を追加してください）")]
                self.video_assignment_list.controls = []
                self.page.update()
                return
            names = [Path(path).name for path in self.videos]
            self.video_list.controls = [
                ft.Row([
                    ft.Text(f"{'🎞' if is_video_source(path) else '🖼'} {Path(path).name}", width=420),
                    ft.Button("削除", on_click=lambda e, p=path: self.remove_video(p)),
                ])
                for path in self.videos
            ]
            resolved = self.resolved_video_assignments()
            rows = []
            for i, path in enumerate(self.materials):
                dropdown = ft.Dropdown(
                    value=str(resolved.get(i, 0)),
                    options=[ft.dropdown.Option(str(j), name) for j, name in enumerate(names)],
                    expand=True,
                )
                dropdown.on_change = lambda e, n=i: self.set_video_assignment(n, e.control.value)
                rows.append(ft.Row([ft.Text(Path(path).name, width=220), dropdown]))
            self.video_assignment_list.controls = rows
            self.page.update()
        except Exception as ex:
            self.video_list.controls = []
            self.video_assignment_list.controls = []
            self.add_log(f"映像一覧の更新エラー: {type(ex).__name__}: {ex}")

    def video_settings(self):
        """動画サイズとFPSの入力を解釈する（不正値は既定値）。"""
        width, height = 1280, 720
        text = (self.video_size.value or "").lower().replace("×", "x").replace("*", "x")
        try:
            left, right = text.split("x")
            width, height = int(left), int(right)
        except ValueError:
            self.add_log(f"動画サイズの値が不正なため1280x720で続行します: {self.video_size.value!r}")
        try:
            fps = int(float(self.video_fps.value or 30))
        except ValueError:
            fps = 30
            self.add_log(f"FPSの値が不正なため30で続行します: {self.video_fps.value!r}")
        return max(16, width), max(16, height), max(1, fps)

    def apply_worker_settings(self):
        """並列数の入力を環境変数へ反映する（次の生成から有効）。

        空欄・0以下・不正値は「自動（既定値）」として扱う。
        """
        summary = []
        for field, env, label in ((self.workers, "YTPMV_WORKERS", "音声"),
                                  (self.video_workers, "YTPMV_VIDEO_WORKERS", "動画")):
            text = (field.value or "").strip()
            value = 0
            if text:
                try:
                    value = int(float(text))
                except ValueError:
                    self.add_log(f"{label}の並列数の値が不正なため自動設定で続行します: {text!r}")
                    value = 0
            if value > 0:
                os.environ[env] = str(value)
                summary.append(f"{label} {value}")
            else:
                # 環境変数を消して自動（コア数に応じた既定値）へ戻す。
                os.environ.pop(env, None)
                auto = _render_workers() if env == "YTPMV_WORKERS" else _video_workers()
                summary.append(f"{label} {auto}(自動)")
        return " / ".join(summary)

    def refresh_assignments(self):
        if not self.midi.value.strip() or not self.materials:
            self.assignment_list.controls = []
            self._announce_skipped_tracks(set())
            return
        skipped = set()
        try:
            mid = mido.MidiFile(self.midi.value.strip())
            notes, _, _ = midi_to_notes(self.midi.value.strip())
            counts = {}
            for note in notes:
                counts[note.track] = counts.get(note.track, 0) + 1
            controls = []
            for track_index, track in enumerate(mid.tracks):
                count = counts.get(track_index, 0)
                if count == 0:
                    # ノートを持たないトラックは割り当て対象がないため一覧から除外する。
                    # MIDIファイル自体は変更しないので、読み込み直せば元に戻る。
                    skipped.add(track_index)
                    continue
                name = track.name or f"トラック {track_index + 1}"
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
                preview = ft.Button("🎧 試聴", on_click=lambda e, t=track_index: self.page.run_task(self.preview_track, t))
                controls.append(ft.Row([ft.Text(f"{name}（{count}音）", width=220), dropdown, volume, preview]))
            self.assignment_list.controls = controls
        except Exception:
            self.assignment_list.controls = []
            skipped = set()
        self._announce_skipped_tracks(skipped)

    def _announce_skipped_tracks(self, skipped):
        """ノートのないトラックの除外内容が変わったときだけログへ出す。"""
        if getattr(self, "_skipped_tracks", None) == skipped:
            return
        self._skipped_tracks = set(skipped)
        if skipped:
            numbers = ", ".join(str(i + 1) for i in sorted(skipped))
            self.add_log(f"ノートのないトラック {numbers} は割り当て一覧から除外しました")

    def set_track_assignment(self, track, material_index):
        self.track_assignments[track] = int(material_index)

    def set_track_volume(self, track, value):
        try:
            self.track_volumes[track] = float(value)
        except ValueError:
            pass

    def project_data(self):
        return {"materials": self.materials, "material_base_notes": self.material_base_notes,
                "material_regions": self.material_regions, "material_profiles": self.material_profiles,
                "track_assignments": self.track_assignments,
                "track_volumes": self.track_volumes, "midi": self.midi.value, "bgm": self.bgm.value,
                "bgm_volume": self.bgm_volume.value,
                "videos": self.videos, "video_assignments": self.video_assignments,
                "video_size": self.video_size.value, "video_fps": self.video_fps.value,
                "workers": self.workers.value, "video_workers": self.video_workers.value,
                "bpm": self.bpm.value, "base_note": self.base_note.value, "output_dir": self.output_dir.value}

    def save_project(self, e):
        path = Path(self.output_dir.value.strip() or OUTPUTS_DIR) / "otomad_project.json"
        path.write_text(json.dumps(self.project_data(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.add_log(f"プロジェクト保存: {path}")

    def load_project(self, e):
        path = Path(self.output_dir.value.strip() or OUTPUTS_DIR) / "otomad_project.json"
        if not path.exists():
            self.add_log(f"プロジェクトがありません: {path}")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.materials = data.get("materials", [])
        self.material_base_notes = data.get("material_base_notes", [60] * len(self.materials))
        self.material_regions = data.get("material_regions", [{"start": 0, "end": 0, "mode": "auto"} for _ in self.materials])
        profiles = data.get("material_profiles") or []
        base_notes = list(self.material_base_notes)
        self.material_base_notes = [
            base_notes[i] if i < len(base_notes) else 60 for i in range(len(self.materials))
        ]
        self.material_profiles = [
            dict(profiles[i]) if i < len(profiles) else {"base_note": self.material_base_notes[i], "brightness": 440.0}
            for i in range(len(self.materials))
        ]
        self.track_assignments = {int(k): int(v) for k, v in data.get("track_assignments", {}).items()}
        self.track_volumes = {int(k): float(v) for k, v in data.get("track_volumes", {}).items()}
        self.videos = list(data.get("videos", []))
        self.video_assignments = {int(k): int(v) for k, v in data.get("video_assignments", {}).items()}
        for field in (self.midi, self.bgm, self.bgm_volume, self.bpm, self.base_note, self.output_dir,
                      self.video_size, self.video_fps, self.workers, self.video_workers):
            key = {self.midi: "midi", self.bgm: "bgm", self.bgm_volume: "bgm_volume", self.bpm: "bpm",
                   self.base_note: "base_note", self.output_dir: "output_dir",
                   self.video_size: "video_size", self.video_fps: "video_fps",
                   self.workers: "workers", self.video_workers: "video_workers"}[field]
            value = data.get(key, field.value)
            field.value = "" if value is None else str(value)
        self.refresh_materials(); self.refresh_assignments(); self.refresh_videos()
        self.add_log(f"プロジェクト読み込み: {path}")

    def auto_assign_tracks(self, e=None):
        """素材をトラックごとに自動割り当てし、音量バランスも自動調整する。"""
        if not self.midi.value.strip() or not self.materials:
            self.add_log("自動割り当てには素材とMIDIが必要です")
            return
        try:
            midi_path = self.midi.value.strip()
            mid = mido.MidiFile(midi_path)
            notes, midi_bpm, tpq = midi_to_notes(midi_path)
            if not notes:
                self.add_log("MIDIにノートがないため自動割り当てできません")
                return

            track_names = {i: (track.name or "") for i, track in enumerate(mid.tracks)}
            profiles = []
            for i, path in enumerate(self.materials):
                profile = dict(self.material_profiles[i]) if i < len(self.material_profiles) else {}
                profile["path"] = path
                profile["base_note"] = self.material_base_notes[i]
                profiles.append(profile)

            assignments = auto_assign_materials(notes, profiles, track_names)
            for track, material_index in assignments.items():
                self.track_assignments[track] = material_index

            bpm = int(float(self.bpm.value))
            if bpm <= 0:
                bpm = midi_bpm
            has_bgm = bool(self.bgm.value.strip())
            volumes = auto_track_volumes(notes, bpm, tpq, self._bgm_volume(), has_bgm=has_bgm)
            self.track_volumes.update(volumes)

            self.refresh_assignments()
            for track in sorted(assignments):
                name = track_names.get(track) or f"トラック {track + 1}"
                material_index = assignments[track]
                self.add_log(
                    f"  {name} → {Path(self.materials[material_index]).name}"
                    f"（基準音 {profiles[material_index]['base_note']} / 音量 {volumes.get(track, 0):+.1f}dB）"
                )
            self.add_log(f"自動割り当て＋バランス調整: {len(assignments)}トラック")
        except Exception as ex:
            self.add_log(f"自動割り当てエラー: {type(ex).__name__}: {ex}")

    def rebalance_volumes(self, e=None):
        """MIDIとBGM設定を元にトラック音量だけを再計算してUIへ反映する。

        BGM音量フィールドやBGMパスの変更時に呼ばれ、素材の再割り当ては行わない。
        MIDI・素材が未設定の場合は何もしない。
        """
        if not self.midi.value.strip() or not self.materials:
            return
        try:
            midi_path = self.midi.value.strip()
            notes, midi_bpm, tpq = midi_to_notes(midi_path)
            if not notes:
                return
            bpm = int(float(self.bpm.value or 0))
            if bpm <= 0:
                bpm = midi_bpm
            has_bgm = bool(self.bgm.value.strip())
            volumes = auto_track_volumes(notes, bpm, tpq, self._bgm_volume(), has_bgm=has_bgm)
            self.track_volumes.update(volumes)
            self.refresh_assignments()
            bgm_label = f"BGM音量 {self._bgm_volume():+.1f}dB" if has_bgm else "BGMなし"
            self.add_log(f"音量を再バランス（{bgm_label}）: " + ", ".join(
                f"T{t+1} {v:+.1f}dB" for t, v in sorted(volumes.items())
            ))
        except Exception as ex:
            self.add_log(f"音量再バランスエラー: {type(ex).__name__}: {ex}")

    async def add_material(self, e):
        await self.pick_file(None, ["wav", "mp3", "flac", "ogg"])

    async def pick_material(self):
        try:
            result = await self.file_picker.pick_files(allow_multiple=True, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["wav", "mp3", "flac", "ogg"])
            if result:
                paths = [item.path or item.name for item in result]
                # 読み込み時に長さと基準音（高さ）を自動判定する。
                # 解析は重いので、まとめて並列実行する（失敗は例外として返る）。
                profiles = await asyncio.to_thread(analyze_materials, paths)
                for path, profile in zip(paths, profiles):
                    if isinstance(profile, Exception):
                        self.add_log(f"素材解析エラー: {Path(path).name}: {type(profile).__name__}: {profile}")
                        profile = {"duration_ms": 0, "base_note": 60, "brightness": 440.0}
                    self.materials.append(path)
                    self.material_base_notes.append(profile["base_note"])
                    self.material_profiles.append(profile)
                    self.material_regions.append(
                        {"start": 0, "end": profile.get("duration_ms", 0), "mode": "auto"}
                    )
                    self.add_log(
                        f"素材を追加: {Path(path).name}"
                        f"（基準音 MIDI {profile['base_note']} / 明るさ {profile.get('brightness', 0):.0f}Hz）"
                    )
                    self.add_log(
                        f"  再生モードを自動判定: {MODE_LABELS.get(profile.get('recommended_mode', 'synth'), '')}"
                        f" — {profile.get('recommended_reason', '')}"
                    )
                self.refresh_assignments()
                self.refresh_materials()
                self.refresh_videos()
                if self.midi.value.strip():
                    # 素材が揃ったら割り当てとバランスを自動で決め直す。
                    self.auto_assign_tracks()
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
                    self.track_volumes = {}
                    self.refresh_assignments()
                    self.page.update()
                    if self.materials:
                        # MIDIを読み込んだら割り当てとバランスを自動で決める。
                        self.auto_assign_tracks()
                    return
                if field is self.bgm:
                    # BGMファイルを選択したらトラック音量を再バランス。
                    self.rebalance_volumes()
                    return
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

    def progress_callback(self, message="制作中…"):
        """ワーカースレッドから呼んでも安全な進捗コールバックを作る。

        Fletのコントロールはイベントループのスレッドからのみ操作する必要がある。
        asyncio.to_thread で動く処理から直接 page.update() を呼ぶと
        「dictionary changed size during iteration」などで落ちるため、
        進捗の反映は call_soon_threadsafe でイベントループへ委譲する。
        """
        loop = asyncio.get_running_loop()

        def apply(value, text):
            try:
                self.progress.value = value
                self.status.value = f"{text} {value * 100:.0f}%"
                self.page.update()
            except Exception:
                # 画面が閉じられた後などは無視する。
                pass

        def cb(value, text=message):
            loop.call_soon_threadsafe(apply, value, text)

        return cb

    async def _generate_audio(self, cb):
        """音声を生成して (wav, mp3, ノート数, 長さms) を返す。"""
        if not self.materials:
            raise ValueError("音声素材を指定してください。")
        self.add_log(f"並列数: {self.apply_worker_settings()}")
        midi = self.midi.value.strip() or None
        bgm = self.bgm.value.strip() or None
        bgm_volume = self._bgm_volume()
        bpm = int(float(self.bpm.value))
        base_note = int(self.base_note.value)
        outdir = Path(self.output_dir.value.strip() or OUTPUTS_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp_label()
        wav = str(outdir / f"{stamp}_otomad_result.wav")
        mp3 = str(outdir / f"{stamp}_otomad_result.mp3")

        if bgm:
            self.add_log(f"音声を生成中…（BGM音量 {bgm_volume:+.1f}dB）")
        else:
            self.add_log("音声を生成中…")
        self.page.update()

        count, duration = await asyncio.to_thread(
            make_song, self.materials, midi, bgm, bpm, base_note, dict(self.track_assignments),
            self.material_base_notes, self.track_volumes, self.material_regions,
            wav, mp3, cb, None, bgm_volume, list(self.material_profiles)
        )
        return wav, mp3, count, duration

    def _bgm_volume(self):
        try:
            return float(self.bgm_volume.value or 0)
        except ValueError:
            self.add_log(f"BGM音量の値が不正なため0dBで続行します: {self.bgm_volume.value!r}")
            return 0.0

    async def run_audio(self, e):
        try:
            self.progress.value = 0
            self.page.update()
            cb = self.progress_callback("制作中…")
            wav, mp3, count, duration = await self._generate_audio(cb)
            self.progress.value = 1
            self.status.value = "制作完了"
            self.add_log(f"完了: {count}ノート / {duration/1000:.1f}秒")
            self.add_log(f"WAV: {wav}")
            self.add_log(f"MP3: {mp3}")
        except Exception as ex:
            self.add_log(f"エラー: {type(ex).__name__}: {ex}")

    def _video_assignments_or_raise(self):
        if not self.videos:
            raise ValueError("映像素材を追加してください（画像または動画）。")
        assignments = self.resolved_video_assignments()
        if not assignments:
            raise ValueError("映像の割り当てを設定してください。")
        return assignments

    async def run_video(self, e):
        """音声を生成し、映像をFFmpegで合成してMP4を書き出す。"""
        try:
            if not self.midi.value.strip():
                raise ValueError("動画生成にはMIDIが必要です。")
            video_assignments = self._video_assignments_or_raise()
            width, height, fps = self.video_settings()
            outdir = Path(self.output_dir.value.strip() or OUTPUTS_DIR)
            outdir.mkdir(parents=True, exist_ok=True)

            self.progress.value = 0
            self.page.update()

            cb = self.progress_callback("制作中…")
            wav, _mp3, count, duration = await self._generate_audio(cb)
            self.add_log(f"音声生成完了: {count}ノート / {duration/1000:.1f}秒")

            stamp = Path(wav).name.split("_otomad_result", 1)[0]
            timeline = str(outdir / f"{stamp}_video_timeline.json")
            clip_count = await asyncio.to_thread(
                build_video_timeline, self.materials, self.midi.value.strip(),
                int(float(self.bpm.value)), int(self.base_note.value),
                dict(self.track_assignments), self.material_base_notes, self.material_regions,
                self.videos, video_assignments, timeline, width, height, fps,
                list(self.material_profiles)
            )
            self.add_log(f"タイムライン: {clip_count}クリップ → {timeline}")

            mp4 = str(outdir / f"{stamp}_otomad_result.mp4")
            video_cb = self.progress_callback("映像を書き出し中…")
            segments = await asyncio.to_thread(
                render_video, timeline, wav, mp4, width, height, fps, "hold", video_cb
            )
            self.progress.value = 1
            self.status.value = "動画生成完了"
            self.add_log(f"動画生成完了: {segments}セグメント / {width}x{height} {fps}fps")
            self.add_log(f"MP4: {mp4}")
        except Exception as ex:
            self.add_log(f"動画生成エラー: {type(ex).__name__}: {ex}")

    async def run_video_template(self, e):
        try:
            if not self.materials or not self.midi.value.strip():
                raise ValueError("動画テンプレートには素材とMIDIが必要です。")
            self.add_log(f"並列数: {self.apply_worker_settings()}")
            outdir = Path(self.output_dir.value.strip() or OUTPUTS_DIR)
            outdir.mkdir(parents=True, exist_ok=True)
            bpm = int(float(self.bpm.value))
            base = int(self.base_note.value)
            width, height, fps = self.video_settings()
            out = str(outdir / f"{timestamp_label()}_video_timeline.json")
            count = await asyncio.to_thread(
                build_video_timeline, self.materials, self.midi.value.strip(), bpm, base,
                dict(self.track_assignments), self.material_base_notes, self.material_regions,
                self.videos, self.resolved_video_assignments(), out, width, height, fps,
                list(self.material_profiles)
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
            ft.Text("素材の割り当て（🎧 試聴でトラック単体を生成）", size=18, weight=ft.FontWeight.BOLD),
            self.assignment_list,
            ft.Row([ft.Button("自動割り当て＋バランス", on_click=self.auto_assign_tracks),
                    ft.Button("プロジェクト保存", on_click=self.save_project),
                    ft.Button("プロジェクト読み込み", on_click=self.load_project)]),
            ft.Row([self.bgm, self.bgm_volume, pick_bgm]),
            ft.Row([self.bpm, self.base_note]),
            ft.Row([self.output_dir, pick_dir]),
            ft.Row([self.workers, self.video_workers,
                    ft.Text("並列数（空欄で自動 / 1で逐次）", color=ft.Colors.GREY_700)]),
            ft.Text("映像（画像/動画）", size=18, weight=ft.FontWeight.BOLD),
            ft.Button("＋ 画像/動画を追加", on_click=lambda e: self.page.run_task(self.pick_video)),
            self.video_list,
            ft.Text("映像の割り当て（音声素材 → 映像）"),
            self.video_assignment_list,
            ft.Row([self.video_size, self.video_fps,
                    ft.Button("🎬 動画を生成 (MP4)", on_click=lambda e: self.page.run_task(self.run_video, e)),
                    ft.Button("🎬 動画テンプレJSON", on_click=lambda e: self.page.run_task(self.run_video_template, e))]),
            ft.Divider(),
            ft.Row([
                ft.Button("▶ 自動制作 (WAV/MP3)", on_click=lambda e: self.page.run_task(self.run_audio, e)),
            ]),
            self.progress,
            self.status,
            self.log,
        ], expand=True, scroll=ft.ScrollMode.AUTO)


def main(page: ft.Page):
    page.add(App(page).build())


if __name__ == "__main__":
    ft.run(main)
