# 音MAD Auto Maker

Python + Flet で作る音MAD自動制作ソフトのプロトタイプです。

## 主な機能
- WAV/MP3素材の読み込み
- 無音区間を利用した自動チョップ
- BPMに合わせたグリッド生成
- MIDI（SMF）読み込み → ノート列化
- 素材をMIDIの音階に合わせてピッチシフトして配置
- BGMとの簡易ミックス
- 音量・パンの自動調整
- WAV / MP3 書き出し
- Flet GUI
- 動画テンプレート用のタイムラインJSON出力
- 複数素材・MIDIトラック別の素材割り当てと音量設定
- 素材ごとの基準音、切り出し範囲、シンセ化／そのまま再生
- CLI実行とGUIプロジェクト保存

## 必要環境
- Python 3.14+
- FFmpeg（MP3入出力に必要）
- MIDI解析用に `mido`
- 音声処理用に `pydub`
- GUI用に `flet`

## 起動
```bash
pip install -r requirements.txt
python main.py
```

## CLI

素材を複数指定して生成できます。

```bash
uv run python cli.py -m vocal.wav -m effect.wav --midi melody.mid --bgm bgm.wav --output-dir out
```

GUIで保存した `otomad_project.json` を使う場合：

```bash
uv run python cli.py --project otomad_project.json
```

音声素材・MIDI・生成物は `.gitignore` 対象です。公開前に個人情報や素材の権利関係を確認してください。

FFmpeg が PATH にない場合は、FFmpeg をインストールして PATH に追加してください。

## MIDIについて
MIDIノートは標準のMIDIノート番号（60=C4）として扱います。
MIDIの各ノートの長さをグリッド長として使い、素材の基準音からの半音差でピッチを変更します。

## 動画
「動画テンプレートを書き出す」で `video_timeline.json` を生成します。
これは動画編集ソフトや独自レンダラーから利用できる、素材・開始時刻・長さ・音程をまとめたテンプレートデータです。
