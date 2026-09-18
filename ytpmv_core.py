"""GUI/CLI共通のYTPMVmaker公開API。"""

from main import (
    analyze_material,
    analyze_materials,
    auto_assign_materials,
    auto_track_volumes,
    build_video_timeline,
    estimate_base_note,
    is_video_source,
    make_song,
    midi_to_notes,
    recommend_play_mode,
    render_video,
    resolve_region_modes,
)

__all__ = [
    "analyze_material",
    "analyze_materials",
    "auto_assign_materials",
    "auto_track_volumes",
    "build_video_timeline",
    "estimate_base_note",
    "is_video_source",
    "make_song",
    "midi_to_notes",
    "recommend_play_mode",
    "render_video",
    "resolve_region_modes",
]
