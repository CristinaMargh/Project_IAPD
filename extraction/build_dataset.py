"""Clean song metadata and build descriptive features for Phase 1."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data/raw/songs.csv"
FINAL_CSV = PROJECT_ROOT / "data/processed/music_popularity_dataset.csv"
COVERAGE_CSV = PROJECT_ROOT / "data/processed/feature_coverage.csv"
REPORT_JSON = PROJECT_ROOT / "data/processed/dataset_report.json"

GENRE_PATTERNS = {
    "genre_pop": r"pop|europop|teen pop|synthpop",
    "genre_rock": r"rock|metal|punk|grunge|indie rock",
    "genre_hiphop": r"hip[ -]?hop|rap|trap|drill",
    "genre_electronic": r"electro|edm|dance|house|techno|trance|dubstep|drum and bass|future bass",
    "genre_latin": r"latin|reggaeton|bachata|salsa|merengue|cumbia",
    "genre_rnb": r"r&b|rhythm and blues|soul|funk",
    "genre_country": r"country|bluegrass|americana",
    "genre_kpop": r"k[ -]?pop|korean pop",
}

CANDIDATE_FEATURES = [
    "release_year", "release_decade", "genre_tag_count", "genre_missing",
    "artist_missing", *GENRE_PATTERNS.keys(),
    "title_word_count", "title_character_count", "title_has_parentheses",
    "title_has_remix", "artist_word_count", "artist_character_count",
]

PREDICTOR_COLUMNS = CANDIDATE_FEATURES

ENRICHMENT_FILES = {
    "musicbrainz_metadata.csv": [
        "song_duration_seconds", "artist_country", "artist_type", "artist_count",
        "is_collaboration", "musicbrainz_release_month", "musicbrainz_status",
    ],
    "youtube_metadata.csv": [
        "youtube_views", "youtube_likes", "youtube_comments",
        "youtube_video_duration_seconds", "youtube_upload_date",
        "youtube_collected_at_utc", "youtube_channel",
    ],
    "audio_features.csv": [
        "tempo_bpm", "audio_rms_energy", "musical_key", "musical_mode",
    ],
}

NUMERIC_ENRICHMENTS = [
    "song_duration_seconds", "artist_count", "is_collaboration", "musicbrainz_release_month",
    "youtube_views", "youtube_likes", "youtube_comments",
    "youtube_video_duration_seconds", "tempo_bpm", "audio_rms_energy",
]

MISSING_MARKERS = {"", "none", "nan", "null", "n/a", "na", "unknown", "not available"}


def normalize_missing_text(series):
    """Convert placeholder strings into real missing values."""
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.casefold().isin(MISSING_MARKERS))


def create_metadata_features(songs):
    """Clean source labels and derive descriptive predictors without inventing values."""
    songs = songs.copy()
    for column in ("genre", "genre_tags", "song_title", "artist"):
        if column not in songs:
            songs[column] = pd.NA
        songs[column] = normalize_missing_text(songs[column])

    songs["genre_missing"] = songs["genre_tags"].isna().astype(int)
    songs["artist_missing"] = songs["artist"].isna().astype(int)
    songs["genre_for_analysis"] = songs["genre"].fillna("Unknown")

    songs["release_year"] = pd.to_numeric(songs["release_year"], errors="coerce")
    songs["release_year"] = songs["release_year"].where(songs["release_year"].between(1900, 2026))
    songs["release_decade"] = (songs["release_year"] // 10 * 10).astype("Int64")
    songs["genre_tag_count"] = pd.to_numeric(
        songs["genre_tags"].map(
            lambda tags: len([tag for tag in tags.split(",") if tag.strip()])
            if pd.notna(tags) else np.nan
        ), errors="coerce"
    ).astype("Float64")

    for column, pattern in GENRE_PATTERNS.items():
        # A zero with genre_missing=1 means unknown, not a confirmed negative.
        songs[column] = songs["genre_tags"].fillna("").str.contains(
            pattern, case=False, regex=True
        ).astype(int)

    songs["title_word_count"] = songs["song_title"].str.split().str.len().astype("float64")
    songs["title_character_count"] = songs["song_title"].str.len().astype("float64")
    songs["title_has_parentheses"] = songs["song_title"].str.contains(
        r"[()]", regex=True, na=False
    ).astype(int)
    songs["title_has_remix"] = songs["song_title"].str.contains(
        r"\b(?:remix|mix|edit|version)\b", flags=re.IGNORECASE,
        regex=True, na=False,
    ).astype(int)
    songs["artist_word_count"] = songs["artist"].str.split().str.len().astype("Float64")
    songs["artist_character_count"] = songs["artist"].str.len().astype("Float64")

    songs["source_snapshot_views"] = pd.to_numeric(
        songs["source_snapshot_views"], errors="coerce"
    )
    valid_views = songs["source_snapshot_views"].where(songs["source_snapshot_views"].ge(0))
    songs["source_snapshot_views"] = valid_views
    songs["target_log1p_source_views"] = np.log1p(valid_views)
    return songs


def merge_available_enrichments(songs):
    """Join optional verified enrichment snapshots by song ID."""
    songs = songs.copy()
    for filename, columns in ENRICHMENT_FILES.items():
        path = PROJECT_ROOT / "data/interim" / filename
        if not path.exists():
            continue
        metadata = pd.read_csv(path, encoding="utf-8-sig", dtype={"song_id": str})
        metadata = metadata.drop_duplicates("song_id", keep="last")
        available = [column for column in columns if column in metadata.columns]
        songs = songs.merge(
            metadata[["song_id", *available]], on="song_id", how="left", validate="one_to_one"
        )

    for column in [item for group in ENRICHMENT_FILES.values() for item in group]:
        if column not in songs:
            songs[column] = np.nan
    for column in NUMERIC_ENRICHMENTS:
        songs[column] = pd.to_numeric(songs[column], errors="coerce")
    songs["youtube_live_target_log1p_views"] = np.log1p(
        songs["youtube_views"].where(songs["youtube_views"].ge(0))
    )
    return songs


def calculate_feature_coverage(dataset):
    """Measure completeness and cardinality of every stored field."""
    records = []
    for column in dataset.columns:
        present = dataset[column].notna()
        if pd.api.types.is_string_dtype(dataset[column]):
            present &= dataset[column].astype("string").str.strip().ne("").fillna(False)
        records.append({
            "column": column,
            "role": (
                "candidate_predictor" if column in CANDIDATE_FEATURES
                else "historical_target" if column == "target_log1p_source_views"
                else "metadata"
            ),
            "non_missing_rows": int(present.sum()),
            "total_rows": len(dataset),
            "coverage_percent": round(float(present.mean()) * 100, 1),
            "unique_non_missing": int(dataset[column].nunique(dropna=True)),
        })
    return pd.DataFrame(records)


def summarize_dataset(dataset, coverage):
    """Summarize acquisition, coverage and known limitations."""
    counts = dict(zip(coverage.column, coverage.non_missing_rows))
    complete = [f for f in CANDIDATE_FEATURES if counts[f] == len(dataset)]
    return {
        "rows": int(len(dataset)),
        "unique_video_ids": int(dataset.youtube_video_id.nunique()),
        "unique_artist_title": int(dataset[["artist", "song_title"]].drop_duplicates().shape[0]),
        "source_views_available": int(dataset.source_snapshot_views.notna().sum()),
        "source_genre_nonempty": int(dataset.genre.notna().sum()),
        "source_artist_nonempty": int(dataset.artist.notna().sum()),
        "complete_engineered_features": complete,
        "complete_engineered_feature_count": len(complete),
        "live_youtube_views_available": int(dataset.youtube_views.notna().sum()),
        "musicbrainz_artist_country_available": int(dataset.artist_country.notna().sum()),
        "tempo_available": int(dataset.tempo_bpm.notna().sum()),
        "notes": [
            "The target is an undated historical view-count snapshot from a public dataset.",
            "Genres and text-based features are metadata-derived, not measured audio descriptors.",
            "Unknown artist and genre labels are not fabricated; missing indicators are included.",
            "Model-specific imputation and scaling must be fitted on training data in Phase 2.",
        ],
    }


def build_final_dataset():
    """Save the cleaned Phase 1 dataset and field-level quality reports."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing source list: {RAW_CSV}")

    source = pd.read_csv(RAW_CSV, encoding="utf-8-sig", keep_default_na=False)
    songs = create_metadata_features(source)
    songs = merge_available_enrichments(songs)

    first_columns = [
        "song_id", "artist", "song_title", "genre", "genre_tags", "genre_for_analysis",
        "release_year", "youtube_video_id", "youtube_url", "source_snapshot_views",
        "target_log1p_source_views",
    ]
    column_order = first_columns + [c for c in CANDIDATE_FEATURES if c not in first_columns]
    column_order += [c for c in songs if c not in column_order]
    songs = songs[column_order]

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    songs.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    coverage = calculate_feature_coverage(songs)
    coverage.to_csv(COVERAGE_CSV, index=False, encoding="utf-8-sig")
    report = summarize_dataset(songs, coverage)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Dataset: {len(songs)} songs; {len(CANDIDATE_FEATURES)} candidate metadata features")
    return songs


if __name__ == "__main__":
    build_final_dataset()
