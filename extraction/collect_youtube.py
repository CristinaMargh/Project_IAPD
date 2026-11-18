"""Collect public video metadata from permitted YouTube URLs with yt-dlp."""

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = PROJECT_ROOT / "data/raw/songs.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/interim/youtube_metadata.csv"
ERRORS_CSV = PROJECT_ROOT / "data/interim/youtube_errors.csv"

OUTPUT_COLUMNS = [
    "song_id", "youtube_video_id", "youtube_title", "youtube_channel",
    "youtube_channel_id", "youtube_views", "youtube_likes", "youtube_comments",
    "youtube_video_duration_seconds", "youtube_upload_date", "youtube_upload_year",
    "youtube_upload_month", "youtube_categories", "youtube_language", "youtube_collected_at_utc",
]
ERROR_COLUMNS = ["song_id", "youtube_video_id", "error", "error_at_utc"]


def append_csv_record(path, columns, record):
    """Save one YouTube result or error without overwriting earlier records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{column: record.get(column, "") for column in columns}]).to_csv(
        path, mode="a", header=not path.exists(), index=False, encoding="utf-8"
    )


def extract_video_metadata(youtube_url, expected_video_id):
    """Read metadata for one video without downloading audio or video."""
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreerrors": False,
        "socket_timeout": 20,
        "retries": 0,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        video = downloader.extract_info(youtube_url, download=False)
    if not video or str(video.get("id")) != str(expected_video_id):
        raise ValueError("Video missing or video ID does not match the source list")

    upload_date = str(video.get("upload_date") or "")
    return {
        "youtube_video_id": expected_video_id,
        "youtube_title": video.get("title"),
        "youtube_channel": video.get("channel") or video.get("uploader"),
        "youtube_channel_id": video.get("channel_id"),
        "youtube_views": video.get("view_count"),
        "youtube_likes": video.get("like_count"),
        "youtube_comments": video.get("comment_count"),
        "youtube_video_duration_seconds": video.get("duration"),
        "youtube_upload_date": upload_date,
        "youtube_upload_year": upload_date[:4],
        "youtube_upload_month": upload_date[4:6] if len(upload_date) >= 6 else "",
        "youtube_categories": " | ".join(video.get("categories") or []),
        "youtube_language": video.get("language"),
        "youtube_collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def collect_youtube_metadata(limit=5, pause_seconds=1.5):
    """Collect unprocessed videos, stopping if access restrictions are detected."""
    songs = pd.read_csv(SONGS_CSV, dtype=str, keep_default_na=False)
    existing = (
        pd.read_csv(OUTPUT_CSV, dtype=str, keep_default_na=False)
        if OUTPUT_CSV.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    pending = songs.loc[~songs.song_id.isin(set(existing.song_id))]
    if limit is not None:
        pending = pending.head(limit)
    print(f"YouTube: processing {len(pending)} songs")

    consecutive_access_errors = 0
    for index, song in enumerate(pending.itertuples(index=False), start=1):
        try:
            metadata = extract_video_metadata(song.youtube_url, song.youtube_video_id)
            metadata["song_id"] = song.song_id
            append_csv_record(OUTPUT_CSV, OUTPUT_COLUMNS, metadata)
            consecutive_access_errors = 0
            print(f"[{index}/{len(pending)}] Collected {song.song_id}: {song.song_title}")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:800]
            append_csv_record(ERRORS_CSV, ERROR_COLUMNS, {
                "song_id": song.song_id,
                "youtube_video_id": song.youtube_video_id,
                "error": message,
                "error_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[{index}/{len(pending)}] Failed {song.song_id}: {message[:140]}")
            if re.search(r"sign in|not a bot|captcha|HTTP Error 403|HTTP Error 429|forbidden|blocked", message, re.I):
                consecutive_access_errors += 1
                if consecutive_access_errors >= 2:
                    print("Stopping after repeated access restrictions")
                    break
        if pause_seconds and index < len(pending):
            time.sleep(pause_seconds)
    return OUTPUT_CSV


def main():
    parser = argparse.ArgumentParser(description="Collect optional YouTube video metadata")
    parser.add_argument("--limit", type=int, default=5, help="0 processes all remaining videos")
    parser.add_argument("--pause", type=float, default=1.5)
    args = parser.parse_args()
    collect_youtube_metadata(None if args.limit == 0 else args.limit, args.pause)


if __name__ == "__main__":
    main()
