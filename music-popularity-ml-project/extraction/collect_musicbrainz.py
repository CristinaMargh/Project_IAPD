"""Collect recording and artist metadata from MusicBrainz."""

import argparse
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = PROJECT_ROOT / "data/raw/songs.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/interim/musicbrainz_metadata.csv"
REVIEW_CSV = PROJECT_ROOT / "data/interim/musicbrainz_review_candidates.csv"
ARTIST_CACHE = PROJECT_ROOT / "data/interim/artist_cache.json"

OUTPUT_COLUMNS = [
    "song_id", "musicbrainz_status", "musicbrainz_recording_id", "musicbrainz_recording_title",
    "musicbrainz_artist_credit", "musicbrainz_artist_id", "musicbrainz_score", "title_similarity",
    "artist_similarity", "song_duration_seconds", "musicbrainz_earliest_listed_release_date",
    "musicbrainz_release_year", "musicbrainz_release_month", "artist_country", "artist_type",
    "artist_count", "is_collaboration", "musicbrainz_recording_url", "musicbrainz_collected_at_utc",
]
REVIEW_COLUMNS = [
    "song_id", "artist", "song_title", "candidate_title", "candidate_artist",
    "candidate_recording_id", "musicbrainz_score", "title_similarity",
    "artist_similarity", "reason",
]


def normalize_text(value):
    """Simplify text before comparing titles and artist names."""
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def calculate_similarity(first, second):
    """Return a normalized similarity score between two strings."""
    first, second = normalize_text(first), normalize_text(second)
    return SequenceMatcher(None, first, second).ratio() if first and second else 0.0


def extract_artist_credits(credits):
    """Return artist records from a MusicBrainz credit list."""
    return [credit for credit in (credits or []) if isinstance(credit, dict) and "artist" in credit]


def extract_artist_names(credits):
    """Extract the credited artist names from one recording."""
    return [
        str(credit.get("name") or credit.get("artist", {}).get("name") or "")
        for credit in extract_artist_credits(credits)
    ]


def calculate_artist_match(artist, credits):
    """Compare the original artist against the recording's credited artists."""
    names = extract_artist_names(credits)
    return max([calculate_similarity(artist, name) for name in names] + [
        calculate_similarity(artist, ", ".join(names)), 0.0
    ])


def append_csv_row(path, columns, values):
    """Save each processed record without losing earlier API results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{name: values.get(name, "") for name in columns}]).to_csv(
        path, mode="a", index=False, header=not path.exists(), encoding="utf-8"
    )


class MusicBrainzClient:
    """Send requests at a rate that respects the public MusicBrainz service."""

    def __init__(self, user_agent):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.last_request_time = 0.0

    def get_json(self, url, params=None):
        """Retrieve JSON while spacing requests by at least 1.15 seconds."""
        remaining = 1.15 - (time.monotonic() - self.last_request_time)
        if remaining > 0:
            time.sleep(remaining)
        self.last_request_time = time.monotonic()
        response = self.session.get(url, params=params, timeout=30)
        if response.status_code in (429, 503):
            raise RuntimeError(f"MusicBrainz temporarily unavailable: HTTP {response.status_code}")
        response.raise_for_status()
        return response.json()

    def search_recordings(self, title, artist):
        """Search for recordings by song title and artist."""
        title = str(title).replace('"', " ").strip()
        artist = str(artist).replace('"', " ").strip()
        query = f'recording:"{title}" AND artistname:"{artist}"'
        data = self.get_json(
            "https://musicbrainz.org/ws/2/recording/",
            {"query": query, "fmt": "json", "limit": 8},
        )
        return data.get("recordings", [])

    def get_artist_details(self, artist_id):
        """Read the country and artist type for a matched artist."""
        return self.get_json(f"https://musicbrainz.org/ws/2/artist/{artist_id}", {"fmt": "json"})


def select_best_recording(candidates, original_title, original_artist):
    """Rank MusicBrainz candidates using title, artist and API match score."""
    ranked = []
    for recording in candidates:
        title_score = calculate_similarity(original_title, recording.get("title", ""))
        artist_score = calculate_artist_match(original_artist, recording.get("artist-credit", []))
        api_score = int(recording.get("score") or 0)
        combined = 0.55 * title_score + 0.30 * artist_score + 0.15 * api_score / 100
        ranked.append((combined, recording, title_score, artist_score, api_score))
    return max(ranked, key=lambda result: result[0]) if ranked else None


def build_match_details(recording, title_score, artist_score, api_score):
    """Build metadata from a confidently matched recording."""
    credits = extract_artist_credits(recording.get("artist-credit", []))
    dates = [
        release.get("date", "") for release in recording.get("releases", [])
        if re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", release.get("date", ""))
    ]
    release_date = min(dates) if dates else ""
    length_ms = recording.get("length")
    artist_id = credits[0].get("artist", {}).get("id", "") if credits else ""
    recording_id = recording.get("id", "")
    return {
        "musicbrainz_status": "MATCHED",
        "musicbrainz_recording_id": recording_id,
        "musicbrainz_recording_url": f"https://musicbrainz.org/recording/{recording_id}",
        "musicbrainz_recording_title": recording.get("title", ""),
        "musicbrainz_artist_credit": ", ".join(extract_artist_names(recording.get("artist-credit", []))),
        "musicbrainz_artist_id": artist_id,
        "musicbrainz_score": api_score,
        "title_similarity": round(title_score, 3),
        "artist_similarity": round(artist_score, 3),
        "song_duration_seconds": round(length_ms / 1000, 2) if isinstance(length_ms, (int, float)) else "",
        "musicbrainz_earliest_listed_release_date": release_date,
        "musicbrainz_release_year": release_date[:4],
        "musicbrainz_release_month": release_date[5:7] if len(release_date) >= 7 else "",
        "artist_count": len(credits),
        "is_collaboration": int(len(credits) > 1),
    }


def collect_musicbrainz_metadata(limit=5, user_agent=None):
    """Enrich unprocessed songs and save uncertain matches for manual review."""
    user_agent = user_agent or os.getenv("MUSICBRAINZ_USER_AGENT")
    if not user_agent:
        raise ValueError("Set MUSICBRAINZ_USER_AGENT with your application name and contact information")

    songs = pd.read_csv(SONGS_CSV, dtype=str, keep_default_na=False)
    previous = (
        pd.read_csv(OUTPUT_CSV, dtype=str, keep_default_na=False)
        if OUTPUT_CSV.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    processed = set(previous.loc[
        previous.musicbrainz_status.isin(["MATCHED", "NEEDS_REVIEW"]), "song_id"
    ])
    pending = songs.loc[~songs.song_id.isin(processed)]
    if limit is not None:
        pending = pending.head(limit)

    artist_cache = json.loads(ARTIST_CACHE.read_text(encoding="utf-8")) if ARTIST_CACHE.exists() else {}
    client = MusicBrainzClient(user_agent)
    print(f"MusicBrainz: processing {len(pending)} songs")

    for song in pending.itertuples(index=False):
        record = {
            "song_id": song.song_id,
            "musicbrainz_collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            best = select_best_recording(
                client.search_recordings(song.song_title, song.artist), song.song_title, song.artist
            )
            if best is None:
                record["musicbrainz_status"] = "NEEDS_REVIEW"
                append_csv_row(REVIEW_CSV, REVIEW_COLUMNS, {
                    "song_id": song.song_id, "artist": song.artist,
                    "song_title": song.song_title, "reason": "No recording found",
                })
            else:
                _, recording, title_score, artist_score, api_score = best
                credits = recording.get("artist-credit", [])
                confident = (
                    api_score >= 85 and title_score >= 0.78
                    and artist_score >= 0.68 and bool(extract_artist_credits(credits))
                )
                if confident:
                    record.update(build_match_details(recording, title_score, artist_score, api_score))
                    artist_id = record["musicbrainz_artist_id"]
                    if artist_id:
                        if artist_id not in artist_cache:
                            artist_cache[artist_id] = client.get_artist_details(artist_id)
                            ARTIST_CACHE.write_text(
                                json.dumps(artist_cache, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                        record["artist_country"] = artist_cache[artist_id].get("country", "")
                        record["artist_type"] = artist_cache[artist_id].get("type", "")
                else:
                    record["musicbrainz_status"] = "NEEDS_REVIEW"
                    append_csv_row(REVIEW_CSV, REVIEW_COLUMNS, {
                        "song_id": song.song_id, "artist": song.artist, "song_title": song.song_title,
                        "candidate_title": recording.get("title", ""),
                        "candidate_artist": ", ".join(extract_artist_names(credits)),
                        "candidate_recording_id": recording.get("id", ""),
                        "musicbrainz_score": api_score,
                        "title_similarity": round(title_score, 3),
                        "artist_similarity": round(artist_score, 3),
                        "reason": "Low-confidence match",
                    })
            append_csv_row(OUTPUT_CSV, OUTPUT_COLUMNS, record)
            print(f"{song.song_id}: {record['musicbrainz_status']}")
        except (requests.RequestException, RuntimeError, ValueError) as error:
            print(f"Stopping at {song.song_id}: {error}")
            break
    return OUTPUT_CSV


def main():
    parser = argparse.ArgumentParser(description="Enrich songs with MusicBrainz metadata")
    parser.add_argument("--limit", type=int, default=5, help="0 processes all remaining songs")
    parser.add_argument("--user-agent", default=None)
    args = parser.parse_args()
    collect_musicbrainz_metadata(None if args.limit == 0 else args.limit, args.user_agent)


if __name__ == "__main__":
    main()
