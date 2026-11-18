"""Extract audio characteristics from recordings supplied with permission."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_FOLDER = PROJECT_ROOT / "data/authorized_audio"
OUTPUT_CSV = PROJECT_ROOT / "data/interim/audio_features.csv"
OUTPUT_COLUMNS = [
    "song_id", "tempo_bpm", "audio_rms_energy", "musical_key", "musical_mode",
    "audio_analyzed_seconds", "audio_feature_source", "key_mode_confidence", "audio_collected_at_utc",
]
MUSICAL_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_musical_key(chroma_profile):
    """Estimate musical key and major/minor mode from chroma energy."""
    results = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for note_index, note in enumerate(MUSICAL_NOTES):
            reference = np.roll(profile, note_index)
            correlation = (
                float(np.corrcoef(chroma_profile, reference)[0, 1])
                if np.std(chroma_profile) else 0.0
            )
            results.append((correlation, note, mode))
    results.sort(reverse=True)
    best, second = results[:2]
    return best[1], best[2], round(best[0] - second[0], 4)


def analyze_audio_file(path):
    """Calculate approximate BPM, RMS energy and tonal features for one recording."""
    import librosa

    samples, sample_rate = librosa.load(str(path), sr=22050, mono=True, duration=180)
    if len(samples) < 3 * sample_rate:
        raise ValueError("The recording is shorter than three seconds")
    tempo, _ = librosa.beat.beat_track(y=samples, sr=sample_rate)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    rms_energy = float(np.mean(librosa.feature.rms(y=samples)[0]))
    chroma = librosa.feature.chroma_stft(y=samples, sr=sample_rate).mean(axis=1)
    musical_key, musical_mode, confidence = estimate_musical_key(chroma)
    return {
        "tempo_bpm": round(bpm, 2),
        "audio_rms_energy": round(rms_energy, 6),
        "musical_key": musical_key,
        "musical_mode": musical_mode,
        "key_mode_confidence": confidence,
        "audio_analyzed_seconds": round(len(samples) / sample_rate, 1),
        "audio_feature_source": f"provided_audio:{path.suffix.lower()}",
        "audio_collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def analyze_authorized_audio(limit=5):
    """Process supplied audio files named with their song IDs."""
    if not AUDIO_FOLDER.exists():
        print("No authorized audio folder found; audio analysis skipped")
        return
    existing = (
        pd.read_csv(OUTPUT_CSV, dtype=str, keep_default_na=False)
        if OUTPUT_CSV.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    completed = set(existing.song_id)
    files = [
        path for path in sorted(AUDIO_FOLDER.iterdir())
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
        and path.stem not in completed
    ]
    if limit is not None:
        files = files[:limit]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    for path in files:
        try:
            result = {"song_id": path.stem, **analyze_audio_file(path)}
            pd.DataFrame([result], columns=OUTPUT_COLUMNS).to_csv(
                OUTPUT_CSV, mode="a", header=not OUTPUT_CSV.exists(),
                index=False, encoding="utf-8",
            )
            print(f"Analyzed: {path.name}")
        except Exception as error:
            print(f"Skipped {path.name}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Analyze audio files supplied with permission")
    parser.add_argument("--limit", type=int, default=5, help="0 processes all files")
    args = parser.parse_args()
    analyze_authorized_audio(None if args.limit == 0 else args.limit)


if __name__ == "__main__":
    main()
