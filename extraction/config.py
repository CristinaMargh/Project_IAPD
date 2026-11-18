"""Paths and model configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / 'data/raw/songs.csv'
DATASET = ROOT / 'data/processed/music_popularity_dataset.csv'
EXPERIMENT = ROOT / 'data/experiments/controlled_missingness.csv'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
FIGURES = ROOT / 'docs/figures'
TARGET = 'source_snapshot_views'
SEED = 42

NUMERIC = [
    'release_year', 'genre_tag_count', 'title_word_count', 'title_character_count',
    'title_has_parentheses', 'title_has_featured', 'title_has_live',
    'artist_word_count', 'artist_character_count', 'artist_collaboration_hint',
    'artist_missing', 'genre_missing',
]
CATEGORICAL = ['genre_family']
PREDICTORS = NUMERIC + CATEGORICAL

GENRE_PATTERNS = {
    'K-pop': r'k[ -]?pop|korean pop',
    'Hip-hop': r'hip[ -]?hop|rap|trap|drill',
    'Latin': r'latin|reggaeton|bachata|salsa|merengue|cumbia',
    'R&B': r'r&b|rhythm and blues|soul|funk',
    'Rock': r'rock|metal|punk|grunge',
    'Electronic': r'electro|edm|dance|house|techno|trance|dubstep',
    'Country': r'country|bluegrass|americana',
    'Pop': r'pop|synthpop|europop',
}
