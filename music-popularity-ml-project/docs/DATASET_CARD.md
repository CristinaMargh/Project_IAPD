# Dataset card

**Name:** Music popularity metadata sample (1,000 songs).

**Source:** Public Wikidata-derived YouTube Music Hits catalog at https://huggingface.co/datasets/akbargherbal/youtube-music-hits. Raw source metadata are stored in `data/raw/songs.csv`, with full source Parquet included for optional resampling.

**Target:** `source_snapshot_views`, historical YouTube views with an undocumented measurement date. `log1p(source_snapshot_views)` is the target used by regression.

**Usable metadata:** release year, genre/tags, title, artist; engineered word/character counts and binary text patterns, music-family encoding. Optional partial MusicBrainz and YouTube enrichments exist but are **excluded** from the supervised model due to coverage/temporal leakage risks.

**Real observed gaps:** 149 missing primary genre labels and 7 artist labels in the 1,000-song snapshot. 43 raw target outliers by IQR. These are preserved/handled reproducibly, not fabricated.

**Experiment only:** a separate CSV intentionally masks predictor metadata using a fixed seed; its missingness is *synthetic* and should never be attributed to data acquisition failures.

**Known biases:** high-view songs are overrepresented in one collection subset; the catalog is not a random sample of released songs. Weighted Random Forest is evaluated but not assumed to repair this bias.

**Privacy:** the sample stores public artist/song metadata and video identifiers, not personal listening histories.

**Usage:** educational descriptive analysis and historical popularity regression. Avoid extrapolating to all songs or claiming true future prediction.
