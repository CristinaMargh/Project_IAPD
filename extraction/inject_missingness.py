"""Create a clearly labeled, controlled missing-data experiment."""
import json
import numpy as np
import pandas as pd
from ml.config import RAW_DATA, EXPERIMENT, REPORTS, SEED

RATES = {'genre': 0.08, 'genre_tags': 0.08, 'artist': 0.04, 'release_year': 0.06}


def create_missingness_experiment(seed=SEED):
    """Mask only predictor metadata, never IDs or the popularity target."""
    original = pd.read_csv(RAW_DATA)
    corrupted = original.copy()
    rng = np.random.default_rng(seed)
    additions = {}
    for column, rate in RATES.items():
        available = np.flatnonzero(original[column].notna().to_numpy())
        count = min(round(rate * len(original)), len(available))
        indexes = rng.choice(available, count, replace=False)
        corrupted.loc[indexes, column] = np.nan
        additions[column] = int(count)
    EXPERIMENT.parent.mkdir(parents=True, exist_ok=True)
    corrupted.to_csv(EXPERIMENT, index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'controlled_missingness_manifest.json').write_text(
        json.dumps({'synthetic_missingness': True, 'random_seed': seed,
                    'original_file': str(RAW_DATA.relative_to(RAW_DATA.parents[2])),
                    'experiment_file': str(EXPERIMENT.relative_to(EXPERIMENT.parents[2])),
                    'added_nulls': additions,
                    'note': 'Controlled teaching/stress-test scenario; these gaps were NOT present in the original observations.'},indent=2),
        encoding='utf-8'
    )
    return corrupted

if __name__ == '__main__':
    create_missingness_experiment()
