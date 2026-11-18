"""Report data quality, missingness and outliers without modifying evidence."""
import json
import pandas as pd
import numpy as np
from ml.config import DATASET, RAW_DATA, REPORTS
from ml.features import clean_label, make_features


def count_iqr(values):
    """Calculate standard 1.5-IQR outlier fences and counts."""
    v = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    delta = q3 - q1
    lo, hi = q1 - 1.5 * delta, q3 + 1.5 * delta
    return {'count': int(((v < lo) | (v > hi)).sum()), 'lower': float(lo), 'upper': float(hi)}


def audit_dataset():
    """Save genuine original-data diagnostics and a preliminary feature audit."""
    raw = pd.read_csv(RAW_DATA)
    dataset = pd.read_csv(DATASET)
    REPORTS.mkdir(parents=True, exist_ok=True)
    missing = []
    for column in ['artist','genre','genre_tags','release_year','source_snapshot_views']:
        values = raw[column].map(clean_label) if raw[column].dtype == object else raw[column]
        missing.append({'attribute': column, 'missing': int(values.isna().sum()), 'pct': round(values.isna().mean()*100,2)})
    pd.DataFrame(missing).to_csv(REPORTS/'observed_missing_values.csv',index=False)
    features = make_features(dataset)
    selection = []
    for column in features.columns:
        distinct = features[column].nunique(dropna=True)
        selection.append({'feature':column, 'unique':int(distinct), 'missing':int(features[column].isna().sum()),
                          'preliminary_action':'exclude_constant' if distinct<=1 else 'keep_candidate'})
    pd.DataFrame(selection).to_csv(REPORTS/'feature_audit.csv',index=False)
    log_views = np.log1p(dataset.source_snapshot_views)
    report = {
        'rows': int(len(dataset)),
        'unique_video_ids':int(dataset.youtube_video_id.nunique()),
        'unique_artist_title':int(dataset[['artist','song_title']].drop_duplicates().shape[0]),
        'observed_missing':{x['attribute']:x['missing'] for x in missing},
        'raw_view_outliers':count_iqr(dataset.source_snapshot_views),
        'log_view_outliers':count_iqr(log_views),
        'target_range': [int(dataset.source_snapshot_views.min()),int(dataset.source_snapshot_views.max())],
        'genre_count':int(dataset.genre.nunique(dropna=True)),
        'candidate_predictors':len(features.columns),
        'actual_source_note':'Historical popularity snapshot; observation date not documented.',
        'data_sampling_warning':'The supplied catalog favors high-view songs in one subset and low-view songs in another. Resampling the full source catalog is recommended before causal or population claims.',
        'imputation_note':'Imputers and outlier-clippers are fitted ONLY within training folds.'
    }
    (REPORTS/'data_quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    print(json.dumps(audit_dataset(),indent=2))
