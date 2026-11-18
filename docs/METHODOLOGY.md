# Methodology and reproducibility

## Acquisition

Catalog records originate in a publicly available Wikidata-derived YouTube dataset. The project includes optional Python extractors for public YouTube metadata and MusicBrainz, but only the historical snapshot has complete view counts. Raw provenance is documented without claiming that all videos were crawled live.

## Quality checks

1. Enforce unique video IDs and normalized artist–title combinations.
2. Validate years and nonnegative view counts; convert text placeholders to genuine nulls.
3. Audit real field-level missingness and calculate target outliers with the 1.5-IQR criterion.
4. Keep raw target outliers; transform the target using log1p rather than deleting popular hits.
5. Build metadata features; avoid IDs, present-day popularity proxies and nearly empty audio feature columns.

## Controlled missingness experiment

A separate dataset masks metadata (genre, genre tags, artist and release year) with reproducible random seed 42. Only predictor information is masked. The source and targets remain unchanged. Two independent training runs on identical train/test song IDs measure performance degradation.

## ML preprocessing

`IQRClipper` learns per-feature fences from training folds and caps predictor extremes, without modifying the target. `SimpleImputer(strategy='median')` handles numerical gaps; categorical genre families are mapped to broad groups including Unknown, encoded via OneHotEncoder. Linear models use StandardScaler; all models use VarianceThreshold after preprocessing. All learned operations sit inside sklearn Pipeline to prevent leakage.

## Modeling and evaluation

The features are metadata-only and describe the songs independent of current YouTube engagement. Split: GroupShuffleSplit(random_state=42), 20% artist-disjoint test set. Selection: three-fold GroupKFold on training songs, minimizing mean log-RMSE. Candidates: median DummyRegressor, LinearRegression, Ridge, RandomForest and weighted RandomForest. The weighted experiment uses fixed historic-view bands to reweight training rows, recomputed on each CV training fold only.

Evaluation: MAE, RMSE and R² on log1p historical views; median absolute error after inverse transformation. Reports include actual vs predicted, residuals, view-band errors and permutation importance.

## Limitation

The public snapshot has no certified measurement timestamp. Sampling and target representativeness are uncertain; training weights cannot reconstruct the full population. The dataset lacks authorized audio descriptors (tempo/energy/key). Results describe historical popularity associations, not causal or future forecasts.
