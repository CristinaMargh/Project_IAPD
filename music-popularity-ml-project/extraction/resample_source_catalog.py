"""Rebuild a less skewed catalog from the full public source Parquet file."""
import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from ml.config import ROOT, RAW_DATA, SEED

CATALOG = ROOT / 'data/raw/source_catalog.parquet'
BOUNDS = [0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, float('inf')]
LABELS = ['<10k', '10k–100k', '100k–1m', '1m–10m', '10m–100m', '100m–1b', '>=1b']


def normalized_key(value):
    """Create comparison keys that ignore punctuation and accents."""
    value = unicodedata.normalize('NFKD', str(value)).casefold()
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return re.sub('[^a-z0-9]+', '', value)


def prepare_candidates(source):
    """Validate metadata and remove repeated videos and song identifiers."""
    expected={'youtubeId','itemLabel','performerLabel','youtubeViews','year','genreLabel'}
    if not expected.issubset(source.columns):
        raise ValueError(f'Invalid source schema: {expected-set(source.columns)}')
    df = source.copy()
    for field in ['youtubeId','itemLabel','performerLabel','genreLabel']:
        df[field] = df[field].fillna('').astype(str).str.strip()
    df['youtubeViews']=pd.to_numeric(df.youtubeViews,errors='coerce')
    df['year']=pd.to_numeric(df.year,errors='coerce')
    df=df[df.youtubeId.str.fullmatch(r'[A-Za-z0-9_-]{11}') & df.youtubeViews.ge(0) & df.year.between(1970,2026)]
    df=df[df.itemLabel.str.len().ge(2) & df.performerLabel.str.len().ge(2)]
    df=df[~df.itemLabel.str.fullmatch(r'Q\d+') & ~df.performerLabel.str.fullmatch(r'Q\d+')]
    df=df.drop_duplicates('youtubeId')
    df['track_key'] = [normalized_key(a)+'-'+normalized_key(t) for a,t in zip(df.performerLabel,df.itemLabel)]
    return df.drop_duplicates('track_key')


def popularity_spread_sample(candidates, target=1000, seed=SEED, artist_limit=8):
    """Sample across fixed view-count ranges, limiting repeated artists."""
    frame=candidates.copy()
    frame['view_band']=pd.cut(frame.youtubeViews, BOUNDS, labels=LABELS, include_lowest=True, right=False)
    bins={label:list(frame.loc[frame.view_band.eq(label)].sample(frac=1,random_state=seed).index) for label in LABELS}
    picked=[]
    artists={}
    while len(picked) < target:
        progress=False
        for label in LABELS:
            while bins[label]:
                idx=bins[label].pop()
                artist=normalized_key(frame.at[idx,'performerLabel'])
                if artists.get(artist,0) >= artist_limit:
                    continue
                picked.append(idx)
                artists[artist]=artists.get(artist,0)+1
                progress=True
                break
            if len(picked)==target:
                break
        if not progress:
            raise ValueError(f'Only {len(picked)} distinct songs satisfy quality and artist limits')
    return frame.loc[picked].sample(frac=1,random_state=seed).reset_index(drop=True)


def resample_source_catalog(target=1000, replace=False, seed=SEED):
    """Generate a fresh 1,000-song catalog without preserving the original high-view subset."""
    try:
        catalog=pd.read_parquet(CATALOG)
    except ImportError as exc:
        raise ImportError('Install optional dependency: pip install pyarrow') from exc
    selected=popularity_spread_sample(prepare_candidates(catalog),target,seed)
    output=pd.DataFrame({
        'song_id':[f'S{i:04d}' for i in range(1,len(selected)+1)],
        'artist':selected.performerLabel.to_numpy(),
        'song_title':selected.itemLabel.to_numpy(),
        'release_year':selected.year.astype(int).to_numpy(),
        'genre':selected.genreLabel.map(lambda x: x.split(',')[0].strip() or np.nan).to_numpy(),
        'youtube_video_id':selected.youtubeId.to_numpy(),
        'youtube_url':['https://www.youtube.com/watch?v='+x for x in selected.youtubeId],
        'genre_tags':selected.genreLabel.replace('',np.nan).to_numpy(),
        'source_snapshot_views':selected.youtubeViews.astype('int64').to_numpy(),
    })
    save_path=ROOT/'data/experiments/stratified_catalog_1000.csv'
    save_path.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(save_path,index=False)
    count=selected.view_band.value_counts().reindex(LABELS,fill_value=0).to_dict()
    (ROOT/'data/experiments/stratified_catalog_manifest.json').write_text(
        json.dumps({'source':'data/raw/source_catalog.parquet','selection':'uniform round-robin across log-spaced view bands',
                    'seed':seed,'artist_limit':8,'count_by_band':count,
                    'note':'A stratified learning sample, not an estimate of population popularity.'},indent=2),encoding='utf-8')
    if replace:
        backup=ROOT/'data/raw/songs_original_snapshot.csv'
        if not backup.exists():
            shutil.copy2(RAW_DATA,backup)
        shutil.copy2(save_path,RAW_DATA)
        old_cache_dir=ROOT/'data/interim/previous_catalog_backup'
        old_cache_dir.mkdir(parents=True,exist_ok=True)
        for old_name in ['musicbrainz_metadata.csv','youtube_metadata.csv','audio_features.csv','artist_cache.json']:
            old=ROOT/'data/interim'/old_name
            if old.exists():
                shutil.move(str(old),str(old_cache_dir/old_name))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--replace',action='store_true',help='Replace the project raw catalog; back up existing file')
    parser.add_argument('--target',type=int,default=1000)
    args=parser.parse_args()
    result=resample_source_catalog(target=args.target,replace=args.replace)
    print(f'Generated {len(result)} songs; use --replace to set as default raw dataset')
