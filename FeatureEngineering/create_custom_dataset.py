from feature_extraction_pipeline import build_feature_pipeline
from pathlib import Path
import pandas as pd
from typing import List

def create_custom_dataset(song_paths: List[str]) -> pd.DataFrame:
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(song_paths)

    columns = []
    for old_c in pipeline.get_feature_names_out():
        if old_c.startswith('filename__'):
            columns.append('filename')
        elif old_c.startswith('genre__'):
            columns.append('genre')
        else:
            columns.append(old_c.split('__')[-1])
    
    df = pd.DataFrame(X, columns=columns)
    return df