import pandas as pd
import librosa
from tqdm.notebook import tqdm
from tslearn.metrics import dtw
from sklearn.model_selection import train_test_split
from typical import find_typical_class_member
from typing import Literal
import numpy as np

from .create_dtw_distances import (feature_to_function, downsample, 
                                  full_filename_from_path, get_rms, 
                                  get_zero_crossing_rate, get_sb)

def create_timeseries_dataframe(
    csv_path: str,
    feature: Literal["rms", "zero_crossing_rate", "spectral_bandwidth"] = "rms",
    downsample_factor: int = 5,
    verbose: bool = False
) -> pd.DataFrame:

    get_fun = feature_to_function[feature]
    df = pd.read_csv(csv_path)

    df = df[df['filename'] != 'jazz.00054.wav'].reset_index(drop=True)

    records = []

    if verbose:
        print("Computing time series...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row['filename']
        label = row['label']

        ts = get_fun(path)
        ts = downsample(ts, factor=downsample_factor)

        records.append({
            "filename": path,
            "ts": ts,
            "label": label
        })

    ts_df = pd.DataFrame(records)
    return ts_df

if __name__ == "__main__":
    from tqdm import tqdm
    input_csv = "Data/features_30_sec.csv"
    output_csv = "Data/ts_data.csv"
    ts_df = create_timeseries_dataframe(input_csv, feature="rms", downsample_factor=10, verbose=True)
    max_length = max(len(ts) for ts in ts_df['ts'])

    X = pd.DataFrame(
        np.array([np.pad(ts, (0, max_length - len(ts))) for ts in ts_df['ts']]),
        columns=[f'ts_{i}' for i in range(max_length)]
    )

    df = pd.read_csv("Data/features_30_sec.csv")
    df = pd.concat([df, X], axis=1)
    
    df.to_csv(output_csv, index=False)

    print(f'saved csv to {output_csv}')