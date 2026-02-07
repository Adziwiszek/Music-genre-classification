import pandas as pd
import librosa
from tqdm.notebook import tqdm
from tslearn.metrics import dtw
from sklearn.model_selection import train_test_split
from typical import find_typical_class_member
from typing import Literal


def full_filename_from_path(path):
    g = path.split('.')[0]
    return f'Data/genres_original/{g}/{path}'


def get_sb(audio_path):
    n_fft = 2048
    hop_length = 512
    audio_path = full_filename_from_path(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length)
    return sb.flatten()


def get_zero_crossing_rate(audio_path):
    frame_length = 2048
    hop_length = 512
    audio_path = full_filename_from_path(audio_path)
    y, _ = librosa.load(audio_path, sr=None)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length,
                                             hop_length=hop_length)
    return zcr.flatten()


def get_rms(audio_path):
    frame_length = 2048
    hop_length = 512
    audio_path = full_filename_from_path(audio_path)
    y, _ = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                              hop_length=hop_length)
    return rms.flatten()


def downsample(x, factor=5):
    return x[::factor]


feature_to_function = {
        "rms": get_rms,
        "zero_crossing_rate": get_zero_crossing_rate,
        "spectral_bandwidth": get_sb,
}


def create_dtw_distances(old_csv_path: str, new_csv_path: str,
                         feature_to_measure: Literal["rms", "zero_crossing_rate", "spectral_bandwidth"] = "rms",
                         downsample_factor: int = 5,
                         verbose=False):
    """
    old_csv_path - path to the old csv
    new_csv_path - path where the new csv will be created
    feature_to_measure - which feature will be used for measuring the distance
    downsample_factor - how much to scale down amount of data measured for a feature
    verbose - print messages about progress

    This function takes old csv and adds 10 new features to each row: distances
    from typical songs from each genre to the song from that row (distances are
    computed with dtw)
    """
    get_fun = feature_to_function[feature_to_measure]

    df = pd.read_csv(old_csv_path)
    # Removing invalid file
    df = df.drop(df[df['filename'] == 'jazz.00054.wav'].index)

    all_paths = list(df['filename'])
    all_paths = list(filter(lambda p: p != 'jazz.00054.wav', all_paths))

    # Get typical songs =======================================================

    genres = df['label'].unique()
    X = df.drop(["filename", "length", "label"], axis=1)
    y = df["label"]
    typical_songs = {}
    for k, v in find_typical_class_member(X=X, y=y, method='centroid').items():
        typical_songs[k] = df.iloc[v]['filename']

    # Compute selected feature ================================================

    if verbose:
        print('Computing time series for all songs...')

    sb_cache = {
        path: downsample(get_fun(path), factor=downsample_factor)
        for path in tqdm(all_paths)
    }

    if verbose:
        print('Get time series for typical songs...')

    typical_sb = {
        g: sb_cache[typical_songs[g]]
        for g in genres
    }

    # Compute distances for each song =========================================

    distance_from_typical = {
        f'dist_from_{g}': []
        for g in genres
    }

    if verbose:
        print('Computing distances...')

    for path in tqdm(all_paths):
        song_sb = sb_cache[path]
        for g in genres:
            g_sb = typical_sb[g]
            dist = dtw(song_sb, g_sb)
            distance_from_typical[f'dist_from_{g}'].append(dist)
    dist_df = pd.DataFrame(distance_from_typical)

    # Add computed distances to the dataframe and save as csv =================

    extra_data = pd.concat(
        [df.reset_index(drop=True),
         dist_df.reset_index(drop=True)],
        axis=1
    )
    extra_data.to_csv(new_csv_path, index=False)


class DTWFeatureGenerator:
    def __init__(self, dist_type='rms', downsample_factor=5):
        self.downsample_factor = downsample_factor
        self.feature_fun = feature_to_function[dist_type]

    def fit(self, df):
        """Takes training df and finds most typical songs."""
        df = df.drop(df[df['filename'] == 'jazz.00054.wav'].index)

        self.train_paths = list(df['filename'])
        self.genres = df['label'].unique()

        # find typical songs
        X = df.drop(["filename", "length", "label"], axis=1)
        y = df["label"]
        self.typical_songs = {}
        for k, v in find_typical_class_member(X=X, y=y, method='centroid').items():
            self.typical_songs[k] = df.iloc[v]['filename']

        # calculate feature for typical songs
        self.typical_cache = {
            g: downsample(self.feature_fun(p), factor=self.downsample_factor)
            for g, p in tqdm(self.typical_songs.items())
        }

    def transform(self, df, new_csv_path=None):
        """Calculates new features for rows from df and optionally saves that
        new df to new_csv_path."""
        paths = list(df['filename'])

        feature_cache = {
            path: downsample(self.feature_fun(path), factor=self.downsample_factor)
            for path in tqdm(paths)
        }

        distance_from_typical = {
            f'dist_from_{g}': []
            for g in self.genres
        }

        for path in tqdm(paths):
            song_feature = feature_cache[path]
            for g in self.genres:
                g_feature = self.typical_cache[g]
                dist = dtw(song_feature, g_feature)
                distance_from_typical[f'dist_from_{g}'].append(dist)
        dist_df = pd.DataFrame(distance_from_typical)

        extra_data = pd.concat(
            [df.reset_index(drop=True),
             dist_df.reset_index(drop=True)],
            axis=1
        )
        if new_csv_path:
            extra_data.to_csv(new_csv_path, index=False)
        return extra_data

def train_test_split_df(old_csv_path, train_csv=None, test_csv=None, test_size=0.3, random_state=42):
    old_df = pd.read_csv(old_csv_path)
    train_df, test_df = train_test_split(old_df, test_size=test_size, random_state=random_state)
    if train_csv:
        train_df.to_csv(train_csv, index=False)
        print(f'saved train df to {train_csv}')
    if test_csv:
        test_df.to_csv(test_csv, index=False)
        print(f'saved test df to {test_csv}')
    return train_df, test_df


if __name__ == "__main__":
    """Crate new csv with calculated dtw features."""
    import os
    from pathlib import Path 
    from tqdm import tqdm

    dtw_path = 'Data/DTWFeatures'
    dtw_dir_path = Path(dtw_path)
    if not dtw_dir_path.exists():
        os.mkdir('Data/DTWFeatures')

    old_csv = 'Data/features_30_sec.csv'
    train_csv_path = dtw_path + '/baseline_train.csv'
    test_csv_path = dtw_path + '/baseline_test.csv'

    train_csv_path_dtw = dtw_path + '/dtw_train.csv'
    test_csv_path_dtw = dtw_path + '/dtw_test.csv'

    # splititng the data
    train_df, test_df = train_test_split_df(old_csv, train_csv=train_csv_path, test_csv=test_csv_path)

    # calculating new features and saving to csv
    generator = DTWFeatureGenerator(dist_type='zero_crossing_rate', downsample_factor=5)
    generator.fit(train_df)
    generator.transform(train_df, new_csv_path=train_csv_path_dtw)
    generator.transform(test_df, new_csv_path=test_csv_path_dtw)

    print(f'Created DTW features and saved them to {train_csv_path} and {test_csv_path}')