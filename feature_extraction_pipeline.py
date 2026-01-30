import librosa
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from pathlib import Path
from typing import List, Callable, Any, Dict, Optional
from sys import stderr
import traceback
import os
from sklearn.preprocessing import FunctionTransformer


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=stderr, **kwargs)


class NoFitTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


class AudioLoader(NoFitTransformer):
    def __init__(self, sr=None):
        self.sr = sr

    def transform(self, X):
        return [librosa.load(fname, sr=self.sr) for fname in X]

    def get_feature_names_out(self, input_features=None):
        return ["raw_librosa_data", "sr"]


class MFCCMeanExtractor(NoFitTransformer):
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc

    def transform(self, X):
        features = []
        for y, sr in X:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.append(np.mean(mfcc, axis=1))
        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return [f"mean_{i}" for i in range(self.n_mfcc)]

class MFCCVarExtractor(NoFitTransformer):
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc

    def transform(self, X):
        features = []
        for y, sr in X:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.append(np.var(mfcc, axis=1))
        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return [f"var_{i}" for i in range(self.n_mfcc)]

class FilenameExtractor(NoFitTransformer):
    def transform(self, X):
        out = []
        for path in X:
            fname = path.name
            out.append((fname,))
        return np.array(out)

    def get_feature_names_out(self, input_features=None):
        return ["filename"]

class GenreExtractor(NoFitTransformer):
    def transform(self, X):
        out = []
        for path in X:
            genre = path.parent.name
            out.append((genre,))
        return np.array(out)

    def get_feature_names_out(self, input_features=None):
        return ["genre"]

if __name__ == "__main__":
    genres = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    example_files = [
        Path("Data") / "genres_original" / genre / f"{genre}.{i:05}.wav"
        for genre in genres
        for i in range(0, 1)
    ]

    feature_union = Pipeline(
        [
            ("load_audio", AudioLoader()),
            (
                "extract_features",
                FeatureUnion(
                    [
                        ("mfcc_mean", MFCCMeanExtractor(n_mfcc=13)),
                        ("mfcc_var", MFCCVarExtractor(n_mfcc=13)),
                    ]
                ),
            ),
        ]
    )
    features = FeatureUnion(
        [
            ("filename", FilenameExtractor()),
            ("genre", GenreExtractor()),
            ("audio", feature_union),
        ]
    )
    pipeline = Pipeline(
        [
            ("features", features),
        ]
    )

    X_files = example_files
    X = pipeline.fit_transform(X_files)
    print(X)
    df = pd.DataFrame(X, columns=pipeline.get_feature_names_out())
    print(df)
