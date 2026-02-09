import librosa
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from pathlib import Path
from typing import Any
from sys import stderr


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


class FilenameExtractor(NoFitTransformer):
    def transform(self, X):
        return np.array([(path.name,) for path in X])

    def get_feature_names_out(self, input_features=None):
        return ["filename"]


class GenreExtractor(NoFitTransformer):
    def transform(self, X):
        return np.array([(path.parent.name,) for path in X])

    def get_feature_names_out(self, input_features=None):
        return ["label"]


class LengthExtractor(NoFitTransformer):
    def transform(self, X):
        return np.array([[len(y) / sr] for y, sr in X])

    def get_feature_names_out(self, input_features=None):
        return ["length"]


class ChromaStftExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            c = librosa.feature.chroma_stft(y=y, sr=sr)
            rows.append([c.mean(), c.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["chroma_stft_mean", "chroma_stft_var"]


class RmsExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            r = librosa.feature.rms(y=y)
            rows.append([r.mean(), r.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["rms_mean", "rms_var"]


class SpectralCentroidExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)
            rows.append([sc.mean(), sc.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["spectral_centroid_mean", "spectral_centroid_var"]


class SpectralBandwidthExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rows.append([sb.mean(), sb.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["spectral_bandwidth_mean", "spectral_bandwidth_var"]


class RolloffExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rows.append([ro.mean(), ro.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["rolloff_mean", "rolloff_var"]


class ZeroCrossingRateExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            zcr = librosa.feature.zero_crossing_rate(y=y)
            rows.append([zcr.mean(), zcr.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["zero_crossing_rate_mean", "zero_crossing_rate_var"]


class HarmonyExtractor(NoFitTransformer):
    """Harmonic component via harmonic-percussive source separation."""

    def transform(self, X):
        rows = []
        for y, sr in X:
            harmony = librosa.effects.harmonic(y)
            rows.append([harmony.mean(), harmony.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["harmony_mean", "harmony_var"]


class PerceptualExtractor(NoFitTransformer):
    """Percussive component via harmonic-percussive source separation."""

    def transform(self, X):
        rows = []
        for y, sr in X:
            perceptr = librosa.effects.percussive(y)
            rows.append([perceptr.mean(), perceptr.var()])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["perceptr_mean", "perceptr_var"]


class TempoExtractor(NoFitTransformer):
    def transform(self, X):
        rows = []
        for y, sr in X:
            t = librosa.feature.tempo(y=y, sr=sr)
            rows.append([t[0]])
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return ["tempo"]


class MFCCExtractor(NoFitTransformer):
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc

    def transform(self, X):
        rows = []
        for y, sr in X:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            row = []
            for i in range(self.n_mfcc):
                row.append(mfcc[i].mean())
                row.append(mfcc[i].var())
            rows.append(row)
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        names = []
        for i in range(self.n_mfcc):
            names.append(f"mfcc{i + 1}_mean")
            names.append(f"mfcc{i + 1}_var")
        return names

EXTRACTORS = {
    "length": LengthExtractor,
    "chroma_stft": ChromaStftExtractor,
    "rms": RmsExtractor,
    "spectral_centroid": SpectralCentroidExtractor,
    "spectral_bandwidth": SpectralBandwidthExtractor,
    "rolloff": RolloffExtractor,
    "zero_crossing_rate": ZeroCrossingRateExtractor,
    "harmony": HarmonyExtractor,
    "perceptr": PerceptualExtractor,
    "tempo": TempoExtractor,
    "mfcc": MFCCExtractor,
}

def build_feature_pipeline(sr: int | None = None, extractors = EXTRACTORS) -> Pipeline:
    extraction_steps = [(name, cls()) for name, cls in extractors.items()]
    
    audio_pipeline = Pipeline(
        [
            ("load_audio", AudioLoader(sr=sr)),
            ("extract_features", FeatureUnion(extraction_steps)),
        ]
    )
    features = FeatureUnion(
        [
            ("filename", FilenameExtractor()),
            ("genre", GenreExtractor()),
            ("audio", audio_pipeline),
        ]
    )
    return Pipeline([("features", features)])

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
    ][:1]

    pipeline = build_feature_pipeline()

    X = pipeline.fit_transform(example_files)
    df = pd.DataFrame(X, columns=pipeline.get_feature_names_out())
    print(f"Extracted {df.shape[1]} features from {df.shape[0]} files.")
    print(df.head(1).to_string())
