from dataclasses import dataclass

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

target_col = 'label'
drop_cols_for_training = ['filename', 'length', 'label']


@dataclass
class TrainingConfig:
    """Class with parameters for training a model."""
    csv_path: str
    pipeline: Pipeline
    test_size: float = 0.2


def train_model(config: TrainingConfig):
    df = pd.read_csv(config.csv_path)
    X, y = df.drop(drop_cols_for_training, axis=1), df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=42
    )

    pipeline = config.pipeline

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
    }

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "pipeline": pipeline,
        "metrics": metrics,
    }
