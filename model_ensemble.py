from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import numpy as np


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_model_constructor, labels_per_model, **model_kwargs):
        if not callable(base_model_constructor):
            raise ValueError(
                "base_model_constructor must be a callable that returns an instance of a model"
            )
        self.labels_per_model = labels_per_model
        self.base_model_constructor = base_model_constructor
        self.model_kwargs = model_kwargs

    _OTHER_LABEL = "__other__"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.label_combinations_ = list(
            combinations(self.classes_, self.labels_per_model)
        )

        self.models_ = []
        for label_group in self.label_combinations_:
            model = self.base_model_constructor(**self.model_kwargs)
            if self.labels_per_model == 1:
                # binary yes/no
                y_binary = np.where(np.isin(y, label_group), y, self._OTHER_LABEL)
                model.fit(X, y_binary)
            else:
                # multiclass
                mask = np.isin(y, label_group)
                model.fit(X[mask], y[mask])
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        scores = np.zeros((X.shape[0], self.n_classes_))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for model, label_group in zip(self.models_, self.label_combinations_):
            proba = model.predict_proba(X)
            for j, label in enumerate(model.classes_):
                if label == self._OTHER_LABEL:
                    continue
                scores[:, class_to_idx[label]] += proba[:, j]

        row_sums = scores.sum(axis=1, keepdims=True)
        return scores / row_sums

    def predict(self, X):
        scores = self.predict_proba(X)
        return self.classes_[np.argmax(scores, axis=1)]


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    df = pd.read_csv("Data/features_30_sec.csv")
    results = []
    X = df.drop(["filename", "length", "label"], axis=1)
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_ = StandardScaler()
    X_tr_scaled = scaler_.fit_transform(X_tr)
    X_te_scaled = scaler_.transform(X_te)
    print("Unique counts of true labels in test set:")
    print(pd.DataFrame(np.unique(y_te, return_counts=True)).to_string(index=False))
    print("======================KNeighborsClassifier======================")
    for n in [1, 2, 3, 5]:
        ec = EnsembleClassifier(
            base_model_constructor=KNeighborsClassifier, labels_per_model=n
        )
        ec.fit(X_tr_scaled, y_tr)
        results = ec.predict(X_te_scaled)
        print(f"{n=}, number of models in ensemble: {len(ec.models_)}")
        print(f"accuracy: {ec.score(X_te_scaled, y_te):.4f}")
    print("======================LogisticRegression=======================")
    for n in [1, 2]:
        ec = EnsembleClassifier(
            base_model_constructor=LogisticRegression, labels_per_model=1, max_iter=1000
        )
        ec.fit(X_tr_scaled, y_tr)
        results = ec.predict(X_te_scaled)
        print(f"{n=}, number of models in ensemble: {len(ec.models_)}")
        print(f"accuracy: {ec.score(X_te_scaled, y_te):.4f}")
