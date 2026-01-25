import numpy as np
import numpy.typing as npt
from typing import Dict, Hashable, Literal, get_args, get_type_hints


def find_typical_class_member(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    method: Literal["medoid", "centroid"] = "centroid",
) -> Dict[Hashable, int]:
    X = np.asarray(X)
    y = np.asarray(y)

    assert len(X) == len(y)

    typical_idxn = {}
    for label in np.unique(y):
        idxn = np.flatnonzero(y == label)
        X_cl = X[idxn]

        match method:
            case "centroid":
                c = X_cl.mean(axis=0)
                dists = np.linalg.norm(X_cl - c, axis=1)
            case "medoid":
                diff = X_cl[:, None, :] - X_cl[None, :, :]
                dists = np.linalg.norm(diff, axis=2).sum(axis=1)
            case _:
                raise ValueError(
                    f"method should be one of {", ".join(get_args(get_type_hints(find_typical_class_member)["method"]))}"
                )
        typical_idxn[label] = idxn[np.argmin(dists)]
    return typical_idxn


if __name__ == "__main__":
    from collections import defaultdict
    import pandas as pd

    df = pd.read_csv("Data/features_30_sec.csv")
    results = []
    X = df.drop(["filename", "length", "label"], axis=1)
    y = df["label"]
    for method in get_args(get_type_hints(find_typical_class_member)["method"]):
        print(method)
        for label, idx in find_typical_class_member(X=X, y=y, method=method).items():
            results.append({
                           "method": method,
                           "label": label,
                           "filename": df.iloc[idx]["filename"]
            })
    print(pd.DataFrame(results))
