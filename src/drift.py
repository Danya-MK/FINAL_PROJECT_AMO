import numpy as np
import pandas as pd


def _psi_for_series(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0

    expected_bins = pd.cut(expected, bins=cuts, include_lowest=True)
    actual_bins = pd.cut(actual, bins=cuts, include_lowest=True)

    e_perc = expected_bins.value_counts(normalize=True).sort_index()
    a_perc = actual_bins.value_counts(normalize=True).sort_index()
    a_perc = a_perc.reindex(e_perc.index).fillna(0.0)

    eps = 1e-6
    e = np.clip(e_perc.values, eps, 1.0)
    a = np.clip(a_perc.values, eps, 1.0)

    return float(np.sum((a - e) * np.log(a / e)))


def compute_psi(train_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str], buckets: int = 10):
    per_feature = {}
    for f in features:
        per_feature[f] = _psi_for_series(train_df[f], current_df[f], buckets=buckets)

    total = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    return total, per_feature