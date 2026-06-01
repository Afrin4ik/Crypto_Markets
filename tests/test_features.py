import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, make_supervised_frame


def test_make_supervised_frame_has_features_after_warmup() -> None:
    rows = 90
    close = np.linspace(100.0, 120.0, rows)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.7,
            "close": close,
            "volume": np.linspace(10.0, 30.0, rows),
        }
    )

    supervised = make_supervised_frame(df, horizon=5)

    assert not supervised.empty
    assert not supervised[FEATURE_COLUMNS].isna().any().any()
    assert set(supervised["target"].unique()) <= {0, 1}
