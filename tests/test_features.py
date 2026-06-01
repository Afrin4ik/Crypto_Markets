import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_COLUMNS,
    candles_to_dataframe,
    load_ohlcv_csv,
    make_supervised_frame,
    normalize_ohlcv,
)


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


def test_normalize_ohlcv_accepts_aliases_sorts_and_deduplicates() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2026-01-01 00:01", "2026-01-01 00:00", "2026-01-01 00:01"],
            "Open": [101, 100, 201],
            "High": [102, 101, 202],
            "Low": [100, 99, 200],
            "Close": [101.5, 100.5, 201.5],
            "Volume BTC": [11, 10, 21],
        }
    )

    normalized = normalize_ohlcv(raw)

    assert normalized["close"].tolist() == [100.5, 201.5]
    assert normalized["volume"].tolist() == [10, 21]
    assert normalized["timestamp"].is_monotonic_increasing


def test_candles_to_dataframe_accepts_mapping_candles() -> None:
    frame = candles_to_dataframe(
        [
            {
                "start_ms": 1_767_225_600_000,
                "open": 100,
                "high": 105,
                "low": 95,
                "close": 101,
                "volume": 10,
            }
        ]
    )

    assert frame.loc[0, "close"] == 101
    assert frame.loc[0, "volume"] == 10


def test_load_ohlcv_csv_skips_exchange_title_line(tmp_path) -> None:
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Binance BTCUSDT minute candles",
                "timestamp,open,high,low,close,volume",
                "2026-01-01 00:00:00,100,101,99,100.5,12",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_ohlcv_csv(csv_path)

    assert len(frame) == 1
    assert frame.loc[0, "close"] == 100.5


def test_make_supervised_frame_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        make_supervised_frame(pd.DataFrame(), horizon=0)
