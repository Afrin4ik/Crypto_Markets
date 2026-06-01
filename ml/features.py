from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


FEATURE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "candle_body",
    "candle_range",
    "upper_shadow",
    "lower_shadow",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_10",
    "ema_20",
    "ema_50",
    "sma_10_20_diff",
    "sma_20_50_diff",
    "ema_10_20_diff",
    "ema_20_50_diff",
    "close_sma_10_ratio",
    "close_sma_20_ratio",
    "close_sma_50_ratio",
    "rsi_14",
    "volatility_10",
    "volatility_30",
    "volume_change",
    "volume_mean_ratio_10",
    "volume_mean_ratio_30",
]

MAX_ROLLING_WINDOW = 50

_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp",
        "datetime",
        "date",
        "time",
        "open_time",
        "start_time",
        "unix",
    ),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "volume_btc", "volume_base", "vol"),
}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load a candle CSV and normalize it to timestamp/open/high/low/close/volume."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    skiprows = 1 if first_line and "," not in first_line else 0
    raw = pd.read_csv(csv_path, skiprows=skiprows)
    return normalize_ohlcv(raw)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    columns = {_clean_column_name(column): column for column in df.columns}
    selected: dict[str, str] = {}
    missing: list[str] = []

    for target, candidates in _COLUMN_CANDIDATES.items():
        source = next((columns[name] for name in candidates if name in columns), None)
        if source is None:
            missing.append(target)
        else:
            selected[target] = source

    if missing:
        available = ", ".join(str(column) for column in df.columns)
        raise ValueError(f"Не найдены OHLCV-колонки {missing}. Доступные колонки: {available}")

    normalized = pd.DataFrame({target: df[source] for target, source in selected.items()})
    normalized["timestamp"] = _parse_timestamp(normalized["timestamp"])
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    return normalized


def candles_to_dataframe(candles: Iterable[object]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for candle in candles:
        if isinstance(candle, Mapping):
            getter = candle.get
        else:
            getter = lambda name, default=None: getattr(candle, name, default)
        rows.append(
            {
                "timestamp": getter("start_ms", getter("timestamp_ms")),
                "open": getter("open"),
                "high": getter("high"),
                "low": getter("low"),
                "close": getter("close"),
                "volume": getter("volume", 0.0),
            }
        )
    return normalize_ohlcv(pd.DataFrame(rows))


def build_feature_frame(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    data = normalize_ohlcv(df)
    data = data.copy()

    open_price = data["open"]
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]
    returns = close.pct_change()

    for window in (1, 3, 5, 10):
        data[f"return_{window}"] = close.pct_change(window)

    data["candle_body"] = _safe_divide(close - open_price, open_price)
    data["candle_range"] = _safe_divide(high - low, close)
    data["upper_shadow"] = _safe_divide(high - np.maximum(open_price, close), close)
    data["lower_shadow"] = _safe_divide(np.minimum(open_price, close) - low, close)

    for window in (10, 20, 50):
        data[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()
        data[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()

    data["sma_10_20_diff"] = _safe_divide(data["sma_10"] - data["sma_20"], close)
    data["sma_20_50_diff"] = _safe_divide(data["sma_20"] - data["sma_50"], close)
    data["ema_10_20_diff"] = _safe_divide(data["ema_10"] - data["ema_20"], close)
    data["ema_20_50_diff"] = _safe_divide(data["ema_20"] - data["ema_50"], close)
    data["close_sma_10_ratio"] = _safe_divide(close, data["sma_10"]) - 1.0
    data["close_sma_20_ratio"] = _safe_divide(close, data["sma_20"]) - 1.0
    data["close_sma_50_ratio"] = _safe_divide(close, data["sma_50"]) - 1.0

    data["rsi_14"] = compute_rsi(close, period=14)
    data["volatility_10"] = returns.rolling(window=10, min_periods=10).std()
    data["volatility_30"] = returns.rolling(window=30, min_periods=30).std()
    data["volume_change"] = volume.pct_change()
    data["volume_mean_ratio_10"] = _safe_divide(
        volume, volume.rolling(window=10, min_periods=10).mean()
    ) - 1.0
    data["volume_mean_ratio_30"] = _safe_divide(
        volume, volume.rolling(window=30, min_periods=30).mean()
    ) - 1.0

    data = data.replace([np.inf, -np.inf], np.nan)
    if dropna:
        data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return data


def make_supervised_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon должен быть положительным числом")

    data = build_feature_frame(df, dropna=False)
    future_close = data["close"].shift(-horizon)
    data["target"] = (future_close > data["close"]).astype("float")
    data.loc[future_close.isna(), "target"] = np.nan
    data["previous_direction"] = (data["close"].diff() > 0).astype("float")
    data.loc[data["close"].diff().isna(), "previous_direction"] = np.nan
    data = data.dropna(subset=FEATURE_COLUMNS + ["target", "previous_direction"])
    data["target"] = data["target"].astype(int)
    data["previous_direction"] = data["previous_direction"].astype(int)
    return data.reset_index(drop=True)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi.fillna(50.0)


def feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> np.ndarray:
    columns = list(feature_columns or FEATURE_COLUMNS)
    return frame[columns].to_numpy(dtype=float)


def _clean_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0 and numeric.notna().sum() >= series.notna().sum() * 0.8:
        max_value = numeric.max()
        unit = "ms" if max_value and max_value > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)
