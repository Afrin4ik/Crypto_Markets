import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Bitcoin PDT Forecast Dashboard"
    bybit_symbol: str = field(default_factory=lambda: os.getenv("BYBIT_SYMBOL", "BTCUSDT"))
    bybit_category: str = field(default_factory=lambda: os.getenv("BYBIT_CATEGORY", "spot"))
    bybit_channel_type: str = field(
        default_factory=lambda: os.getenv("BYBIT_CHANNEL_TYPE", "spot")
    )
    bybit_testnet: bool = field(default_factory=lambda: _env_bool("BYBIT_TESTNET", False))
    kline_interval: str = field(default_factory=lambda: os.getenv("BYBIT_KLINE_INTERVAL", "1"))
    history_limit: int = field(default_factory=lambda: _env_int("BYBIT_HISTORY_LIMIT", 180))
    templates_dir: Path = BASE_DIR / "app" / "templates"
    static_dir: Path = BASE_DIR / "app" / "static"
    pdt_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PDT_MODEL_PATH", str(BASE_DIR / "models" / "pdt_btc_direction.joblib"))
        )
    )
    pdt_metadata_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PDT_METADATA_PATH",
                str(BASE_DIR / "models" / "pdt_btc_direction_metadata.json"),
            )
        )
    )
    pdt_horizon_minutes: int = field(default_factory=lambda: _env_int("PDT_HORIZON_MINUTES", 10))
    pdt_max_rolling_window: int = field(
        default_factory=lambda: _env_int("PDT_MAX_ROLLING_WINDOW", 60)
    )
    pdt_min_candles: int = field(default_factory=lambda: _env_int("PDT_MIN_CANDLES", 80))

    @property
    def kline_interval_ms(self) -> int:
        if self.kline_interval.isdigit():
            return int(self.kline_interval) * 60_000
        return 60_000


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом, получено: {value!r}") from exc


@lru_cache
def get_settings() -> Settings:
    return Settings()
