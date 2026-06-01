import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Bitcoin PDT Forecast Dashboard"
    bybit_symbol: str = os.getenv("BYBIT_SYMBOL", "BTCUSDT")
    bybit_category: str = os.getenv("BYBIT_CATEGORY", "spot")
    bybit_channel_type: str = os.getenv("BYBIT_CHANNEL_TYPE", "spot")
    bybit_testnet: bool = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    kline_interval: str = os.getenv("BYBIT_KLINE_INTERVAL", "1")
    history_limit: int = int(os.getenv("BYBIT_HISTORY_LIMIT", "180"))
    templates_dir: Path = BASE_DIR / "app" / "templates"
    static_dir: Path = BASE_DIR / "app" / "static"
    pdt_model_path: Path = Path(
        os.getenv("PDT_MODEL_PATH", str(BASE_DIR / "models" / "pdt_btc_direction.joblib"))
    )
    pdt_metadata_path: Path = Path(
        os.getenv(
            "PDT_METADATA_PATH",
            str(BASE_DIR / "models" / "pdt_btc_direction_metadata.json"),
        )
    )
    pdt_horizon_minutes: int = int(os.getenv("PDT_HORIZON_MINUTES", "10"))
    pdt_max_rolling_window: int = int(os.getenv("PDT_MAX_ROLLING_WINDOW", "50"))
    pdt_min_candles: int = int(os.getenv("PDT_MIN_CANDLES", "60"))

    @property
    def kline_interval_ms(self) -> int:
        if self.kline_interval.isdigit():
            return int(self.kline_interval) * 60_000
        return 60_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
