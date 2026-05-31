from typing import Literal

from pydantic import BaseModel, Field


class Candle(BaseModel):
    start_ms: int
    end_ms: int
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    turnover: float = 0.0
    confirm: bool = False
    timestamp_ms: int


class Ticker(BaseModel):
    symbol: str
    last_price: float
    price_24h_pct: float | None = None
    high_price_24h: float | None = None
    low_price_24h: float | None = None
    volume_24h: float | None = None
    turnover_24h: float | None = None
    timestamp_ms: int


class StreamStatus(BaseModel):
    connected: bool = False
    message: str = "Поток рыночных данных ещё не запущен"
    updated_at_ms: int | None = None


class MarketSnapshot(BaseModel):
    symbol: str
    interval: str
    candles: list[Candle] = Field(default_factory=list)
    ticker: Ticker | None = None
    status: StreamStatus = Field(default_factory=StreamStatus)


class MarketEvent(BaseModel):
    type: Literal["snapshot", "candle", "ticker", "status"]
    payload: MarketSnapshot | Candle | Ticker | StreamStatus


class PredictionResponse(BaseModel):
    symbol: str
    direction: Literal["up", "down"]
    current_price: float
    target_price: float
    model_name: str
    model_ready: bool
    generated_at_ms: int
