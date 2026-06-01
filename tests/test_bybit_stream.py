import pytest

from app.core.config import Settings
from app.schemas.market import Candle
from app.services.bybit_stream import BybitMarketStream
from app.services.market_data import MarketDataHub


def make_stream(settings: Settings | None = None) -> tuple[BybitMarketStream, MarketDataHub]:
    stream_settings = settings or Settings()
    hub = MarketDataHub(stream_settings)
    return BybitMarketStream(stream_settings, hub), hub


def make_candle(close: float = 100.0) -> Candle:
    return Candle(
        start_ms=60_000,
        end_ms=119_999,
        interval="1",
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10,
        timestamp_ms=60_000,
    )


def test_parse_http_candle_uses_interval_boundaries() -> None:
    stream, _ = make_stream(Settings(kline_interval="5"))

    candle = stream._parse_http_candle(
        ["1000", "100.0", "110.0", "95.0", "105.0", "3.5", "367.5"]
    )

    assert candle.start_ms == 1000
    assert candle.end_ms == 300_999
    assert candle.interval == "5"
    assert candle.close == 105.0
    assert candle.confirm is True


def test_parse_ws_candle_applies_defaults() -> None:
    stream, _ = make_stream()

    candle = stream._parse_ws_candle(
        {
            "start": "1000",
            "end": "60999",
            "open": "100",
            "high": "101",
            "low": "99",
            "close": "100.5",
        }
    )

    assert candle.volume == 0
    assert candle.turnover == 0
    assert candle.timestamp_ms == 60_999
    assert candle.confirm is False


def test_parse_ticker_uses_last_price_and_optional_fields() -> None:
    stream, _ = make_stream()

    ticker = stream._parse_ticker(
        {
            "symbol": "BTCUSDT",
            "lastPrice": "70000.5",
            "price24hPcnt": "0.0125",
            "highPrice24h": "71000",
            "lowPrice24h": "69000",
            "volume24h": "123.45",
            "turnover24h": "8642000",
        },
        timestamp_ms=123_456,
    )

    assert ticker.last_price == 70000.5
    assert ticker.price_24h_pct == 0.0125
    assert ticker.high_price_24h == 71000
    assert ticker.low_price_24h == 69000
    assert ticker.timestamp_ms == 123_456


def test_parse_ticker_falls_back_to_hub_latest_price(monkeypatch) -> None:
    stream, hub = make_stream()
    hub.replace_candles([make_candle(close=123.45)])
    monkeypatch.setattr("app.services.bybit_stream.time.time", lambda: 100.123)

    ticker = stream._parse_ticker({"symbol": "BTCUSDT"}, timestamp_ms=None)

    assert ticker.last_price == 123.45
    assert ticker.timestamp_ms == 100_123


def test_handle_ticker_message_records_status_for_invalid_payload() -> None:
    stream, hub = make_stream()

    stream._handle_ticker_message({"data": {"symbol": "BTCUSDT"}})

    status = hub.snapshot().status
    assert status.connected is True
    assert "ticker" in status.message
    assert "lastPrice" in status.message


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1", 1),
        ("15", 15),
        ("D", "D"),
    ],
)
def test_kline_stream_interval_preserves_non_numeric_intervals(interval, expected) -> None:
    stream, _ = make_stream(Settings(kline_interval=interval))

    assert stream._kline_stream_interval() == expected
