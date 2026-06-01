import asyncio

import pytest

from app.core.config import Settings
from app.schemas.market import Candle, Ticker
from app.services.market_data import MarketDataHub


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_candle(start_ms: int, close: float) -> Candle:
    return Candle(
        start_ms=start_ms,
        end_ms=start_ms + 59_999,
        interval="1",
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=10,
        timestamp_ms=start_ms,
    )


def test_replace_candles_sorts_and_trims_to_history_limit() -> None:
    hub = MarketDataHub(Settings(history_limit=3))

    hub.replace_candles(
        [
            make_candle(180_000, 103),
            make_candle(60_000, 101),
            make_candle(120_000, 102),
            make_candle(240_000, 104),
        ]
    )

    snapshot = hub.snapshot()

    assert [candle.start_ms for candle in snapshot.candles] == [120_000, 180_000, 240_000]
    assert [candle.close for candle in snapshot.candles] == [102, 103, 104]
    assert hub.latest_price() == 104


def test_upsert_candle_updates_existing_candle_and_preserves_order() -> None:
    hub = MarketDataHub(Settings(history_limit=5))
    hub.replace_candles([make_candle(60_000, 101), make_candle(180_000, 103)])

    hub.upsert_candle(make_candle(120_000, 202))
    hub.upsert_candle(make_candle(180_000, 303))

    snapshot = hub.snapshot()

    assert [candle.start_ms for candle in snapshot.candles] == [60_000, 120_000, 180_000]
    assert [candle.close for candle in snapshot.candles] == [101, 202, 303]


def test_latest_price_prefers_ticker_over_latest_candle() -> None:
    hub = MarketDataHub(Settings())
    hub.replace_candles([make_candle(60_000, 101)])

    hub.set_ticker(Ticker(symbol="BTCUSDT", last_price=999.5, timestamp_ms=123))

    assert hub.latest_price() == 999.5


@pytest.mark.anyio
async def test_publish_sends_events_to_subscribers() -> None:
    hub = MarketDataHub(Settings())
    hub.bind_loop(asyncio.get_running_loop())
    queue = hub.subscribe()

    hub.set_status(True, "connected")

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["type"] == "status"
    assert event["payload"]["connected"] is True
    assert event["payload"]["message"] == "connected"

    hub.unsubscribe(queue)
