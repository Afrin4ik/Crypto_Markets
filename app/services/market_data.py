import asyncio
import time
from collections import OrderedDict
from threading import RLock

from app.core.config import Settings
from app.schemas.market import Candle, MarketSnapshot, StreamStatus, Ticker


class MarketDataHub:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._candles: OrderedDict[int, Candle] = OrderedDict()
        self._ticker: Ticker | None = None
        self._status = StreamStatus()
        self._lock = RLock()
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def snapshot(self) -> MarketSnapshot:
        with self._lock:
            return MarketSnapshot(
                symbol=self._settings.bybit_symbol,
                interval=self._settings.kline_interval,
                candles=list(self._candles.values()),
                ticker=self._ticker,
                status=self._status,
            )

    def latest_price(self) -> float | None:
        with self._lock:
            if self._ticker is not None:
                return self._ticker.last_price
            if self._candles:
                return next(reversed(self._candles.values())).close
            return None

    def replace_candles(self, candles: list[Candle]) -> None:
        with self._lock:
            self._candles.clear()
            for candle in sorted(candles, key=lambda item: item.start_ms):
                self._candles[candle.start_ms] = candle
            self._trim_candles()
            snapshot = self.snapshot()
        self._publish("snapshot", snapshot.model_dump(mode="json"))

    def upsert_candle(self, candle: Candle) -> None:
        with self._lock:
            self._candles[candle.start_ms] = candle
            self._candles = OrderedDict(sorted(self._candles.items()))
            self._trim_candles()
        self._publish("candle", candle.model_dump(mode="json"))

    def set_ticker(self, ticker: Ticker) -> None:
        with self._lock:
            self._ticker = ticker
        self._publish("ticker", ticker.model_dump(mode="json"))

    def set_status(self, connected: bool, message: str) -> None:
        status = StreamStatus(
            connected=connected,
            message=message,
            updated_at_ms=int(time.time() * 1000),
        )
        with self._lock:
            self._status = status
        self._publish("status", status.model_dump(mode="json"))

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(queue)

    def _trim_candles(self) -> None:
        while len(self._candles) > self._settings.history_limit:
            self._candles.popitem(last=False)

    def _publish(self, event_type: str, payload: dict) -> None:
        event = {"type": event_type, "payload": payload}
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(self._publish_in_loop, event)

    def _publish_in_loop(self, event: dict) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)
