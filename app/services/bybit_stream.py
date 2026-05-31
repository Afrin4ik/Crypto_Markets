import asyncio
from typing import Any

from pybit.unified_trading import HTTP, WebSocket

from app.core.config import Settings
from app.schemas.market import Candle, Ticker
from app.services.market_data import MarketDataHub


class BybitMarketStream:
    def __init__(self, settings: Settings, market_hub: MarketDataHub) -> None:
        self._settings = settings
        self._market_hub = market_hub
        self._http: HTTP | None = None
        self._ws: WebSocket | None = None
        self._running = False

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._market_hub.bind_loop(loop)
        await asyncio.to_thread(self._start_sync)

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_sync)

    def _start_sync(self) -> None:
        if self._running:
            return

        try:
            self._http = HTTP(testnet=self._settings.bybit_testnet)
            try:
                self._seed_history()
            except Exception as exc:
                self._market_hub.set_status(
                    False,
                    f"Не удалось загрузить исторические свечи Bybit: {exc}",
                )

            self._ws = WebSocket(
                testnet=self._settings.bybit_testnet,
                channel_type=self._settings.bybit_channel_type,
            )
            self._ws.kline_stream(
                interval=int(self._settings.kline_interval),
                symbol=self._settings.bybit_symbol,
                callback=self._handle_kline_message,
            )
            self._ws.ticker_stream(
                symbol=self._settings.bybit_symbol,
                callback=self._handle_ticker_message,
            )
            self._running = True
            self._market_hub.set_status(
                connected=True,
                message=f"Подключено к Bybit WebSocket: {self._settings.bybit_symbol}",
            )
        except Exception as exc:
            self._running = False
            self._market_hub.set_status(False, f"Ошибка подключения к Bybit: {exc}")

    def _stop_sync(self) -> None:
        if self._ws is not None:
            try:
                self._ws.exit()
            finally:
                self._ws = None
        self._running = False
        self._market_hub.set_status(False, "Поток Bybit остановлен")

    def _seed_history(self) -> None:
        if self._http is None:
            return

        response = self._http.get_kline(
            category=self._settings.bybit_category,
            symbol=self._settings.bybit_symbol,
            interval=self._settings.kline_interval,
            limit=self._settings.history_limit,
        )
        raw_candles = response.get("result", {}).get("list", [])
        candles = [self._parse_http_candle(item) for item in reversed(raw_candles)]
        self._market_hub.replace_candles(candles)

    def _handle_kline_message(self, message: dict[str, Any]) -> None:
        try:
            for item in message.get("data", []):
                self._market_hub.upsert_candle(self._parse_ws_candle(item))
        except Exception as exc:
            self._market_hub.set_status(True, f"Получено некорректное kline-сообщение: {exc}")

    def _handle_ticker_message(self, message: dict[str, Any]) -> None:
        try:
            data = message.get("data", {})
            if isinstance(data, list):
                data = data[0] if data else {}
            self._market_hub.set_ticker(self._parse_ticker(data, message.get("ts")))
        except Exception as exc:
            self._market_hub.set_status(True, f"Получено некорректное ticker-сообщение: {exc}")

    def _parse_http_candle(self, item: list[str]) -> Candle:
        start_ms = int(item[0])
        return Candle(
            start_ms=start_ms,
            end_ms=start_ms + self._settings.kline_interval_ms - 1,
            interval=self._settings.kline_interval,
            open=float(item[1]),
            high=float(item[2]),
            low=float(item[3]),
            close=float(item[4]),
            volume=float(item[5]),
            turnover=float(item[6]),
            confirm=True,
            timestamp_ms=start_ms,
        )

    def _parse_ws_candle(self, item: dict[str, Any]) -> Candle:
        return Candle(
            start_ms=int(item["start"]),
            end_ms=int(item["end"]),
            interval=str(item.get("interval", self._settings.kline_interval)),
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume=float(item.get("volume", 0) or 0),
            turnover=float(item.get("turnover", 0) or 0),
            confirm=bool(item.get("confirm", False)),
            timestamp_ms=int(item.get("timestamp") or item["end"]),
        )

    def _parse_ticker(self, data: dict[str, Any], timestamp_ms: int | None) -> Ticker:
        return Ticker(
            symbol=str(data.get("symbol", self._settings.bybit_symbol)),
            last_price=float(data["lastPrice"]),
            price_24h_pct=self._optional_float(data.get("price24hPcnt")),
            high_price_24h=self._optional_float(data.get("highPrice24h")),
            low_price_24h=self._optional_float(data.get("lowPrice24h")),
            volume_24h=self._optional_float(data.get("volume24h")),
            turnover_24h=self._optional_float(data.get("turnover24h")),
            timestamp_ms=int(timestamp_ms or 0),
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        return float(value)
