from types import SimpleNamespace

import pytest
from fastapi import HTTPException, WebSocketDisconnect

from app.api.routes import health, market_snapshot, market_websocket, predict
from app.core.config import Settings
from app.schemas.market import Candle, PredictionResponse, Ticker
from app.services.market_data import MarketDataHub


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StaticPredictor:
    def __init__(self) -> None:
        self.last_symbol: str | None = None

    def predict(self, snapshot):
        self.last_symbol = snapshot.symbol
        return PredictionResponse(
            symbol=snapshot.symbol,
            direction="UP",
            message="Прогнозируется рост цены Bitcoin",
            confidence=0.75,
            horizon_minutes=60,
            current_price=70000,
            model_name="Permutation Decision Tree",
            model_ready=True,
            generated_at_ms=1,
        )


class FailingPredictor:
    def predict(self, snapshot):
        raise ValueError("model is temporarily unavailable")


class FakeWebSocket:
    def __init__(self, app) -> None:
        self.app = app
        self.accepted = False
        self.sent_events: list[dict] = []
        self.subscriber_counts: list[int] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, event: dict) -> None:
        self.sent_events.append(event)
        self.subscriber_counts.append(len(self.app.state.market_hub._subscribers))
        raise WebSocketDisconnect()


def make_candle() -> Candle:
    return Candle(
        start_ms=60_000,
        end_ms=119_999,
        interval="1",
        open=100,
        high=110,
        low=90,
        close=105,
        volume=1,
        timestamp_ms=60_000,
    )


def make_app(predictor=None):
    settings = Settings()
    hub = MarketDataHub(settings)
    hub.replace_candles([make_candle()])
    hub.set_ticker(Ticker(symbol=settings.bybit_symbol, last_price=105, timestamp_ms=60_000))
    hub.set_status(True, "ready")

    return SimpleNamespace(
        state=SimpleNamespace(
            market_hub=hub,
            predictor=predictor or StaticPredictor(),
        )
    )


@pytest.mark.anyio
async def test_health_returns_stream_status() -> None:
    app = make_app()

    response = await health(SimpleNamespace(app=app))

    assert response == {
        "ok": True,
        "market_stream_connected": True,
        "message": "ready",
    }


@pytest.mark.anyio
async def test_market_snapshot_route_returns_current_hub_state() -> None:
    app = make_app()

    snapshot = await market_snapshot(SimpleNamespace(app=app))

    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.ticker is not None
    assert snapshot.ticker.last_price == 105
    assert snapshot.candles[0].close == 105


@pytest.mark.anyio
async def test_predict_route_returns_predictor_response() -> None:
    app = make_app()

    response = await predict(SimpleNamespace(app=app))

    assert response.direction == "UP"
    assert app.state.predictor.last_symbol == "BTCUSDT"


@pytest.mark.anyio
async def test_predict_route_maps_value_error_to_503() -> None:
    app = make_app(FailingPredictor())

    with pytest.raises(HTTPException) as exc_info:
        await predict(SimpleNamespace(app=app))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "model is temporarily unavailable"


@pytest.mark.anyio
async def test_market_websocket_sends_initial_snapshot_and_unsubscribes() -> None:
    app = make_app()
    websocket = FakeWebSocket(app)

    await market_websocket(websocket)

    assert websocket.accepted is True
    assert websocket.sent_events[0]["type"] == "snapshot"
    assert websocket.sent_events[0]["payload"]["symbol"] == "BTCUSDT"
    assert websocket.subscriber_counts == [1]
    assert len(app.state.market_hub._subscribers) == 0
