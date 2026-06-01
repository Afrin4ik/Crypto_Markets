import joblib

from app.core.config import Settings
from app.schemas.market import Candle, MarketSnapshot, Ticker
from app.services.prediction import PermutationDecisionTreePredictor


class InvalidModel:
    pass


class UpModel:
    def predict(self, features):
        return ["UP"]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


def make_candles(count: int = 100) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        close = 70_000 + index
        start_ms = index * 60_000
        candles.append(
            Candle(
                start_ms=start_ms,
                end_ms=start_ms + 59_999,
                interval="1",
                open=close - 5,
                high=close + 10,
                low=close - 10,
                close=close,
                volume=10 + index,
                timestamp_ms=start_ms,
                confirm=True,
            )
        )
    return candles


def test_predictor_reports_invalid_model_artifact(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump({"model": InvalidModel()}, model_path)

    predictor = PermutationDecisionTreePredictor(
        Settings(pdt_model_path=model_path, pdt_metadata_path=metadata_path)
    )
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        interval="1",
        ticker=Ticker(symbol="BTCUSDT", last_price=70000.0, timestamp_ms=1),
    )

    response = predictor.predict(snapshot)

    assert response.model_ready is False
    assert "predict" in response.message


def test_predictor_returns_forecast_with_loaded_artifact_and_metadata(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump({"model": UpModel(), "horizon_minutes": 30}, model_path)
    metadata_path.write_text('{"horizon_minutes": 45}', encoding="utf-8")
    candles = make_candles()

    predictor = PermutationDecisionTreePredictor(
        Settings(
            pdt_model_path=model_path,
            pdt_metadata_path=metadata_path,
            pdt_min_candles=80,
        )
    )
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        interval="1",
        candles=candles,
        ticker=Ticker(symbol="BTCUSDT", last_price=70123.456, timestamp_ms=1),
    )

    response = predictor.predict(snapshot)

    assert response.model_ready is True
    assert response.direction == "UP"
    assert response.confidence == 0.8
    assert response.horizon_minutes == 45
    assert response.current_price == 70123.46


def test_predictor_reports_missing_market_data_without_loading_features(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump({"model": UpModel()}, model_path)
    predictor = PermutationDecisionTreePredictor(
        Settings(pdt_model_path=model_path, pdt_metadata_path=metadata_path)
    )

    response = predictor.predict(MarketSnapshot(symbol="BTCUSDT", interval="1"))

    assert response.model_ready is False
    assert "Нет актуальных рыночных данных" in response.message
