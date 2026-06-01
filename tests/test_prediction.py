import joblib

from app.core.config import Settings
from app.schemas.market import MarketSnapshot, Ticker
from app.services.prediction import PermutationDecisionTreePredictor


class InvalidModel:
    pass


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
