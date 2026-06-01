import json
import time
from typing import Any

import joblib

from app.core.config import Settings, get_settings
from app.schemas.market import MarketSnapshot, PredictionResponse
from ml.features import FEATURE_COLUMNS, candles_to_dataframe, feature_matrix, build_feature_frame


class PermutationDecisionTreePredictor:
    model_name = "Permutation Decision Tree"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any | None = None
        self._feature_columns: list[str] = list(FEATURE_COLUMNS)
        self._metadata: dict[str, Any] = {}
        self._horizon_minutes = self._settings.pdt_horizon_minutes
        self._loaded = False
        self._load_error: str | None = None

    def predict(self, snapshot: MarketSnapshot) -> PredictionResponse:
        current_price = snapshot.ticker.last_price if snapshot.ticker else None
        candles = snapshot.candles
        if current_price is None and candles:
            current_price = candles[-1].close

        self._load_model()
        if self._model is None:
            return self._not_ready(
                snapshot,
                current_price,
                self._load_error
                or "PDT-модель не найдена. Обучите её командой: python -m ml.train_pdt",
            )

        if current_price is None:
            return self._not_ready(snapshot, current_price, "Нет актуальных рыночных данных для прогноза")

        if len(candles) < self._settings.pdt_min_candles:
            return self._not_ready(
                snapshot,
                current_price,
                (
                    "Недостаточно свечей для PDT-признаков: "
                    f"нужно минимум {self._settings.pdt_min_candles}, сейчас {len(candles)}"
                ),
            )

        try:
            candle_frame = candles_to_dataframe(candles)
            feature_frame = build_feature_frame(candle_frame, dropna=True)
        except Exception as exc:
            return self._not_ready(snapshot, current_price, f"Не удалось построить PDT-признаки: {exc}")

        if feature_frame.empty:
            return self._not_ready(
                snapshot,
                current_price,
                "Недостаточно истории для rolling-признаков PDT",
            )

        features = feature_matrix(feature_frame.tail(1), self._feature_columns)
        direction = str(self._model.predict(features)[0])
        probabilities = self._model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        message = (
            "Прогнозируется рост цены Bitcoin"
            if direction == "UP"
            else "Прогнозируется падение цены Bitcoin"
        )

        return PredictionResponse(
            symbol=snapshot.symbol,
            direction=direction,
            message=message,
            confidence=confidence,
            horizon_minutes=self._horizon_minutes,
            current_price=round(current_price, 2),
            model_name=self.model_name,
            model_ready=True,
            generated_at_ms=int(time.time() * 1000),
        )

    def _load_model(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        model_path = self._settings.pdt_model_path
        if not model_path.exists():
            self._load_error = (
                f"PDT-модель не найдена по пути {model_path}. "
                "Обучите её командой: python -m ml.train_pdt"
            )
            return

        try:
            artifact = joblib.load(model_path)
            if isinstance(artifact, dict):
                self._model = artifact.get("model")
                self._feature_columns = list(artifact.get("feature_columns") or FEATURE_COLUMNS)
                self._horizon_minutes = int(
                    artifact.get("horizon_minutes") or self._settings.pdt_horizon_minutes
                )
            else:
                self._model = artifact
                self._feature_columns = list(FEATURE_COLUMNS)
            if self._model is None:
                raise ValueError("artifact не содержит ключ model")
            self._metadata = self._load_metadata()
            self._horizon_minutes = int(
                self._metadata.get("horizon_minutes", self._horizon_minutes)
            )
        except Exception as exc:
            self._model = None
            self._load_error = f"Не удалось загрузить PDT-модель: {exc}"

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self._settings.pdt_metadata_path
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _not_ready(
        self,
        snapshot: MarketSnapshot,
        current_price: float | None,
        message: str,
    ) -> PredictionResponse:
        return PredictionResponse(
            symbol=snapshot.symbol,
            direction=None,
            message=message,
            confidence=None,
            horizon_minutes=self._horizon_minutes,
            current_price=round(current_price, 2) if current_price is not None else None,
            model_name=self.model_name,
            model_ready=False,
            generated_at_ms=int(time.time() * 1000),
        )
