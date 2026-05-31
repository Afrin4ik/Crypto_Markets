import time

from app.schemas.market import MarketSnapshot, PredictionResponse


class PermutationDecisionTreePredictor:
    model_name = "Permutation Decision Trees"
    model_ready = False

    def predict(self, snapshot: MarketSnapshot) -> PredictionResponse:
        current_price = snapshot.ticker.last_price if snapshot.ticker else None
        candles = snapshot.candles
        if current_price is None and candles:
            current_price = candles[-1].close
        if current_price is None:
            raise ValueError("Нет актуальных рыночных данных для прогноза")

        direction = "up"
        if len(candles) >= 2 and candles[-1].close < candles[-2].close:
            direction = "down"

        recent = candles[-12:] if candles else []
        volatility = self._recent_volatility(recent, current_price)
        move = max(0.0015, min(volatility * 0.35, 0.008))
        multiplier = 1 + move if direction == "up" else 1 - move

        return PredictionResponse(
            symbol=snapshot.symbol,
            direction=direction,
            current_price=round(current_price, 2),
            target_price=round(current_price * multiplier, 2),
            model_name=self.model_name,
            model_ready=self.model_ready,
            generated_at_ms=int(time.time() * 1000),
        )

    @staticmethod
    def _recent_volatility(candles, current_price: float) -> float:
        if not candles or current_price <= 0:
            return 0.003
        ranges = [(candle.high - candle.low) / current_price for candle in candles]
        return sum(ranges) / len(ranges)
