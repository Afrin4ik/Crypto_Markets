from app.schemas.market import PredictionResponse


def test_prediction_response_has_direction_contract_without_target_price() -> None:
    response = PredictionResponse(
        symbol="BTCUSDT",
        direction="UP",
        message="Прогнозируется рост цены Bitcoin",
        confidence=0.61,
        horizon_minutes=10,
        current_price=70000.0,
        model_name="Permutation Decision Tree",
        model_ready=True,
        generated_at_ms=1,
    )

    payload = response.model_dump()

    assert payload["direction"] == "UP"
    assert payload["message"] == "Прогнозируется рост цены Bitcoin"
    assert "target_price" not in payload
