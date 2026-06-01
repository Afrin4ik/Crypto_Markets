from app.core.config import get_settings


def test_settings_reads_environment_when_created(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("BYBIT_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("BYBIT_TESTNET", "yes")
    monkeypatch.setenv("BYBIT_HISTORY_LIMIT", "42")

    settings = get_settings()

    assert settings.bybit_symbol == "ETHUSDT"
    assert settings.bybit_testnet is True
    assert settings.history_limit == 42
    get_settings.cache_clear()
