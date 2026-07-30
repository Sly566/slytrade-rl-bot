from slytrade.core.config import load_config


def test_load_config():
    cfg = load_config()

    assert cfg.assets["execution_timeframe"] == "M1"
    assert "XAUUSD" in cfg.assets["primary"]
    assert cfg.data["tick_data"]["enabled"] is True
    assert cfg.data["bar_data"]["enabled"] is True
    assert cfg.risk["risk_per_trade"] > 0
