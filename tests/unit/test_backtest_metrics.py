from slytrade.backtest.metrics import compute_max_drawdown, compute_performance_metrics, compute_sharpe_like


def test_max_drawdown():
    assert compute_max_drawdown([100, 110, 99, 120]) == 0.1


def test_sharpe_like_flat_curve_zero():
    assert compute_sharpe_like([100, 100, 100]) == 0.0


def test_performance_metrics():
    metrics = compute_performance_metrics([100, 105, 110], trades=1)

    assert metrics.start_equity == 100
    assert metrics.final_equity == 110
    assert metrics.total_return == 0.1
    assert metrics.trades == 1
