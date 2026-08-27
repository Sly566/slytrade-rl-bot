"""Tests for Layer 3 MTF causal alignment."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slytrade.config import DataConfig
from slytrade.data.features import FeatureConfig, process_bars
from slytrade.data.mtf_align import (
    _HTF_DROP_COLS as HTF_DROP_COLS,
)
from slytrade.data.mtf_align import (
    HTFS,
    _asof_merge,
    _prep_htf_frame,
    align_all,
    inspect_aligned,
)
from slytrade.data.storage import _atomic_write_parquet, _normalize_for_parquet, bar_partition

UTC = UTC


def _make_ohlcv(start: datetime, n: int, freq_minutes: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV bars as a deterministic random walk."""
    rng = np.random.default_rng(seed)
    times = [start + timedelta(minutes=freq_minutes * i) for i in range(n)]
    rets = rng.normal(0.0, 0.001, size=n)
    close = 2500.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, n)))
    opn = close * (1 + rng.normal(0, 0.0003, n))
    vol = rng.integers(50, 500, n)
    return pd.DataFrame({
        "time": times,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": vol.astype(np.float64),
        "spread": np.full(n, 10, dtype=np.int32),
        "real_volume": np.zeros(n, dtype=np.float64),
    })


def _write_processed_bars(root: Path, symbol: str, tf: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for (y, m), grp in df.groupby([df["time"].dt.year, df["time"].dt.month]):
        d = bar_partition(root, "", symbol, tf, int(y), int(m))
        _atomic_write_parquet(_normalize_for_parquet(grp, time_col="time"), d / "part-0.parquet")


class TestPrepHtfFrame:
    def test_drops_expected_columns(self):
        """After prep, dropped columns must not appear with prefix."""
        raw = _make_ohlcv(datetime(2025, 1, 1, tzinfo=UTC), 100, 5)
        cfg = FeatureConfig()
        proc = process_bars(raw, "M5", cfg)
        proc["decision_time"] = proc["time"] + timedelta(minutes=5)
        prepped = _prep_htf_frame(proc, "M5")
        for col in HTF_DROP_COLS:
            assert f"M5_{col}" not in prepped.columns, f"Column {col} should have been dropped"
        assert "M5_bar_time" in prepped.columns
        assert "M5_close" in prepped.columns
        assert "decision_time" in prepped.columns
        for suffix in ("swing_high_idx", "swing_low_idx", "ob_idx", "fvg_idx"):
            assert f"M5_minor_{suffix}" not in prepped.columns
            assert f"M5_major_{suffix}" not in prepped.columns

    def test_empty_frame(self):
        empty = pd.DataFrame()
        out = _prep_htf_frame(empty, "M5")
        assert list(out.columns) == ["decision_time"]
        assert len(out) == 0


class TestCausalMerge:
    """Verify strict causality: HTF info at M1 bar t must come only
    from HTF bars whose decision_time (close) <= t."""

    def test_exact_boundary_match(self):
        """M5 bar [10:00-10:05] closes at 10:05; M1 bar 10:05 opens at 10:05.
        The M5 bar SHOULD be visible at M1 10:05 (exact match allowed)."""
        m1_times = pd.to_datetime([
            "2025-01-01 10:00", "2025-01-01 10:01", "2025-01-01 10:02",
            "2025-01-01 10:03", "2025-01-01 10:04", "2025-01-01 10:05",
            "2025-01-01 10:06", "2025-01-01 10:07", "2025-01-01 10:08",
            "2025-01-01 10:09", "2025-01-01 10:10",
        ], utc=True)
        m1 = pd.DataFrame({
            "time": m1_times,
            "open": np.zeros(len(m1_times)),
            "high": np.zeros(len(m1_times)),
            "low": np.zeros(len(m1_times)),
            "close": np.zeros(len(m1_times)),
            "tick_volume": np.full(len(m1_times), 100.0),
            "spread": np.full(len(m1_times), 10),
            "real_volume": np.zeros(len(m1_times)),
        })
        m5_times = pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:05"], utc=True)
        m5 = pd.DataFrame({
            "time": m5_times,
            "open": [0.0, 0.0],
            "high": [0.0, 0.0],
            "low": [0.0, 0.0],
            "close": [100.0, 200.0],
            "tick_volume": [100.0, 100.0],
            "spread": [10, 10],
            "real_volume": [0.0, 0.0],
        })
        cfg = FeatureConfig(ema_slow=5)
        m1_proc = process_bars(m1, "M1", cfg)
        m5_proc = process_bars(m5, "M5", cfg)
        m5_proc["decision_time"] = m5_proc["time"] + timedelta(minutes=5)
        prepped = _prep_htf_frame(m5_proc, "M5")
        merged = _asof_merge(m1_proc, prepped, "M5")

        # M1 bars 10:00-10:04 should NOT yet see the 10:00-10:05 M5 bar.
        for i in range(5):
            assert pd.isna(merged.loc[i, "M5_close"]), (
                f"M1 bar at {m1_times[i]} should NOT see M5 close yet, got {merged.loc[i, 'M5_close']}"
            )
        # M1 bar 10:05 should see M5_close=100 (bar that closed exactly at 10:05).
        idx_1005 = m1_times.get_loc(pd.Timestamp("2025-01-01 10:05", tz=UTC))
        assert merged.loc[idx_1005, "M5_close"] == pytest.approx(100.0)
        assert merged.loc[idx_1005, "M5_bar_time"] == pd.Timestamp("2025-01-01 10:00", tz=UTC)
        # M1 bars 10:06-10:09 still see close=100.
        for i in range(idx_1005 + 1, len(m1_times) - 1):
            assert merged.loc[i, "M5_close"] == pytest.approx(100.0)
        # M1 10:10 sees M5_close=200 (second M5 bar closed exactly at 10:10).
        idx_1010 = m1_times.get_loc(pd.Timestamp("2025-01-01 10:10", tz=UTC))
        assert merged.loc[idx_1010, "M5_close"] == pytest.approx(200.0)

    def test_no_future_leak(self):
        """HTF bar that has NOT closed yet must never appear on M1."""
        m1_times = pd.to_datetime([
            "2025-01-01 10:05", "2025-01-01 10:06", "2025-01-01 10:07",
            "2025-01-01 10:08", "2025-01-01 10:09",
        ], utc=True)
        m1 = pd.DataFrame({
            "time": m1_times,
            "open": np.zeros(len(m1_times)), "high": np.zeros(len(m1_times)),
            "low": np.zeros(len(m1_times)), "close": np.zeros(len(m1_times)),
            "tick_volume": np.full(len(m1_times), 100.0),
            "spread": np.full(len(m1_times), 10),
            "real_volume": np.zeros(len(m1_times)),
        })
        m5_times = pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:05"], utc=True)
        m5 = pd.DataFrame({
            "time": m5_times,
            "open": [0.0, 0.0], "high": [0.0, 0.0], "low": [0.0, 0.0],
            "close": [100.0, 999.0],
            "tick_volume": [100.0, 100.0], "spread": [10, 10], "real_volume": [0.0, 0.0],
        })
        cfg = FeatureConfig(ema_slow=5)
        m1_proc = process_bars(m1, "M1", cfg)
        m5_proc = process_bars(m5, "M5", cfg)
        m5_proc["decision_time"] = m5_proc["time"] + timedelta(minutes=5)
        prepped = _prep_htf_frame(m5_proc, "M5")
        merged = _asof_merge(m1_proc, prepped, "M5")
        # The second M5 bar (open 10:05) closes at 10:10 — after all M1 bars.
        # Every M1 row must therefore see M5_close=100, NOT 999.
        for i in range(len(merged)):
            assert merged.loc[i, "M5_close"] == pytest.approx(100.0), (
                f"M1 at {merged.loc[i, 'time']} saw future M5_close={merged.loc[i, 'M5_close']}"
            )


class TestAlignAllEndToEnd:
    def test_full_pipeline(self, tmp_path):
        data_cfg = DataConfig(raw_root=tmp_path / "raw", processed_root=tmp_path / "processed")
        sym = "TEST"
        start = datetime(2025, 1, 1, tzinfo=UTC)
        for tf, mins in [("M1", 1), ("M5", 5), ("M15", 15), ("M30", 30),
                          ("H1", 60), ("H4", 240), ("D1", 1440), ("W1", 10080)]:
            n = int(3 * 1440 / mins) + 300
            raw = _make_ohlcv(start, n, mins, seed=hash(tf) & 0xffff)
            proc = process_bars(raw, tf, FeatureConfig())
            _write_processed_bars(data_cfg.processed_bars_path, sym, tf, proc)

        res = align_all(data_cfg, symbol="TEST", raw_symbol=sym)
        assert res.rows > 0
        assert res.files > 0
        assert res.columns > 91
        for tf in HTFS:
            assert res.per_tf_cols[tf] > 50

        diag = inspect_aligned(data_cfg, sym)
        assert diag.rows == res.rows
        assert diag.files == res.files
        assert diag.columns == res.columns
        assert not diag.issues, f"Unexpected issues: {diag.issues}"

    def test_missing_m1_is_graceful(self, tmp_path):
        data_cfg = DataConfig(raw_root=tmp_path / "raw", processed_root=tmp_path / "processed")
        res = align_all(data_cfg, symbol="X", raw_symbol="X")
        assert res.rows == 0
        assert res.files == 0

    def test_missing_htf_is_graceful(self, tmp_path):
        """If only M1 is present, alignment still runs (HTFs produce NaN)."""
        data_cfg = DataConfig(raw_root=tmp_path / "raw", processed_root=tmp_path / "processed")
        sym = "TEST"
        start = datetime(2025, 1, 1, tzinfo=UTC)
        raw = _make_ohlcv(start, 2000, 1, seed=1)
        proc = process_bars(raw, "M1", FeatureConfig())
        _write_processed_bars(data_cfg.processed_bars_path, sym, "M1", proc)
        res = align_all(data_cfg, symbol="TEST", raw_symbol=sym)
        assert res.rows == 2000
        assert res.columns == 91


class TestCausalityEndToEnd:
    """Full pipeline causality check across all HTFs."""

    def test_no_future_leak_full_pipeline(self, tmp_path):
        data_cfg = DataConfig(raw_root=tmp_path / "raw", processed_root=tmp_path / "processed")
        sym = "TEST"
        start = datetime(2025, 1, 1, tzinfo=UTC)
        for tf, mins in [("M1", 1), ("M5", 5), ("M15", 15), ("M30", 30),
                          ("H1", 60), ("H4", 240), ("D1", 1440), ("W1", 10080)]:
            n = int(5 * 1440 / mins) + 300
            raw = _make_ohlcv(start, n, mins, seed=hash(tf + "cl") & 0xffff)
            proc = process_bars(raw, tf, FeatureConfig())
            _write_processed_bars(data_cfg.processed_bars_path, sym, tf, proc)

        res = align_all(data_cfg, symbol="TEST", raw_symbol=sym)
        diag = inspect_aligned(data_cfg, sym)
        assert not any("causality violation" in i for i in diag.issues), (
            f"Causality violations: {diag.issues}"
        )
        assert res.rows > 0
