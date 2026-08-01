"""MTF bar collection commands."""
from slytrade.config.mtf import DEFAULT_MTF


def collect_mtf_bars(symbol: str, start: str, end: str, timeframes=None, output_dir="data/raw"):
    from slytrade.cli import collect_bars
    tfs = timeframes or DEFAULT_MTF.observe_timeframes
    for tf in tfs:
        collect_bars(symbol=symbol, timeframe=tf, start=start, end=end, chunk_size="month", output_dir=output_dir)
    print(f"[MTF] Collected bars for {len(tfs)} timeframes: {tfs}")
