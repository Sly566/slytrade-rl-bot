#!/usr/bin/env python3
"""SlyTrade RL Full Pipeline — Run this on YOUR machine.

This script:
1. Downloads/loads the 2-year XAUUSD data from the release
2. Processes through feature pipeline (M1+M5+M15+M30+H1)
3. Aligns all TFs onto M1 (same as live trader)
4. Trains PPO agent with M15 structure (broader intraday blanket)
5. Evaluates on held-out test set
6. Saves model + metrics

Usage:
    # Install dependencies first
    pip install -e ".[data,rl]"

    # Run the full pipeline
    python scripts/rl_pipeline.py

    # Or with custom data path
    python scripts/rl_pipeline.py --data path/to/xauusd-2y-processed-2026-08-21.tar.zst
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slytrade.data.features import DEFAULT_CONFIG, process_bars
from slytrade.data.mtf_align import _asof_merge, _prep_htf_frame
from slytrade.data.resample import resample_bars_to_timeframe
from slytrade.data.time import timeframe_timedelta
from slytrade.rl.env import SlyTradeEnv, OBS_DIM


def extract_data(archive_path: str, output_dir: str = "data/raw") -> str:
    """Extract tar.zst archive to raw data directory."""
    print(f"\n{'='*70}")
    print("STEP 1: EXTRACT DATA")
    print(f"{'='*70}")

    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)...")

    # tar.zst needs zstandard
    try:
        import zstandard as zstd
        with open(archive, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=out)
    except ImportError:
        # Fallback: try system tar
        import subprocess
        subprocess.run(["tar", "--use-compress-program=zstd", "-xf", str(archive), "-C", str(out)], check=True)

    # Find parquet files
    parquets = list(out.rglob("*.parquet"))
    print(f"  Extracted {len(parquets)} parquet files")
    for p in parquets[:10]:
        print(f"    {p.relative_to(out)} ({p.stat().st_size / 1e6:.1f} MB)")

    return str(out)


def find_m1_data(raw_dir: str, symbol: str = "XAUUSD") -> str:
    """Find the M1 parquet file in the raw data directory."""
    raw = Path(raw_dir)

    # Look for M1 parquet files
    candidates = []
    for p in raw.rglob("*.parquet"):
        name = p.name.lower()
        if "m1" in name or "1min" in name or "1m" in name:
            if symbol.lower() in name or "xau" in name:
                candidates.append(p)

    if not candidates:
        # Try any parquet with XAU in name
        candidates = list(raw.rglob(f"*{symbol}*.parquet"))

    if not candidates:
        # Try any parquet at all
        candidates = list(raw.rglob("*.parquet"))

    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    # Pick the largest one (most likely the M1 data)
    best = max(candidates, key=lambda p: p.stat().st_size)
    print(f"  Using: {best.relative_to(raw)} ({best.stat().st_size / 1e6:.1f} MB)")
    return str(best)


def process_and_align(m1_path: str, output_path: str = "data/aligned") -> str:
    """Process M1 data through feature pipeline and align all TFs.

    This replicates EXACTLY what the live trader does:
    1. Compute M1 features (ATR, structure flags, etc.)
    2. Resample to M5, M15, M30, H1
    3. Compute features on each TF
    4. Causal asof merge onto M1
    """
    print(f"\n{'='*70}")
    print("STEP 2: PROCESS FEATURES + ALIGN MTF")
    print(f"{'='*70}")

    print(f"\n  Loading M1 data from {m1_path}...")
    m1_raw = pd.read_parquet(m1_path)
    print(f"  {len(m1_raw):,} raw M1 bars")
    print(f"  Columns: {m1_raw.columns.tolist()}")
    print(f"  Date range: {m1_raw['time'].min()} to {m1_raw['time'].max()}")

    # Ensure required columns exist
    for col in ["tick_volume", "real_volume"]:
        if col not in m1_raw.columns:
            if "volume" in m1_raw.columns:
                m1_raw[col] = m1_raw["volume"]
            else:
                m1_raw[col] = 0.0

    # Step 1: Compute M1 features
    print(f"\n  Computing M1 features...")
    m1 = process_bars(m1_raw, "M1", DEFAULT_CONFIG)
    print(f"  M1: {len(m1):,} bars, {len(m1.columns)} columns")

    # Step 2: Resample to HTFs and compute features
    htf_frames = {}
    for tf in ["M5", "M15", "M30", "H1"]:
        print(f"  Processing {tf}...")
        htf_raw = resample_bars_to_timeframe(m1_raw, tf)
        htf = process_bars(htf_raw, tf, DEFAULT_CONFIG)
        htf_frames[tf] = htf
        print(f"    {tf}: {len(htf):,} bars, {len(htf.columns)} columns")

    # Step 3: Causal asof merge onto M1
    print(f"\n  Aligning HTF features onto M1...")
    df = m1.copy().sort_values("time").reset_index(drop=True)
    for tf, htf in htf_frames.items():
        if htf.empty:
            continue
        dur = timeframe_timedelta(tf)
        htf = htf.copy()
        htf["decision_time"] = htf["time"] + dur
        prepped = _prep_htf_frame(htf, tf)
        df = _asof_merge(df, prepped, tf)
        print(f"    Aligned {tf}: {len(df):,} rows, {len(df.columns)} columns")

    # Save
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "XAUUSD_2y_aligned.parquet"
    df.to_parquet(out_path)
    print(f"\n  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(df):,} bars, {len(df.columns)} columns")

    # Show structure features
    structure_cols = [c for c in df.columns if any(x in c for x in ['disp', 'bos', 'choch', 'sweep'])]
    print(f"\n  Structure features ({len(structure_cols)}):")
    for tf_prefix in ["M1_", "M5_", "M15_", "M30_", "H1_"]:
        tf_cols = [c for c in structure_cols if c.startswith(tf_prefix) or (tf_prefix == "M1_" and not c.startswith(("M5_", "M15_", "M30_", "H1_")))]
        if tf_cols:
            print(f"    {tf_prefix.rstrip('_')}: {len(tf_cols)} features")

    return str(out_path)


def train_agent(aligned_path: str, output_dir: str = "models", timesteps: int = 500_000) -> str:
    """Train PPO agent on aligned data."""
    print(f"\n{'='*70}")
    print("STEP 3: TRAIN RL AGENT")
    print(f"{'='*70}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    print(f"\n  Loading aligned data...")
    df = pd.read_parquet(aligned_path)
    print(f"  {len(df):,} bars, {len(df.columns)} columns")
    print(f"  Observation space: {OBS_DIM} dimensions")

    # Train/test split (80/20, chronological)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(train_df):,} bars ({train_df['time'].min()} to {train_df['time'].max()})")
    print(f"  Test:  {len(test_df):,} bars ({test_df['time'].min()} to {test_df['time'].max()})")

    # Create environments
    def make_train_env():
        env = SlyTradeEnv(train_df, max_bars=5000)
        return Monitor(env)

    def make_test_env():
        env = SlyTradeEnv(test_df, max_bars=5000)
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])

    # Create PPO agent
    print(f"\n  Creating PPO agent...")
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, n_steps=2048, batch_size=256, n_epochs=10,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
        verbose=0, seed=42,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Parameters: {n_params:,}")

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_sharpe = -999
    start_time = time.time()
    n_iters = max(1, timesteps // 50_000)

    print(f"\n  Training for {timesteps:,} timesteps ({n_iters} iterations)...")
    for iteration in range(n_iters):
        model.learn(total_timesteps=50_000, reset_num_timesteps=False)

        # Evaluate on test set
        eval_env = SlyTradeEnv(test_df, max_bars=len(test_df))
        obs, info = eval_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                break

        metrics = eval_env.get_metrics()
        elapsed = time.time() - start_time

        print(f"    iter={iteration+1}/{n_iters}  "
              f"trades={metrics['n_trades']}  wr={metrics['win_rate']:.0%}  "
              f"pnl={metrics['total_pnl']:+.0f}  sharpe={metrics['sharpe_ratio']:.2f}  "
              f"max_dd={metrics['max_drawdown']:.1%}  pf={metrics['profit_factor']:.2f}  "
              f"elapsed={elapsed:.0f}s")

        if metrics['sharpe_ratio'] > best_sharpe and metrics['n_trades'] > 10:
            best_sharpe = metrics['sharpe_ratio']
            model.save(f"{output_dir}/ppo_XAUUSD_best")
            print(f"      *** New best (Sharpe={best_sharpe:.2f}) ***")

    # Save final
    model.save(f"{output_dir}/ppo_XAUUSD_final")
    print(f"\n  Models saved to {output_dir}/")
    return output_dir


def evaluate_model(model_dir: str, aligned_path: str) -> dict:
    """Final evaluation of best model on full test set."""
    print(f"\n{'='*70}")
    print("STEP 4: FINAL EVALUATION")
    print(f"{'='*70}")

    from stable_baselines3 import PPO

    df = pd.read_parquet(aligned_path)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    model = PPO.load(f"{model_dir}/ppo_XAUUSD_best")
    print(f"  Loaded best model")

    env = SlyTradeEnv(test_df, max_bars=len(test_df))
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    metrics = env.get_metrics()

    print(f"\n  {'='*50}")
    print(f"  RESULTS ON TEST SET ({len(test_df):,} bars)")
    print(f"  {'='*50}")
    print(f"  Trades:      {metrics['n_trades']}")
    print(f"  Win Rate:    {metrics['win_rate']:.1%}")
    print(f"  Total P&L:   {metrics['total_pnl']:+.2f}")
    print(f"  Return:      {metrics['return_pct']:.1%}")
    print(f"  Sharpe:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Profit Fac:  {metrics['profit_factor']:.2f}")
    print(f"  Max DD:      {metrics['max_drawdown']:.1%}")
    print(f"  Avg Win:     {metrics['avg_win']:+.2f}")
    print(f"  Avg Loss:    {metrics['avg_loss']:+.2f}")

    with open(f"{model_dir}/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Metrics saved to {model_dir}/final_metrics.json")

    return metrics


def main():
    ap = argparse.ArgumentParser(description="SlyTrade RL Full Pipeline")
    ap.add_argument("--data", default="data/raw/xauusd-2y-processed-2026-08-21.tar.zst",
                    help="Path to data archive (tar.zst) or directory with parquet files")
    ap.add_argument("--output", default="models", help="Output directory for models")
    ap.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    ap.add_argument("--skip-extract", action="store_true", help="Skip extraction (data already extracted)")
    ap.add_argument("--skip-process", action="store_true", help="Skip processing (aligned data exists)")
    ap.add_argument("--aligned", default="data/aligned/XAUUSD_2y_aligned.parquet",
                    help="Path to pre-aligned data (if --skip-process)")
    args = ap.parse_args()

    print("=" * 70)
    print("SLYTRADE RL FULL PIPELINE")
    print("=" * 70)
    print(f"  Data:       {args.data}")
    print(f"  Output:     {args.output}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Obs space:  {OBS_DIM} dimensions (M1+M5+M15 structure)")

    start = time.time()

    # Step 1: Extract
    if args.skip_extract:
        raw_dir = "data/raw"
    else:
        raw_dir = extract_data(args.data)

    # Step 2: Process + Align
    if args.skip_process:
        aligned_path = args.aligned
    else:
        m1_path = find_m1_data(raw_dir)
        aligned_path = process_and_align(m1_path)

    # Step 3: Train
    model_dir = train_agent(aligned_path, args.output, args.timesteps)

    # Step 4: Evaluate
    metrics = evaluate_model(model_dir, aligned_path)

    total_time = time.time() - start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE ({total_time:.0f}s)")
    print(f"{'='*70}")
    print(f"  Model: {model_dir}/ppo_XAUUSD_best.zip")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
