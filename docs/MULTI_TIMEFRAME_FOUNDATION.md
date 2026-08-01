# Phase 24: Full Multi-Timeframe Stack

The bot is now natively multi-timeframe.

## Configuration
See src/slytrade/config/mtf.py

## observe_timeframes
M1, M5, M15, H1, H4, D1, W1

## Key Features
- Forward-filled HTF ICT features into every M1 bar
- mtf_confluence_score and mtf_bias
- MTFICTConfluenceStrategy requires macro + micro alignment

This completes the foundation for true macro + micro ICT/SMC trading.
