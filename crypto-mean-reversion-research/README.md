# Crypto Mean Reversion Research (Binance USDT-M Perpetual)

Research on whether a z-score mean reversion strategy has an edge that
survives fees and realistic execution on Binance perpetual futures.

**Conclusion:** the original edge was an artifact of unrealistic fill
assumptions and pair-selection bias. The strategy was discontinued and the
out-of-sample data (2024+) was deliberately left untouched.

## Key findings
- Old backtest: +0.463% per trade. Realistic execution model: -0.135% per trade.
- 450 ideas screened; the only two that passed validation were one finding.
- Pair selection on the same data was snooping. On 202 never-selected pairs the
  candidate failed its pre-registered criteria.
- Past pair performance did not predict future performance (mean IC -0.037, t -0.97).

## Structure
```
reports/   full research reports (EN / ID)
scripts/   analysis scripts, in chronological order
data/      pairs_train_selected.txt (pair list used in the original strategy)
```

## Scripts (chronological)
1. `generate_train_trades_locked_2.py`
2. `analyze_trades.py`
3. `diagnose_entry_features.py`
4. `sweep_entry_exit.py`
5. `sweep_time_stop.py`
6. `sweep_residual.py`
7. `robust_residual.py`
8. `diagnose_orderflow.py`
9. `download_binance_vision.py`
10. `screen_ideas.py`
11. `robust_survivor.py`
12. `oos_survivor.py`
13. `build_universe.py`
14. `pair_persistence.py`

## Methodology
Point-in-time universe, criteria written before looking at results,
discovery/validation split, pair persistence test, single OOS run.

## Limitations
All numbers come from training data (Dec 2019 - Dec 2023). Fees of
0.09%-0.25% are assumptions, not measured slippage.

## Disclaimer
Research only, not financial advice.

Author: Raffi Arganta Viari
