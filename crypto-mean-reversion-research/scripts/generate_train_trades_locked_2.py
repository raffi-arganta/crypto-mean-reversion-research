"""
Generate trade-level CSV untuk periode TRAIN dengan kombinasi parameter
yang sudah dikunci (LOCKED) dan exit logic sl_delayed_only.

Sinyal DIAMBIL kalau entry_time <= train-end (bukan >= test-start seperti
versi OOS).

Locked params dan exit logic (sl_delayed_only) IDENTIK dengan
generate_oos_trades_locked.py -- WAJIB tetap begitu, supaya train dan
OOS benar-benar comparable (kombinasi/logic yang sama, cuma beda periode).

PENTING: exit sl_delayed_only di sini ditulis ulang sendiri, bukan reuse
exit_resolver.py / run_oos_test_sl_delayed_only.py. Cross-check n_trades-nya
kalau ada hasil pembanding dari script asli.

Output: <out-dir>/train_trades_locked_sl_delayed_only.csv
Kolom : entry_time, exit_time, direction, net_return, pair
(schema sama, langsung bisa dipakai analyze_trades.py)

RECONSTRUCTED: file asli ini sempat tertimpa/hilang. Direkonstruksi
berdasarkan generate_oos_trades_locked.py (yang masih utuh dan strukturnya
identik selain filter periode) -- WAJIB divalidasi ulang: jalankan lalu
bandingkan n_trades / statistik ringkas dengan hasil lama yang sudah kamu
punya (train_trades_locked_sl_delayed_only.csv versi sebelumnya, kalau masih
ada) untuk memastikan tidak ada perbedaan logic yang terlewat.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zscore import calculate_zscore
from atr import calculate_atr, calculate_atr_percentile
from adx import calculate_adx
from half_life import calculate_half_life
from signal_construction import combine_single_pair_entry

FEE_ROUND_TRIP = 0.0009
NO_ADX_THRESHOLD = 999.0

# ==== KOMBINASI TERKUNCI -- HARUS SAMA PERSIS DENGAN generate_oos_trades_locked.py ====
LOCKED = dict(
    z_threshold=3.8,
    hl_threshold=20.0,
    tp_mult=2.5,
    sl_mult=0.5,
    min_hold_bars=3,
    atr_threshold=0.85,
)


def load_and_prepare_one(base, data_dir, z_window, atr_n, atr_lookback, adx_n, hl_window):
    csv_path = data_dir / f"{base}USDT.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 1000:
        return None

    z = calculate_zscore(df, window=z_window)
    atr_raw = calculate_atr(df, n=atr_n)
    atr_pct = calculate_atr_percentile(df, atr_n=atr_n, lookback=atr_lookback)
    adx = calculate_adx(df, n=adx_n)["adx"]
    half_life = calculate_half_life(df, window=hl_window)["half_life"]

    return {
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "timestamps": df["timestamp"].to_numpy(),
        "z": z, "atr_raw": atr_raw.to_numpy(), "atr_pct": atr_pct,
        "adx": adx, "half_life": half_life,
    }


def resolve_sl_delayed_exit(direction, tp, sl, high, low, min_hold_bars):
    """TP dicek dari bar 1, SL baru aktif setelah min_hold_bars, tie-break SL menang."""
    n = len(high)
    for i in range(n):
        bar_offset = i + 1
        tp_hit = (high[i] >= tp) if direction == "long" else (low[i] <= tp)
        sl_active = bar_offset > min_hold_bars
        sl_hit = sl_active and ((low[i] <= sl) if direction == "long" else (high[i] >= sl))

        if sl_hit:  # tie-break: SL menang (juga saat tp_hit True bersamaan)
            return bar_offset, sl, "SL"
        if tp_hit:
            return bar_offset, tp, "TP"

    return None  # TIMEOUT


def backtest_pair_train_locked(open_, high, low, timestamps, entry_long, entry_short,
                                atr_arr, combo, train_end, pair_name):
    n = len(open_)
    trades = []
    last_exit_idx = -1

    for t in range(n - 1):
        if t <= last_exit_idx:
            continue

        ts = pd.Timestamp(timestamps[t])
        if ts > train_end:
            break

        direction = None
        if entry_long[t]:
            direction = "long"
        elif entry_short[t]:
            direction = "short"
        if direction is None:
            continue

        execution_idx = t + 1
        entry_price = open_[execution_idx]
        atr_val = atr_arr[t]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        if direction == "long":
            tp = entry_price + combo["tp_mult"] * atr_val
            sl = entry_price - combo["sl_mult"] * atr_val
        else:
            tp = entry_price - combo["tp_mult"] * atr_val
            sl = entry_price + combo["sl_mult"] * atr_val

        result = resolve_sl_delayed_exit(
            direction, tp, sl,
            high[execution_idx:], low[execution_idx:],
            combo["min_hold_bars"],
        )
        if result is None:
            last_exit_idx = n
            continue

        bar_offset, exit_price, exit_type = result
        exit_idx = execution_idx + (bar_offset - 1)
        last_exit_idx = exit_idx

        if direction == "long":
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price

        trades.append({
            "entry_time": ts,
            "exit_time": pd.Timestamp(timestamps[exit_idx]),
            "direction": direction,
            "net_return": raw_return - FEE_ROUND_TRIP,
            "pair": pair_name,
            "exit_type": exit_type,        # "TP" atau "SL"
            "exit_bar_offset": bar_offset,  # bar ke berapa (sejak entry fill=bar 1) exit terjadi
        })

    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                         help="Pair list untuk universe train (mis. pairs_train_selected.txt).")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--train-end", required=True, help="YYYY-MM-DD")

    parser.add_argument("--z-window", type=int, default=50)
    parser.add_argument("--atr-n", type=int, default=14)
    parser.add_argument("--atr-lookback", type=int, default=100)
    parser.add_argument("--adx-n", type=int, default=14)
    parser.add_argument("--hl-window", type=int, default=100)
    args = parser.parse_args()

    train_end = pd.Timestamp(args.train_end)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    print(f"[*] {len(pairs)} pair | train-end={train_end.date()} | LOCKED={LOCKED}")

    all_trades = []
    for base in pairs:
        d = load_and_prepare_one(base, data_dir, args.z_window, args.atr_n,
                                  args.atr_lookback, args.adx_n, args.hl_window)
        if d is None:
            print(f"  [!] {base}: skip (data tidak ada / < 1000 bar)")
            continue

        signal = combine_single_pair_entry(
            d["z"], d["atr_pct"], d["adx"], d["half_life"],
            z_threshold=LOCKED["z_threshold"], atr_threshold=LOCKED["atr_threshold"],
            adx_threshold=NO_ADX_THRESHOLD, hl_threshold=LOCKED["hl_threshold"],
        )
        entry_long = signal["entry_long"].to_numpy()
        entry_short = signal["entry_short"].to_numpy()

        trades = backtest_pair_train_locked(
            d["open"], d["high"], d["low"], d["timestamps"], entry_long, entry_short,
            d["atr_raw"], LOCKED, train_end, base,
        )
        all_trades.extend(trades)
        print(f"  {base:8s}: {len(trades)} trade (train)")

    out_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    out_path = out_dir / "train_trades_locked_sl_delayed_only.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\n[+] Total trade (train, semua pair): {len(out_df)}")
    print(f"[+] Disimpan ke: {out_path}")


if __name__ == "__main__":
    main()