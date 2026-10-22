"""
Diagnostik timing entry: fitur candle sinyal mana yang membedakan trade yang
langsung kena SL di bar 1-3 (pisau jatuh) vs yang tidak.

Input : CSV trade train (hasil generate_train_trades_locked*.py)
        + folder data harga {BASE}USDT.csv (sama dengan yang dipakai backtest)
Hanya memakai periode TRAIN (entry_time <= --train-end). Jangan diberi data OOS.

Semua fitur dihitung dari bar sinyal t (dan open[t+1] untuk gap) -> tidak ada
lookahead. Hasil (MAE, outcome) dihitung dari bar t+1 dst.

Counterfactual: tiap trade yang SAMA disimulasikan ulang dengan
min_hold_bars=0 (SL aktif dari bar 1) untuk melihat dampak kalau fitur
"kebal 3 bar" dihapus. Catatan: trade berikutnya di pair yang sama bisa
berubah kalau exit lebih cepat; ini diabaikan (diagnostik, bukan backtest ulang).

Fitur diorientasikan: nilai BESAR = lebih "sudah berbalik / searah posisi",
untuk long maupun short.

Contoh:
  python diagnose_entry_features.py ^
    --trades train_trades_locked_sl_delayed_only.csv ^
    --data-dir D:\\...\\futures_1h --train-end 2023-12-31 --out-dir .\\diag
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zscore import calculate_zscore
from atr import calculate_atr, calculate_atr_percentile

try:
    from half_life import calculate_half_life
except Exception:  # half_life opsional, cuma untuk fitur tambahan
    calculate_half_life = None

FEE_ROUND_TRIP = 0.0009
LOCKED = dict(z_threshold=3.8, tp_mult=2.5, sl_mult=0.5, min_hold_bars=3)
EARLY_BARS = 3


# ───────────────────────── data ─────────────────────────
def load_pair(base, data_dir, z_window, atr_n, atr_lookback, hl_window):
    p = data_dir / f"{base}USDT.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    d = {
        "ts": pd.DatetimeIndex(df["timestamp"]),
        "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
        "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
        "z": calculate_zscore(df, window=z_window).to_numpy(float),
        "atr": calculate_atr(df, n=atr_n).to_numpy(float),
        "atr_pct": calculate_atr_percentile(df, atr_n=atr_n, lookback=atr_lookback).to_numpy(float),
        "hl": None,
    }
    if calculate_half_life is not None:
        try:
            d["hl"] = calculate_half_life(df, window=hl_window)["half_life"].to_numpy(float)
        except Exception:
            pass
    return d


def cf_exit(direction, entry, tp, sl, high, low, min_hold):
    """Exit counterfactual. SL aktif setelah min_hold bar; tie -> SL menang."""
    n = len(high)
    if n == 0:
        return None
    if direction == "long":
        tp_hit, sl_hit = high >= tp, low <= sl
    else:
        tp_hit, sl_hit = low <= tp, high >= sl
    sl_hit = sl_hit & (np.arange(1, n + 1) > min_hold)
    hit = tp_hit | sl_hit
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    return (i + 1, sl, "SL") if sl_hit[i] else (i + 1, tp, "TP")


# ───────────────────────── fitur per trade ─────────────────────────
def build_features(trades, data_dir, args):
    rows, cache = [], {}
    for r in trades.itertuples(index=False):
        base = r.pair
        if base not in cache:
            cache[base] = load_pair(base, data_dir, args.z_window, args.atr_n,
                                    args.atr_lookback, args.hl_window)
        d = cache[base]
        if d is None:
            continue
        t = d["ts"].get_indexer([r.entry_time])[0]
        if t < 6 or t + 1 + EARLY_BARS >= len(d["o"]):
            continue

        sign = 1.0 if r.direction == "long" else -1.0
        o, h, l, c, z, atr = d["o"], d["h"], d["l"], d["c"], d["z"], d["atr"][t]
        if not np.isfinite(atr) or atr <= 0:
            continue

        ex = t + 1
        entry = o[ex]
        rng = h[t] - l[t]
        s_z = -sign * z                         # positif = ekstrem searah posisi
        s_z_win = s_z[t - 4:t + 1]

        pos = (c[t] - l[t]) / rng if rng > 0 else np.nan
        close_pos = pos if sign > 0 else 1 - pos
        if rng > 0:
            wick = ((min(o[t], c[t]) - l[t]) if sign > 0 else (h[t] - max(o[t], c[t]))) / rng
        else:
            wick = np.nan

        run = 0
        while t - run >= 0 and s_z[t - run] >= args.z_threshold:
            run += 1

        # outcome: adverse/favorable excursion bar 1..EARLY_BARS (satuan ATR)
        hh, ll = h[ex:ex + EARLY_BARS], l[ex:ex + EARLY_BARS]
        if sign > 0:
            mae, mfe = (entry - ll.min()) / atr, (hh.max() - entry) / atr
        else:
            mae, mfe = (hh.max() - entry) / atr, (entry - ll.min()) / atr

        tp = entry + sign * LOCKED["tp_mult"] * atr
        sl = entry - sign * LOCKED["sl_mult"] * atr
        res = cf_exit(r.direction, entry, tp, sl, h[ex:], l[ex:], 0)
        if res is None:
            exit0, ret0, off0 = "TIMEOUT", np.nan, np.nan
        else:
            off0, px, exit0 = res
            raw = sign * (px - entry) / entry
            ret0 = raw - FEE_ROUND_TRIP

        rows.append({
            "pair": base, "entry_time": r.entry_time, "direction": r.direction,
            "ret_orig": r.net_return,
            "exit_orig": getattr(r, "exit_type", None),
            # fitur
            "z_abs": abs(z[t]),
            "dz_orient": sign * (z[t] - z[t - 1]),
            "pullback_z": np.nanmax(s_z_win) - s_z[t],
            "close_pos": close_pos,
            "wick_orient": wick,
            "body_orient": sign * (c[t] - o[t]) / atr,
            "range_atr": rng / atr,
            "ret3_orient": sign * (c[t] - c[t - 3]) / atr,
            "run_len": run,
            "gap_orient": sign * (entry - c[t]) / atr,
            "atr_pct": d["atr_pct"][t],
            "atr_rel": atr / c[t],
            "half_life": d["hl"][t] if d["hl"] is not None else np.nan,
            # outcome
            "mae3_atr": mae, "mfe3_atr": mfe,
            "early_sl": int(mae >= LOCKED["sl_mult"]),
            "exit_cf0": exit0, "ret_cf0": ret0, "off_cf0": off0,
        })
    return pd.DataFrame(rows)


# ───────────────────────── analisis ─────────────────────────
FEATURES = ["z_abs", "dz_orient", "pullback_z", "close_pos", "wick_orient", "body_orient",
            "range_atr", "ret3_orient", "run_len", "gap_orient", "atr_pct", "atr_rel", "half_life"]


def auc(x, y):
    """AUC fitur x memprediksi y=1 (early_sl). 0.5 = tidak informatif."""
    m = x.notna()
    x, y = x[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = x.rank()
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(g):
    r0 = g["ret_cf0"].dropna()
    gl = -r0[r0 <= 0].sum()
    return pd.Series({
        "n": len(g),
        "early_sl_%": g["early_sl"].mean() * 100,
        "sl_cf0_%": (g["exit_cf0"] == "SL").mean() * 100,
        "tp_cf0_%": (g["exit_cf0"] == "TP").mean() * 100,
        "avg_ret_cf0_%": r0.mean() * 100 if len(r0) else np.nan,
        "pf_cf0": r0[r0 > 0].sum() / gl if gl > 0 else np.inf,
        "avg_ret_orig_%": g["ret_orig"].mean() * 100,
        "avg_mae3_atr": g["mae3_atr"].mean(),
    })


def bucket_table(df, feat, q=5):
    s = df[feat]
    if s.nunique() <= 8:
        key = s.clip(upper=6) if feat == "run_len" else s
    else:
        key = pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop")
    out = df.groupby(key, observed=True).apply(metrics)
    lo = df.groupby(key, observed=True)[feat].agg(["min", "max"])
    out.insert(0, f"{feat}_range", lo.apply(lambda r: f"{r['min']:.3g}..{r['max']:.3g}", axis=1))
    out.index.name = "bucket"
    return out


def rule_table(df):
    rules = {
        "BASELINE (semua trade)": pd.Series(True, index=df.index),
        "dz_orient > 0 (z mulai berbalik)": df["dz_orient"] > 0,
        "pullback_z > 0 (turun dari puncak z)": df["pullback_z"] > 0,
        "close_pos >= 0.5": df["close_pos"] >= 0.5,
        "body_orient > 0 (candle searah posisi)": df["body_orient"] > 0,
        "gap_orient > 0 (open t+1 searah)": df["gap_orient"] > 0,
        "dz>0 & close_pos>=0.5": (df["dz_orient"] > 0) & (df["close_pos"] >= 0.5),
        "dz>0 & body>0": (df["dz_orient"] > 0) & (df["body_orient"] > 0),
        "dz>0 & close_pos>=0.5 & gap>0": (df["dz_orient"] > 0) & (df["close_pos"] >= 0.5) & (df["gap_orient"] > 0),
    }
    rows = {}
    for name, m in rules.items():
        g = df[m.fillna(False)]
        if len(g) == 0:
            continue
        s = metrics(g)
        s["kept_%"] = len(g) / len(df) * 100
        rows[name] = s
    return pd.DataFrame(rows).T[["n", "kept_%", "early_sl_%", "sl_cf0_%", "tp_cf0_%",
                                  "avg_ret_cf0_%", "pf_cf0", "avg_ret_orig_%"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True, help="YYYY-MM-DD, trade setelah ini dibuang")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--z-threshold", type=float, default=LOCKED["z_threshold"])
    ap.add_argument("--z-window", type=int, default=50)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--atr-lookback", type=int, default=100)
    ap.add_argument("--hl-window", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.trades)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr = tr[tr["entry_time"] <= pd.Timestamp(args.train_end)].reset_index(drop=True)
    print(f"[*] {len(tr)} trade (train, <= {args.train_end}) | {tr['pair'].nunique()} pair")

    df = build_features(tr, Path(args.data_dir), args)
    print(f"[*] {len(df)} trade punya fitur lengkap")
    df.to_csv(out_dir / "trade_features.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:,.3f}"

    print("\n" + "=" * 70 + "\nBASELINE: dampak menghapus kebal 3 bar (SL aktif dari bar 1)\n" + "=" * 70)
    print(f"  early_sl (MAE bar 1-3 >= {LOCKED['sl_mult']} ATR): {df['early_sl'].mean()*100:.1f}% dari trade")
    print(f"  avg return ORIGINAL (kebal 3 bar): {df['ret_orig'].mean()*100:+.3f}%")
    print(f"  avg return CF0      (tanpa kebal): {df['ret_cf0'].mean()*100:+.3f}%")
    print(f"  distribusi exit CF0: {df['exit_cf0'].value_counts(normalize=True).mul(100).round(1).to_dict()}")

    print("\n" + "=" * 70 + "\nAUC fitur vs early_sl (0.5 = tidak informatif; <0.5 = fitur besar -> lebih aman)\n" + "=" * 70)
    auc_df = pd.DataFrame({
        "auc": {f: auc(df[f], df["early_sl"]) for f in FEATURES if f in df and df[f].notna().any()}
    })
    auc_df["abs_edge"] = (auc_df["auc"] - 0.5).abs()
    auc_df = auc_df.sort_values("abs_edge", ascending=False)
    print(auc_df.to_string(float_format=fmt))
    auc_df.to_csv(out_dir / "feature_auc.csv")

    print("\n" + "=" * 70 + "\nBUCKET per fitur (5 kuantil, bucket 0 = nilai terkecil)\n" + "=" * 70)
    for f in auc_df.index[:6]:
        t = bucket_table(df, f)
        print(f"\n--- {f} (AUC={auc_df.loc[f, 'auc']:.3f}) ---")
        print(t.to_string(float_format=fmt))
        t.to_csv(out_dir / f"bucket_{f}.csv")

    print("\n" + "=" * 70 + "\nUJI ATURAN FILTER SEDERHANA (hasil counterfactual min_hold=0)\n" + "=" * 70)
    rt = rule_table(df)
    print(rt.to_string(float_format=fmt))
    rt.to_csv(out_dir / "rule_table.csv")

    print("\nCara baca: aturan bagus = early_sl_% turun jelas dan avg_ret_cf0_% / pf_cf0 naik,")
    print("dengan kept_% tidak anjlok (idealnya > 40-50%). Validasi dulu stabil per tahun")
    print("dan per pair di TRAIN sebelum menyentuh OOS.")


if __name__ == "__main__":
    main()