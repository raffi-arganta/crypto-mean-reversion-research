"""
Uji sumber informasi baru untuk sinyal mean reversion (TRAIN saja):

  1. Z-score RESIDUAL: z dihitung pada indeks return koin yang sudah di-hedge BTC
       resid_ret[t] = r_coin[t] - beta[t-1] * r_btc[t]   (beta rolling, tanpa lookahead)
       X = cumsum(resid_ret) ; z_res = rolling z-score X
     Pembanding: z "raw" (harga koin langsung), sama seperti sistem sekarang.
  2. Filter REGIME pasar (opsional):
       btcvol  : BTC atr_pct <= --btc-atr-max (hindari saat volatilitas BTC ekstrem)
       breadth : jumlah pair yang SEARAH ekstrem (|z_raw| >= --breadth-z) di bar yang sama
                 <= --breadth-max (hindari gerakan seluruh pasar / dump massal)
       both    : keduanya

Filter dasar lain sama dengan sistem sekarang: atr_pct <= 0.85 (tanpa ADX).
Half-life filter TIDAK dipakai di sini (supaya raw vs resid apple-to-apple), jadi
baris raw/none tidak persis sama dengan CSV train lama.

Exit memakai model eksekusi realistis (bukan model ideal):
  live_mh3 : TP 2.5 ATR, SL 0.5 ATR dipasang setelah bar ke-4, gap -> market close (sweep_entry_exit)
  time     : TP 2.0 ATR, catastrophic 5 ATR, time stop 24 bar (sweep_time_stop)

Kolom penting: gross_% (sebelum fee) vs avg_ret_% (setelah fee) untuk melihat apakah
ada edge mentah atau habis dimakan fee.

Butuh di folder yang sama: sweep_entry_exit.py, sweep_time_stop.py, zscore.py, atr.py,
adx.py, half_life.py, signal_construction.py. Data harus punya BTCUSDT.csv.

Contoh:
  py sweep_residual.py --pairs <pairs_train_selected.txt> --data-dir <futures_1h> ^
     --train-end 2023-12-31 --out-dir <sweep_resid>
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sweep_entry_exit as se
from sweep_time_stop import resolve_time_exit
from zscore import calculate_zscore
from atr import calculate_atr, calculate_atr_percentile

FEE = se.FEE_ROUND_TRIP
ATR_THR = 0.85
FAST = se.FAST_WINDOW

EXITS = {
    "live_mh3": dict(kind="live", sl=0.5, tp=2.5, mh=3),
    "time": dict(kind="time", tp=2.0, cat=5.0, n=24),
}


# ───────────────────────── data ─────────────────────────
def read_df(base, data_dir):
    p = data_dir / f"{base}USDT.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df if len(df) >= 1000 else None


def prepare(pairs, data_dir, a):
    btc_df = read_df("BTC", data_dir)
    if btc_df is None:
        raise SystemExit("BTCUSDT.csv tidak ditemukan di --data-dir (dibutuhkan untuk hedge & regime).")
    bts = pd.DatetimeIndex(btc_df["timestamp"])
    btc_ret = pd.Series(np.log(btc_df["close"].to_numpy(float)), index=bts).diff()
    btc_atrp = pd.Series(
        calculate_atr_percentile(btc_df, atr_n=a.atr_n, lookback=a.atr_lookback).to_numpy(float), index=bts)

    data = {}
    for base in pairs:
        df = read_df(base, data_dir)
        if df is None:
            print(f"  [!] {base}: skip")
            continue
        ts = pd.DatetimeIndex(df["timestamp"])
        r = pd.Series(np.log(df["close"].to_numpy(float)), index=ts).diff()
        rb = btc_ret.reindex(ts)

        z_raw = calculate_zscore(df, window=a.z_window).to_numpy(float).copy()

        if base == "BTC":
            z_res = np.full(len(df), np.nan)
        else:
            w = a.beta_window
            beta = r.rolling(w, min_periods=w // 2).cov(rb) / rb.rolling(w, min_periods=w // 2).var()
            inc = r - beta.shift(1) * rb
            cnt = inc.notna().rolling(a.z_window).sum()
            x = pd.DataFrame({"x": inc.fillna(0).cumsum().to_numpy()})
            z_res = calculate_zscore(x, window=a.z_window, price_col="x").to_numpy(float).copy()
            z_res[(cnt < a.z_window).to_numpy()] = np.nan

        data[base] = {
            "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
            "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
            "ts": ts,
            "atr": calculate_atr(df, n=a.atr_n).to_numpy(float),
            "atr_pct": calculate_atr_percentile(df, atr_n=a.atr_n, lookback=a.atr_lookback).to_numpy(float),
            "z_raw": z_raw, "z_res": z_res,
            "btc_atrp": btc_atrp.reindex(ts).to_numpy(float),
        }

    # breadth: berapa pair searah ekstrem di bar yang sama (pakai z raw semua pair)
    Z = pd.DataFrame({b: pd.Series(d["z_raw"], index=d["ts"]) for b, d in data.items()})
    n_long = (Z <= -a.breadth_z).sum(axis=1)
    n_short = (Z >= a.breadth_z).sum(axis=1)
    for d in data.values():
        d["br_long"] = n_long.reindex(d["ts"]).to_numpy(float)
        d["br_short"] = n_short.reindex(d["ts"]).to_numpy(float)
    return data


# ───────────────────────── sinyal & backtest ─────────────────────────
def build_signals(d, mode, thr, regime, a):
    z = d["z_res"] if mode == "resid" else d["z_raw"]
    ok = d["atr_pct"] <= ATR_THR
    el, es = (z <= -thr) & ok, (z >= thr) & ok
    if regime in ("btcvol", "both"):
        calm = d["btc_atrp"] <= a.btc_atr_max
        el, es = el & calm, es & calm
    if regime in ("breadth", "both"):
        el = el & (d["br_long"] <= a.breadth_max)
        es = es & (d["br_short"] <= a.breadth_max)
    el, es = np.nan_to_num(el, nan=0).astype(bool), np.nan_to_num(es, nan=0).astype(bool)
    return el, es


def run_exit(ex_cfg, direction, entry, atr, d, ex):
    if ex_cfg["kind"] == "live":
        tp = entry + direction * ex_cfg["tp"] * atr
        sl = entry - direction * ex_cfg["sl"] * atr
        res = se._resolve("live", direction, tp, sl, d, ex, ex + FAST, ex_cfg["mh"])
        if res is None:
            res = se._resolve("live", direction, tp, sl, d, ex, None, ex_cfg["mh"])
        return res
    n = ex_cfg["n"]
    tp = entry + direction * ex_cfg["tp"] * atr
    cat = entry - direction * ex_cfg["cat"] * atr
    return resolve_time_exit(direction, tp, cat, n, d["o"][ex:ex + n], d["h"][ex:ex + n],
                             d["l"][ex:ex + n], d["c"][ex:ex + n])


def backtest(data, train_end, mode, thr, regime, ex_cfg, a):
    recs, n_sig = [], 0
    for base, d in data.items():
        el, es = build_signals(d, mode, thr, regime, a)
        n = len(d["o"])
        last_exit = -1
        for t in np.flatnonzero(el | es):
            if t >= n - 1 or t <= last_exit:
                continue
            if d["ts"][t] > train_end:
                break
            atr = d["atr"][t]
            if not np.isfinite(atr) or atr <= 0:
                continue
            direction = 1 if el[t] else -1
            n_sig += 1
            ex = t + 1
            entry = d["o"][ex]
            res = run_exit(ex_cfg, direction, entry, atr, d, ex)
            if res is None:
                last_exit = n
                continue
            off, px, typ = res
            last_exit = ex + off - 1
            recs.append((d["ts"][t], base, direction * (px - entry) / entry - FEE, typ))
    return pd.DataFrame(recs, columns=["entry_time", "pair", "ret", "type"]), n_sig


def summarize(tr, n_sig):
    if len(tr) == 0:
        return {"n_trades": 0}
    s = se.summarize(tr, n_sig, len(tr))
    s["gross_%"] = (tr["ret"].mean() + FEE) * 100
    s["worst_trade_%"] = tr["ret"].min() * 100
    s["p5_%"] = tr["ret"].quantile(0.05) * 100
    s.pop("fill_%", None)
    return s


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--thr-grid", default="2.5,3.0,3.5,3.8")
    ap.add_argument("--regimes", default="none,btcvol,breadth,both")
    ap.add_argument("--exits", default="live_mh3,time")
    ap.add_argument("--beta-window", type=int, default=500, help="window beta rolling terhadap BTC (bar)")
    ap.add_argument("--btc-atr-max", type=float, default=0.8)
    ap.add_argument("--breadth-z", type=float, default=2.5)
    ap.add_argument("--breadth-max", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=300)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--z-window", type=int, default=50)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--atr-lookback", type=int, default=100)
    a = ap.parse_args()

    train_end = pd.Timestamp(a.train_end)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.pairs, "r", encoding="utf-8") as f:
        pairs = [x.strip() for x in f if x.strip() and not x.strip().startswith("#")]

    print(f"[*] load & siapkan {len(pairs)} pair (beta, z residual, breadth) ...")
    data = prepare(pairs, Path(a.data_dir), a)
    print(f"[*] {len(data)} pair siap | train-end={train_end.date()}")

    thrs = [float(x) for x in a.thr_grid.split(",")]
    regimes = a.regimes.split(",")
    exits = a.exits.split(",")
    combos = list(itertools.product(["raw", "resid"], thrs, regimes, exits))
    print(f"[*] sweep {len(combos)} konfigurasi ...")

    rows = []
    for k, (mode, thr, regime, ex_name) in enumerate(combos, 1):
        tr, n_sig = backtest(data, train_end, mode, thr, regime, EXITS[ex_name], a)
        s = summarize(tr, n_sig)
        s.update({"mode": mode, "thr": thr, "regime": regime, "exit": ex_name})
        rows.append(s)
        if k % 16 == 0:
            print(f"  {k}/{len(combos)}")

    res = pd.DataFrame(rows)
    cols = ["mode", "thr", "regime", "exit", "n_trades", "win_%", "tp_%", "gross_%", "avg_ret_%",
            "median_%", "pf", "t_stat", "years_pos", "worst_year_%", "worst_trade_%", "p5_%"]
    res = res[[c for c in cols if c in res.columns]]
    res.to_csv(out_dir / "sweep_residual_results.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fmt = lambda v: f"{v:,.3f}"

    ok = res[res["n_trades"] >= a.min_trades].sort_values("avg_ret_%", ascending=False)
    print("\n" + "=" * 100 + f"\nTOP {a.top} by avg_ret_% (n_trades >= {a.min_trades})\n" + "=" * 100)
    print(ok.head(a.top).to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100 + "\nRAW vs RESID (regime=none): avg_ret_% / gross_% per threshold & exit\n" + "=" * 100)
    base = res[res["regime"] == "none"]
    for col in ["avg_ret_%", "gross_%", "n_trades"]:
        print(f"\n{col}:")
        print(base.pivot_table(index=["exit", "mode"], columns="thr", values=col).to_string(float_format=fmt))

    print("\n" + "=" * 100 + "\nPENGARUH REGIME (rata-rata avg_ret_% dan n_trades, per mode)\n" + "=" * 100)
    print(res.groupby(["mode", "regime"])[["avg_ret_%", "pf", "n_trades"]].mean().to_string(float_format=fmt))

    print("\nCara baca: resid > raw secara konsisten (di banyak thr/exit) = hedge BTC menambah edge.")
    print("gross_% > 0 tapi avg_ret_% <= 0 = edge mentah habis dimakan fee. Kandidat layak:")
    print("avg_ret_% > 0, pf > 1.2, years_pos mayoritas, n_trades cukup, tetangga grid juga positif.")
    print(f"[+] Semua hasil: {out_dir / 'sweep_residual_results.csv'}")


if __name__ == "__main__":
    main()