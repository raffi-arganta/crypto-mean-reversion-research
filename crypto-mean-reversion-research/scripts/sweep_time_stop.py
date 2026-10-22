"""
Sweep exit TANPA SL ketat (TRAIN saja): TP + catastrophic stop lebar + time stop.

Aturan exit per trade (entry di open bar setelah sinyal, 1 posisi per pair):
  - TP      : limit di entry +/- tp_mult*ATR, aktif dari bar 1 (fill di harga tp).
  - CAT     : catastrophic stop di entry -/+ cat_mult*ATR, aktif dari bar 1
              (dipasang saat entry, seperti live). Kalau open bar sudah melewati
              level -> fill di open (gap). cat_mult = "none" -> tanpa pengaman.
  - TIME    : kalau sampai bar ke-N belum kena TP/CAT, tutup di CLOSE bar ke-N.
  - Tie dalam satu bar (TP & CAT sama-sama kena): CAT menang (konservatif).
  - Fee: round trip FEE_ROUND_TRIP.

Pembanding dari sweep sebelumnya (train): live_mh3 = -0.135% (PF 0.89),
ideal_mh3 = +0.463% (semu). Kandidat bagus harus > 0 secara konsisten per tahun.

Butuh file di folder yang sama: sweep_entry_exit.py (dipakai ulang untuk load data
& summary), plus zscore.py, atr.py, adx.py, half_life.py, signal_construction.py.

Contoh:
  py sweep_time_stop.py --pairs <pairs_train_selected.txt> --data-dir <futures_1h> ^
     --train-end 2023-12-31 --out-dir <sweep_time>
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sweep_entry_exit as se
from sweep_entry_exit import load_pair, summarize, find_decision_bar, FEE_ROUND_TRIP


def resolve_time_exit(direction, tp, cat, n_bars, o, h, l, c):
    """(bar_offset, price, type) atau None kalau data kurang dari n_bars."""
    if len(h) < n_bars:
        return None
    o, h, l, c = o[:n_bars], h[:n_bars], l[:n_bars], c[:n_bars]
    if direction == 1:
        tp_hit = h >= tp
        cat_hit = (l <= cat) if cat is not None else np.zeros(n_bars, bool)
    else:
        tp_hit = l <= tp
        cat_hit = (h >= cat) if cat is not None else np.zeros(n_bars, bool)
    hit = tp_hit | cat_hit
    if hit.any():
        i = int(np.argmax(hit))
        if cat_hit[i]:
            gap = (o[i] <= cat) if direction == 1 else (o[i] >= cat)
            return i + 1, (o[i] if gap else cat), "CAT"
        return i + 1, tp, "TP"
    return n_bars, c[n_bars - 1], "TIME"


def backtest(data, train_end, wait, confirm, tp_mult, cat_mult, n_bars):
    recs, n_sig, n_fill = [], 0, 0
    for base, d in data.items():
        n = len(d["o"])
        last_exit = -1
        for t in d["sig_idx"]:
            if t >= n - 1 or t <= last_exit:
                continue
            if d["ts"][t] > train_end:
                break
            direction = 1 if d["el"][t] else -1
            n_sig += 1

            b = find_decision_bar(d, t, direction, wait, confirm)
            if b is None or b + 1 >= n:
                continue
            atr = d["atr"][b]
            if not np.isfinite(atr) or atr <= 0:
                continue
            n_fill += 1

            ex = b + 1
            entry = d["o"][ex]
            tp = entry + direction * tp_mult * atr
            cat = None if cat_mult is None else entry - direction * cat_mult * atr

            res = resolve_time_exit(direction, tp, cat, n_bars, d["o"][ex:ex + n_bars],
                                    d["h"][ex:ex + n_bars], d["l"][ex:ex + n_bars],
                                    d["c"][ex:ex + n_bars])
            if res is None:
                last_exit = n
                continue
            off, px, typ = res
            last_exit = ex + off - 1
            raw = direction * (px - entry) / entry
            recs.append((d["ts"][t], base, raw - FEE_ROUND_TRIP, typ, off))

    return pd.DataFrame(recs, columns=["entry_time", "pair", "ret", "type", "bars"]), n_sig, n_fill


def full_summary(tr, n_sig, n_fill):
    s = summarize(tr, n_sig, n_fill)
    if len(tr) == 0:
        return s
    s["time_%"] = (tr["type"] == "TIME").mean() * 100
    s["cat_%"] = (tr["type"] == "CAT").mean() * 100
    s["avg_bars"] = tr["bars"].mean()
    s["worst_trade_%"] = tr["ret"].min() * 100
    s["p5_%"] = tr["ret"].quantile(0.05) * 100
    s["sum_ret_%"] = tr["ret"].sum() * 100
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tp-grid", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--cat-grid", default="3,4,5,none", help="kelipatan ATR; 'none' = tanpa catastrophic stop")
    ap.add_argument("--n-grid", default="6,12,24,48", help="time stop (bar)")
    ap.add_argument("--waits", default="0", help="tunggu konfirmasi (bar); 0 = entry langsung")
    ap.add_argument("--confirm", default="z_turn")
    ap.add_argument("--min-trades", type=int, default=300)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--z-window", type=int, default=50)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--atr-lookback", type=int, default=100)
    ap.add_argument("--adx-n", type=int, default=14)
    ap.add_argument("--hl-window", type=int, default=100)
    a = ap.parse_args()

    train_end = pd.Timestamp(a.train_end)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(a.pairs, "r", encoding="utf-8") as f:
        pairs = [x.strip() for x in f if x.strip() and not x.strip().startswith("#")]

    print(f"[*] load {len(pairs)} pair ...")
    data = {}
    for b in pairs:
        d = load_pair(b, Path(a.data_dir), a)
        if d is None:
            print(f"  [!] {b}: skip")
            continue
        data[b] = d
    print(f"[*] {len(data)} pair siap | train-end={train_end.date()}")

    tps = [float(x) for x in a.tp_grid.split(",")]
    cats = [None if x.strip().lower() == "none" else float(x) for x in a.cat_grid.split(",")]
    ns = [int(x) for x in a.n_grid.split(",")]
    waits = [int(x) for x in a.waits.split(",")]

    total = len(waits) * len(tps) * len(cats) * len(ns)
    print(f"[*] sweep {total} konfigurasi ...")
    rows, k = [], 0
    for w, tp, cat, n_bars in itertools.product(waits, tps, cats, ns):
        k += 1
        tr, n_sig, n_fill = backtest(data, train_end, w, a.confirm, tp, cat, n_bars)
        s = full_summary(tr, n_sig, n_fill)
        s.update({"wait": w, "tp": tp, "cat": "none" if cat is None else cat, "N": n_bars})
        rows.append(s)
        if k % 20 == 0:
            print(f"  {k}/{total}")

    res = pd.DataFrame(rows)
    cols = ["wait", "tp", "cat", "N", "n_trades", "fill_%", "win_%", "tp_%", "time_%", "cat_%",
            "avg_bars", "avg_ret_%", "median_%", "pf", "t_stat", "years_pos",
            "worst_year_%", "worst_trade_%", "p5_%", "sum_ret_%"]
    res = res[[c for c in cols if c in res.columns]]
    res.to_csv(out_dir / "sweep_time_stop_results.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fmt = lambda v: f"{v:,.3f}"

    ok = res[res["n_trades"] >= a.min_trades].sort_values("avg_ret_%", ascending=False)
    print("\n" + "=" * 90 + f"\nTOP {a.top} by avg_ret_% (n_trades >= {a.min_trades})\n" + "=" * 90)
    print(ok.head(a.top).to_string(index=False, float_format=fmt))

    print("\n" + "=" * 90 + "\nPENGARUH TIAP PARAMETER (rata-rata avg_ret_% atas parameter lain)\n" + "=" * 90)
    for p in ["tp", "cat", "N"]:
        print(f"\n{p}:")
        print(res.groupby(p)[["avg_ret_%", "pf", "win_%", "worst_trade_%"]].mean().to_string(float_format=fmt))

    best_cat = ok.iloc[0]["cat"] if len(ok) else res["cat"].iloc[0]
    sub = res[(res["cat"] == best_cat) & (res["wait"] == waits[0])]
    print("\n" + "=" * 90 + f"\nPIVOT avg_ret_% : TP x N  (cat={best_cat})\n" + "=" * 90)
    print(sub.pivot(index="tp", columns="N", values="avg_ret_%").to_string(float_format=fmt))
    print(f"\nPIVOT PF : TP x N  (cat={best_cat})")
    print(sub.pivot(index="tp", columns="N", values="pf").to_string(float_format=fmt))

    print("\nCara baca: kandidat layak = avg_ret_% > 0 DAN pf > 1.2 DAN years_pos mayoritas,")
    print("dengan tetangga grid (tp/N sebelah) juga positif. Perhatikan worst_trade_% dan p5_%:")
    print("tanpa SL ketat, ekor kerugian lebih besar -> ukuran posisi/leverage harus disesuaikan.")
    print(f"[+] Semua hasil: {out_dir / 'sweep_time_stop_results.csv'}")


if __name__ == "__main__":
    main()