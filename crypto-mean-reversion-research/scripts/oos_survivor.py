"""
OOS SATU KALI untuk kandidat terkunci:  C_zres 1h  LONG-ONLY

PARAMETER TERKUNCI (jangan diubah setelah melihat hasil OOS):
  sinyal  : z-score window 100 dari akumulasi return RESIDUAL terhadap BTC (beta rolling 500 bar)
  entry   : LONG saat z <= -4.0, di open bar setelah sinyal
  exit    : open setelah 3 bar (time stop). Tanpa TP, tanpa SL, tanpa catastrophic stop.
  porto   : maks 5 posisi bersamaan, prioritas |z| terbesar, notional per posisi = 20% equity awal
            (total exposure <= 100%, tanpa leverage), non-compounding, MTM per jam
  fee     : 0.12% round trip (stres 0.09% .. 0.25%)
  pair    : universe yang sama dengan train (pairs_train_selected.txt), BTC hanya hedge

KRITERIA LULUS (ditulis SEBELUM OOS dijalankan):
  C1  net rata-rata per trade > 0 pada fee 0.12%
  C2  net rata-rata per trade > 0 pada fee 0.20%
  C3  portofolio pada fee 0.20%: return/maxDD >= 1.0 dan return tahunan > 0
  C4  maxDD (MTM) pada fee 0.12% <= 26% (sekitar 2x DD train 12-13%)
  C5  jumlah trade >= 100 (kalau kurang -> INCONCLUSIVE, bukan lulus)
  Kalau salah satu C1-C4 gagal: kandidat DIBUANG. Tidak ada tuning ulang di OOS.

Referensi train (fee 0.12% / 0.20%), untuk perbandingan:
  validation: ann 16.2% / 11.5%, maxDD -9.1% / -9.7%, ret/DD 2.68 / 1.79 ; net per trade 0.28% / 0.20%

Butuh di folder yang sama: screen_ideas.py, robust_survivor.py.
Data 1h harus mencakup periode OOS (file futures_1h yang sama, sudah di-update sampai tanggal OOS).

Contoh:
  py oos_survivor.py --pairs <pairs_train_selected.txt> --data-dir <futures_1h> ^
     --test-start 2024-01-01 --out-dir <oos_result>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import screen_ideas as si
import robust_survivor as rs

LOCKED = dict(zw=100, thr=4.0, H=3, slots=5, f=0.20, side="long", fee=0.0012)


def day_cluster_t(net, times):
    dm = pd.Series(net).groupby(pd.DatetimeIndex(times).floor("D")).mean()
    if len(dm) < 3 or dm.std(ddof=1) == 0:
        return np.nan
    return dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--test-end", default="2100-01-01", help="default: seluruh data setelah test-start")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--fees", default="0.0009,0.0012,0.0015,0.0020,0.0025")
    # parameter terkunci (jangan diubah kecuali untuk tes teknis):
    ap.add_argument("--min-qvol", type=float, default=None)
    ap.add_argument("--min-age-days", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--zw", type=int, default=LOCKED["zw"])
    ap.add_argument("--thr", type=float, default=LOCKED["thr"])
    ap.add_argument("--H", type=int, default=LOCKED["H"])
    ap.add_argument("--slots", type=int, default=LOCKED["slots"])
    ap.add_argument("--f", type=float, default=LOCKED["f"])
    ap.add_argument("--side", default=LOCKED["side"])
    a = ap.parse_args()

    rs.SIDE = a.side
    test_start = pd.Timestamp(a.test_start)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(a.pairs, "r", encoding="utf-8") as fh:
        pairs = [x.strip() for x in fh if x.strip() and not x.strip().startswith("#")]
    pairs = list(dict.fromkeys(["BTC"] + pairs))

    print(f"[*] load {len(pairs)} pair (termasuk warm-up sebelum {test_start.date()}) ...")
    P = si.load_panels(pairs, Path(a.data_dir), pd.Timestamp(a.test_end))
    index, cols = P["C"].index, list(P["C"].columns)
    if index[-1] < test_start:
        raise SystemExit(f"Data berakhir {index[-1]} sebelum test-start {test_start}. Update data 1h dulu.")
    idx0 = int(index.searchsorted(test_start))
    if idx0 < 700:
        raise SystemExit("Warm-up kurang (butuh >= 600 bar sebelum test-start untuk beta/z).")
    n_bars = len(index)
    Cn = P["C"].to_numpy()
    years = (index[-1] - index[idx0]) / pd.Timedelta(days=365.25)

    btc = P["C"]["BTC"].iloc[idx0:].dropna()
    btc_ret = (btc.iloc[-1] / btc.iloc[0] - 1) * 100
    btc_dd = ((btc / btc.cummax()) - 1).min() * 100
    print("\n" + "=" * 90)
    print(f"OOS: {index[idx0]} -> {index[-1]}  ({years:.2f} tahun, {n_bars-idx0:,} bar, {P['C'].shape[1]} pair)")
    print(f"Konteks pasar: BTC {btc_ret:+.1f}% selama periode, maxDD BTC {btc_dd:.1f}%")
    print(f"Parameter: zw={a.zw} thr={a.thr} H={a.H} side={a.side} slots={a.slots} f={a.f} (tanpa SL/TP/cat)")
    print("=" * 90)

    res = si.resid_returns(P["C"], si.BETA_W["1h"])
    rs.ELIG = rs.make_elig(P, a.min_qvol, a.min_age_days, a.top_n)
    if rs.ELIG is not None:
        print(f"[*] universe point-in-time: rata-rata {rs.ELIG[idx0:].sum(axis=1).mean():.1f} pair eligible per jam di periode OOS "
              f"(min_qvol={a.min_qvol}, min_age={a.min_age_days}, top_n={a.top_n})")
    ev = rs.build_events(P, res, a.zw, a.thr, a.H, None, LOCKED["fee"])
    m = ev["t"] >= idx0
    ev = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}
    n_opp = len(rs.nonoverlap(ev))
    acc = rs.simulate_slots(ev, a.slots)
    if len(acc) == 0:
        raise SystemExit("Tidak ada trade di periode OOS.")
    t_times = index[ev["t"][acc]]
    gross = ev["gross"][acc]
    print(f"Peluang entry (tanpa batas slot): {n_opp} | trade terambil: {len(acc)} | "
          f"dilewati karena slot penuh: {max(0, (1 - len(acc)/max(n_opp,1))*100):.1f}%")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:,.3f}"
    fees = [float(x) for x in a.fees.split(",")]

    # ───── 1. per trade ─────
    net12 = gross - 0.0012
    print("\n" + "=" * 90 + "\n1. PER TRADE (fee 0.12%)\n" + "=" * 90)
    d = rs.dist_row(net12)
    d["win_%"] = (net12 > 0).mean() * 100
    d["t_day_cluster"] = day_cluster_t(net12, t_times)
    print(pd.Series(d).to_frame("OOS").T.to_string(float_format=fmt))

    # ───── 2. stres fee per trade + portofolio ─────
    print("\n" + "=" * 90 + "\n2. STRES FEE: per trade dan portofolio (slots/f terkunci)\n" + "=" * 90)
    rows = []
    port = {}
    for fee in fees:
        net = ev["gross"] - fee
        curve, conc = rs.unit_curve(ev, acc, Cn, n_bars, net)
        c = curve[idx0:]
        eq = 1 + a.f * c
        total = (eq[-1] - 1) * 100
        dd = float((eq / np.maximum.accumulate(eq) - 1).min()) * 100
        daily = pd.Series(eq, index=index[idx0:]).groupby(index[idx0:].floor("D")).last()
        r = daily.pct_change().dropna()
        sharpe = r.mean() / r.std(ddof=1) * np.sqrt(365) if r.std(ddof=1) > 0 else np.nan
        row = {"fee_%": fee * 100, "net/trade_%": (gross - fee).mean() * 100,
               "PF": rs.pf(gross - fee), "total_%": total, "ann_%": total / years,
               "maxDD_%": dd, "ret/DD": total / abs(dd) if dd < 0 else np.nan, "sharpe": sharpe,
               "avg_conc": conc[idx0:].mean()}
        rows.append(row)
        port[round(fee, 4)] = row
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))
    print("Catatan: return non-compounding (% dari equity awal), DD dari close per jam (intrabar tidak tertangkap).")

    # ───── 3. per tahun / kuartal ─────
    tr = pd.DataFrame({"entry_time": t_times, "pair": np.array(cols)[ev["col"][acc]],
                       "net": net12, "exit_idx": ev["xb"][acc]})
    tr["year"] = tr["entry_time"].dt.year
    tr["quarter"] = tr["entry_time"].dt.to_period("Q").astype(str)
    print("\n" + "=" * 90 + "\n3. PER PERIODE (fee 0.12%; pnl_% = jumlah f x net, non-compounding)\n" + "=" * 90)
    for key in ("year", "quarter"):
        g = tr.groupby(key)["net"].agg(n="count", mean_pct=lambda x: x.mean() * 100,
                                       win_pct=lambda x: (x > 0).mean() * 100, pf=rs.pf,
                                       pnl_pct=lambda x: x.sum() * a.f * 100)
        print(g.to_string(float_format=fmt))
        print()
    pq = tr.groupby("quarter")["net"].sum()
    print(f"Kuartal profit: {(pq > 0).sum()} / {len(pq)}")

    pp = tr.groupby("pair")["net"].agg(n="count", sum_pct=lambda x: x.sum() * 100).sort_values("sum_pct", ascending=False)
    pos = pp.loc[pp["sum_pct"] > 0, "sum_pct"].sum()
    print(f"Pair profit: {(pp['sum_pct'] > 0).sum()} / {len(pp)} | 3 pair teratas = "
          f"{pp['sum_pct'].head(3).sum() / pos * 100 if pos > 0 else float('nan'):.0f}% dari profit kotor")
    print("10 trade terburuk:")
    print(tr.nsmallest(10, "net")[["entry_time", "pair", "net"]].assign(net=lambda x: x["net"] * 100)
          .to_string(index=False, float_format=fmt))

    # ───── 4. ekspor ─────
    out = pd.DataFrame({
        "entry_time": t_times, "exit_time": index[ev["xb"][acc]], "direction": "long",
        "net_return": net12, "pair": np.array(cols)[ev["col"][acc]], "exit_type": "TIME"})
    out.sort_values("entry_time").to_csv(out_dir / "oos_trades_c_zres_long.csv", index=False)

    # ───── 5. verdict ─────
    r12, r20 = port[0.0012], port[0.002]
    checks = [
        ("C1 net/trade > 0 @ fee 0.12%", r12["net/trade_%"] > 0, f"{r12['net/trade_%']:+.3f}%"),
        ("C2 net/trade > 0 @ fee 0.20%", r20["net/trade_%"] > 0, f"{r20['net/trade_%']:+.3f}%"),
        ("C3 ret/DD >= 1 & ann > 0 @ fee 0.20%", (r20["ret/DD"] >= 1.0) and (r20["ann_%"] > 0),
         f"ret/DD={r20['ret/DD']:.2f}, ann={r20['ann_%']:+.1f}%"),
        ("C4 maxDD <= 26% @ fee 0.12%", abs(r12["maxDD_%"]) <= 26, f"{r12['maxDD_%']:.1f}%"),
    ]
    print("\n" + "=" * 90 + "\nVERDICT (kriteria terkunci)\n" + "=" * 90)
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ->  {val}")
    n_ok = len(acc) >= 100
    print(f"  [{'OK  ' if n_ok else 'WARN'}] C5 jumlah trade >= 100  ->  {len(acc)}")
    if not n_ok:
        print("\nHASIL: INCONCLUSIVE (sampel terlalu kecil). Jangan disimpulkan lulus atau gagal; kumpulkan data lebih lama.")
    elif all(ok for _, ok, _ in checks):
        print("\nHASIL: LULUS kriteria. Lanjut ke demo kecil (modal kecil, f <= 0.1) untuk mengukur slippage nyata,")
        print("       BUKAN langsung live dengan ukuran penuh. Parameter tetap terkunci.")
    else:
        print("\nHASIL: GAGAL. Kandidat dibuang. Jangan menyetel ulang parameter di OOS yang sama.")
    print(f"\n[+] Trade OOS: {out_dir / 'oos_trades_c_zres_long.csv'}  (schema analyze_trades.py)")


if __name__ == "__main__":
    main()