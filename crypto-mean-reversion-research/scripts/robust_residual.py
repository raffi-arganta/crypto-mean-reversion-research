"""
Uji KETAHANAN kandidat: z residual (hedge BTC) + filter breadth  -- TRAIN saja.

Bagian 1 - Grid parameter di sekitar kandidat:
    beta-window x breadth-z x breadth-max x thr x exit
    Pertanyaan: apakah hasil positif di SEBAGIAN BESAR sel grid (plateau), atau
    hanya satu sel (noise/overfit)?
Bagian 2 - Bedah kandidat tengah:
    per tahun, per arah, per pair (konsentrasi), leave-top-pairs-out,
    t-stat & bootstrap CI yang di-cluster per HARI (trade antar pair saling berkorelasi),
    stres fee/slippage.
Bagian 3 - Ekspor trade kandidat ke CSV (schema analyze_trades.py):
    entry_time, exit_time, direction, net_return, pair, exit_type

Butuh di folder yang sama: sweep_residual.py, sweep_entry_exit.py, sweep_time_stop.py,
zscore.py, atr.py, adx.py, half_life.py, signal_construction.py. Data harus ada BTCUSDT.csv.

Contoh:
  py robust_residual.py --pairs <pairs_train_selected.txt> --data-dir <futures_1h> ^
     --train-end 2023-12-31 --out-dir <robust>
"""

import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import sweep_residual as sr
from sweep_residual import EXITS, FEE
from zscore import calculate_zscore
from atr import calculate_atr, calculate_atr_percentile


# ───────────────────────── data (dipecah supaya murah di-grid) ─────────────────────────
def base_prepare(pairs, data_dir, a):
    btc_df = sr.read_df("BTC", data_dir)
    if btc_df is None:
        raise SystemExit("BTCUSDT.csv tidak ditemukan di --data-dir.")
    bts = pd.DatetimeIndex(btc_df["timestamp"])
    btc_ret = pd.Series(np.log(btc_df["close"].to_numpy(float)), index=bts).diff()
    btc_atrp = pd.Series(
        calculate_atr_percentile(btc_df, atr_n=a.atr_n, lookback=a.atr_lookback).to_numpy(float), index=bts)

    data = {}
    for base in pairs:
        df = sr.read_df(base, data_dir)
        if df is None:
            print(f"  [!] {base}: skip")
            continue
        ts = pd.DatetimeIndex(df["timestamp"])
        data[base] = {
            "name": base, "ts": ts,
            "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
            "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
            "atr": calculate_atr(df, n=a.atr_n).to_numpy(float),
            "atr_pct": calculate_atr_percentile(df, atr_n=a.atr_n, lookback=a.atr_lookback).to_numpy(float),
            "z_raw": calculate_zscore(df, window=a.z_window).to_numpy(float).copy(),
            "r": pd.Series(np.log(df["close"].to_numpy(float)), index=ts).diff(),
            "rb": btc_ret.reindex(ts),
            "btc_atrp": btc_atrp.reindex(ts).to_numpy(float),
        }
    return data


def add_resid(data, beta_window, z_window):
    for base, d in data.items():
        if base == "BTC":
            d["z_res"] = np.full(len(d["o"]), np.nan)
            continue
        r, rb, w = d["r"], d["rb"], beta_window
        beta = r.rolling(w, min_periods=w // 2).cov(rb) / rb.rolling(w, min_periods=w // 2).var()
        inc = r - beta.shift(1) * rb
        cnt = inc.notna().rolling(z_window).sum()
        x = pd.DataFrame({"x": inc.fillna(0).cumsum().to_numpy()})
        z = calculate_zscore(x, window=z_window, price_col="x").to_numpy(float).copy()
        z[(cnt < z_window).to_numpy()] = np.nan
        d["z_res"] = z


def add_breadth(data, breadth_z):
    Z = pd.DataFrame({b: pd.Series(d["z_raw"], index=d["ts"]) for b, d in data.items()})
    n_long = (Z <= -breadth_z).sum(axis=1)
    n_short = (Z >= breadth_z).sum(axis=1)
    for d in data.values():
        d["br_long"] = n_long.reindex(d["ts"]).to_numpy(float)
        d["br_short"] = n_short.reindex(d["ts"]).to_numpy(float)


# ───────────────────────── backtest dengan detail trade ─────────────────────────
def backtest_full(data, train_end, thr, regime, ex_cfg, ns):
    recs = []
    for base, d in data.items():
        el, es = sr.build_signals(d, "resid", thr, regime, ns)
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
            ex = t + 1
            entry = d["o"][ex]
            res = sr.run_exit(ex_cfg, direction, entry, atr, d, ex)
            if res is None:
                last_exit = n
                continue
            off, px, typ = res
            last_exit = ex + off - 1
            recs.append({
                "entry_time": d["ts"][t], "exit_time": d["ts"][last_exit],
                "direction": "long" if direction == 1 else "short",
                "net_return": direction * (px - entry) / entry - FEE,
                "pair": base, "exit_type": typ,
            })
    return pd.DataFrame(recs)


def pf(r):
    gl = -r[r <= 0].sum()
    return r[r > 0].sum() / gl if gl > 0 else np.inf


def cluster_stats(tr, n_boot=2000, seed=0):
    day = tr["entry_time"].dt.floor("D")
    g = tr.groupby(day)["net_return"].agg(["sum", "count"])
    dm = g["sum"] / g["count"]
    t_cluster = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 1 and dm.std(ddof=1) > 0 else np.nan
    rng = np.random.default_rng(seed)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return len(g), t_cluster, np.percentile(boots, 2.5) * 100, np.percentile(boots, 97.5) * 100


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--beta-grid", default="250,500,1000")
    ap.add_argument("--bz-grid", default="2.0,2.5,3.0")
    ap.add_argument("--bmax-grid", default="3,5,8,12")
    ap.add_argument("--thr-grid", default="3.5,3.8")
    ap.add_argument("--exits", default="live_mh3,time")
    ap.add_argument("--c-beta", type=int, default=500)
    ap.add_argument("--c-bz", type=float, default=2.5)
    ap.add_argument("--c-bmax", type=int, default=5)
    ap.add_argument("--c-thr", type=float, default=3.8)
    ap.add_argument("--c-exit", default="live_mh3")
    ap.add_argument("--z-window", type=int, default=50)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--atr-lookback", type=int, default=100)
    a = ap.parse_args()

    train_end = pd.Timestamp(a.train_end)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.pairs, "r", encoding="utf-8") as f:
        pairs = [x.strip() for x in f if x.strip() and not x.strip().startswith("#")]

    print(f"[*] load {len(pairs)} pair ...")
    data = base_prepare(pairs, Path(a.data_dir), a)
    print(f"[*] {len(data)} pair siap | train-end={train_end.date()}")

    betas = [int(x) for x in a.beta_grid.split(",")]
    bzs = [float(x) for x in a.bz_grid.split(",")]
    bmaxs = [int(x) for x in a.bmax_grid.split(",")]
    thrs = [float(x) for x in a.thr_grid.split(",")]
    exits = a.exits.split(",")

    total = len(betas) * len(bzs) * len(bmaxs) * len(thrs) * len(exits)
    print(f"[*] grid ketahanan: {total} konfigurasi ...")
    rows, k = [], 0
    for beta in betas:
        add_resid(data, beta, a.z_window)
        for bz in bzs:
            add_breadth(data, bz)
            for thr, bmax, ex in itertools.product(thrs, bmaxs, exits):
                ns = SimpleNamespace(btc_atr_max=0.8, breadth_max=bmax)
                tr, n_sig = sr.backtest(data, train_end, "resid", thr, "breadth", EXITS[ex], ns)
                s = sr.summarize(tr, n_sig)
                s.update({"beta": beta, "bz": bz, "bmax": bmax, "thr": thr, "exit": ex})
                rows.append(s)
                k += 1
                if k % 24 == 0:
                    print(f"  {k}/{total}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "robust_grid.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fmt = lambda v: f"{v:,.3f}"

    print("\n" + "=" * 90 + "\nBAGIAN 1 - PLATEAU: sebaran avg_ret_% di seluruh grid\n" + "=" * 90)
    for ex in exits:
        g = res[res["exit"] == ex]
        print(f"\nexit={ex}: {len(g)} sel | positif: {(g['avg_ret_%'] > 0).mean()*100:.0f}% | "
              f"PF>1.1: {(g['pf'] > 1.1).mean()*100:.0f}% | median={g['avg_ret_%'].median():.3f}% | "
              f"min={g['avg_ret_%'].min():.3f}% | max={g['avg_ret_%'].max():.3f}%")

    cx = a.c_exit
    base = res[(res["exit"] == cx)]
    print(f"\n--- avg_ret_% : breadth_max x breadth_z (beta={a.c_beta}, thr={a.c_thr}, exit={cx}) ---")
    sub = base[(base["beta"] == a.c_beta) & (base["thr"] == a.c_thr)]
    print(sub.pivot(index="bmax", columns="bz", values="avg_ret_%").to_string(float_format=fmt))
    print(f"\n--- avg_ret_% : beta_window x thr (bz={a.c_bz}, bmax={a.c_bmax}, exit={cx}) ---")
    sub = base[(base["bz"] == a.c_bz) & (base["bmax"] == a.c_bmax)]
    print(sub.pivot(index="beta", columns="thr", values="avg_ret_%").to_string(float_format=fmt))
    print(f"\n--- n_trades : breadth_max x breadth_z (beta={a.c_beta}, thr={a.c_thr}) ---")
    sub = base[(base["beta"] == a.c_beta) & (base["thr"] == a.c_thr)]
    print(sub.pivot(index="bmax", columns="bz", values="n_trades").to_string(float_format=lambda v: f"{v:,.0f}"))

    # ───── Bagian 2: bedah kandidat tengah ─────
    print("\n" + "=" * 90 + f"\nBAGIAN 2 - KANDIDAT: resid thr={a.c_thr} beta={a.c_beta} breadth(z>={a.c_bz}, max={a.c_bmax}) exit={cx}\n" + "=" * 90)
    add_resid(data, a.c_beta, a.z_window)
    add_breadth(data, a.c_bz)
    ns = SimpleNamespace(btc_atr_max=0.8, breadth_max=a.c_bmax)
    tr = backtest_full(data, train_end, a.c_thr, "breadth", EXITS[cx], ns)
    if len(tr) == 0:
        print("Tidak ada trade."); return
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    r = tr["net_return"]
    print(f"n_trades={len(tr)} | avg={r.mean()*100:+.3f}% | gross={(r.mean()+FEE)*100:+.3f}% | "
          f"win={(r>0).mean()*100:.1f}% | PF={pf(r):.3f}")

    n_days, t_cl, lo, hi = cluster_stats(tr)
    print(f"Cluster per hari: {n_days} hari | t-stat cluster={t_cl:.2f} | bootstrap 95% CI avg_ret = [{lo:+.3f}%, {hi:+.3f}%]")
    print("  (CI yang menyentuh/menyeberangi 0 = edge belum terbukti secara statistik)")

    tr["year"] = tr["entry_time"].dt.year
    print("\nPer tahun:")
    print(tr.groupby("year")["net_return"].agg(
        n="count", avg_pct=lambda x: x.mean() * 100, win_pct=lambda x: (x > 0).mean() * 100, pf=pf
    ).to_string(float_format=fmt))

    print("\nPer arah:")
    print(tr.groupby("direction")["net_return"].agg(
        n="count", avg_pct=lambda x: x.mean() * 100, win_pct=lambda x: (x > 0).mean() * 100, pf=pf
    ).to_string(float_format=fmt))

    print("\nPer exit type:")
    print(tr.groupby("exit_type")["net_return"].agg(
        n="count", avg_pct=lambda x: x.mean() * 100).to_string(float_format=fmt))

    pp = tr.groupby("pair")["net_return"].agg(n="count", sum_pct=lambda x: x.sum() * 100,
                                              avg_pct=lambda x: x.mean() * 100).sort_values("sum_pct", ascending=False)
    pos_total = pp.loc[pp["sum_pct"] > 0, "sum_pct"].sum()
    print(f"\nKonsentrasi pair: {(pp['sum_pct'] > 0).sum()}/{len(pp)} pair profit | "
          f"3 pair teratas = {pp['sum_pct'].head(3).sum() / pos_total * 100:.0f}% dari profit kotor")
    print("Top 5 / bottom 5 pair (sum_pct):")
    print(pd.concat([pp.head(5), pp.tail(5)]).to_string(float_format=fmt))
    for kk in (3, 5, 10):
        drop = pp.head(kk).index
        rr = tr.loc[~tr["pair"].isin(drop), "net_return"]
        if len(rr) == 0:
            continue
        print(f"  tanpa {kk} pair teratas: avg={rr.mean()*100:+.3f}% PF={pf(rr):.3f} n={len(rr)}")

    print("\nStres fee/slippage (round trip, asumsi dasar 0.09%):")
    for fee in (0.0009, 0.0012, 0.0015, 0.0020):
        rr = r + FEE - fee
        print(f"  fee {fee*100:.2f}%: avg={rr.mean()*100:+.3f}% PF={pf(rr):.3f}")

    out = tr[["entry_time", "exit_time", "direction", "net_return", "pair", "exit_type"]]
    out_path = out_dir / "candidate_trades_resid_breadth.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[+] Trade kandidat: {out_path}  (siap untuk analyze_trades.py)")
    print(f"[+] Grid lengkap  : {out_dir / 'robust_grid.csv'}")
    print("\nCara baca: kandidat layak jika (1) mayoritas sel grid positif dan tetangga tengah juga positif,")
    print("(2) CI bootstrap per-hari > 0 atau hampir, (3) tidak bergantung pada <=3 pair,")
    print("(4) masih positif di fee 0.12-0.15%, (5) tahun-tahun negatif kecil.")


if __name__ == "__main__":
    main()