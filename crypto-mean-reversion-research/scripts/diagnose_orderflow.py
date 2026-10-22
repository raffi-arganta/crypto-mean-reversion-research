"""
Diagnostik order flow (TRAIN saja): apakah data 15m / funding / open interest membedakan
trade kandidat yang berakhir TP dari yang berakhir SLGAP (harga sudah melewati SL saat
SL boleh dipasang)?

Input : candidate_trades_resid_breadth.csv (kolom entry_time, direction, net_return, pair, exit_type)
        + data hasil download_binance_vision.py (klines_15m, funding, metrics)
Waktu : entry_time di CSV = timestamp bar 1h SINYAL. Keputusan entry di akhir bar itu
        (= entry_time + 1 jam). Semua fitur hanya memakai data SEBELUM titik itu -> tanpa lookahead.

Fitur (diorientasikan: nilai BESAR = lebih searah/menguntungkan posisi, kecuali dinyatakan lain):
  d_sig        delta ratio jam sinyal = (taker_buy - taker_sell)/volume (long: +, = pembeli dominan)
  d_last15     delta ratio bar 15m terakhir jam sinyal
  d_first45    delta ratio 45 menit pertama
  d_turn       d_last15 - d_first45  (positif = tekanan jual/beli sudah berbalik ke arah posisi)
  d_prev2h     delta ratio 2 jam sebelum jam sinyal
  vol_spike    volume jam sinyal / rata-rata volume per jam 24 jam sebelumnya (klimaks)
  vol_last_ratio  volume bar 15m terakhir / rata-rata 3 bar sebelumnya (<1 = volume mengering)
  ret_last15   return bar 15m terakhir (%), searah posisi = positif
  rng_last15   range bar 15m terakhir / median range 15m 24 jam sebelumnya
  pos_last15   posisi close di range bar 15m terakhir (1 = di sisi menguntungkan)
  bounce_pct   jarak close terakhir dari ekstrem jam sinyal (%) = pantulan yang sudah terjadi
  fund_orient  -sign * funding rate terakhir (%) ; positif = sisi lawan padat (potensi squeeze)
  oi_drop_1h / oi_drop_4h  penurunan OI (%) ; positif = deleveraging/flush (TIDAK diorientasikan)
  tkr_orient   sign * log(taker long/short vol ratio)
oi_drop memakai JUMLAH kontrak OI (sum_open_interest), bukan nilai USD yang ikut harga.
OI/taker ratio hanya ada sejak ~2021-12 untuk altcoin (BTC sejak 2020-09).

Interpretasi AUC:
  auc_slgap < 0.5 : nilai fitur besar -> LEBIH JARANG SLGAP (fitur "pengaman")
  auc_tp    > 0.5 : nilai fitur besar -> LEBIH SERING TP
  Dengan ~1500 trade, noise AUC sekitar +/-0.017 (1 SE). Anggap bermakna kalau |AUC-0.5| >= 0.05
  DAN arahnya sama di paruh waktu pertama dan kedua (kolom _h1/_h2). Ada ~15 fitur x 2 target:
  satu-dua yang "lolos" secara kebetulan itu wajar.

Contoh:
  py diagnose_orderflow.py --trades <candidate_trades_resid_breadth.csv> ^
     --bv-dir <...\\binance_vision> --out-dir <...\\diag_orderflow>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NS_H = 3600 * 10 ** 9
NS_15M = 900 * 10 ** 9
FEATURES = ["d_sig", "d_last15", "d_first45", "d_turn", "d_prev2h", "vol_spike", "vol_last_ratio",
            "ret_last15", "rng_last15", "pos_last15", "bounce_pct", "fund_orient",
            "oi_drop_1h", "oi_drop_4h", "tkr_orient"]


# ───────────────────────── loader ─────────────────────────
def to_ns(series):
    return pd.to_datetime(series).to_numpy().astype("datetime64[ns]").astype("int64")


def load_klines(path):
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return {
        "t": to_ns(df["timestamp"]),
        "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
        "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
        "v": df["volume"].to_numpy(float), "tb": df["taker_buy_volume"].to_numpy(float),
    }


def load_funding(path):
    if not path.exists():
        return None
    df = pd.read_csv(path).sort_values("timestamp")
    return {"t": to_ns(df["timestamp"]), "r": df["funding_rate"].to_numpy(float)}


def load_metrics(path):
    if not path.exists():
        return None
    df = pd.read_csv(path).sort_values("timestamp")
    return {"t": to_ns(df["timestamp"]),
            "oi": df["sum_open_interest"].to_numpy(float),   # JUMLAH kontrak (bukan nilai USD: nilai USD ikut harga)
            "tkr": df["sum_taker_long_short_vol_ratio"].to_numpy(float)}


def asof(arr_t, arr_v, t, max_age):
    j = np.searchsorted(arr_t, t, side="right") - 1
    if j < 0 or t - arr_t[j] > max_age:
        return np.nan
    return arr_v[j]


# ───────────────────────── fitur per trade ─────────────────────────
def trade_features(k, fund, met, t0, sign):
    """t0 = entry_time (ns) bar 1h sinyal. Return dict fitur (NaN jika data tidak lengkap)."""
    out = {f: np.nan for f in FEATURES}
    end = t0 + NS_H
    i1 = int(np.searchsorted(k["t"], end, side="left"))   # indeks pertama bar 15m SETELAH jam sinyal
    i0 = i1 - 4
    if i0 < 104 or i1 > len(k["t"]):
        return out
    if k["t"][i0] != t0 or k["t"][i1 - 1] != end - NS_15M:
        return out  # bar 15m jam sinyal tidak lengkap

    v, tb, o, h, l, c = k["v"], k["tb"], k["o"], k["h"], k["l"], k["c"]
    delta = 2 * tb - v

    def ratio(a, b):
        s = v[a:b].sum()
        return delta[a:b].sum() / s if s > 0 else np.nan

    out["d_sig"] = sign * ratio(i0, i1)
    out["d_last15"] = sign * ratio(i1 - 1, i1)
    out["d_first45"] = sign * ratio(i0, i1 - 1)
    out["d_turn"] = out["d_last15"] - out["d_first45"]
    out["d_prev2h"] = sign * ratio(i0 - 8, i0)

    prev_mean_v = v[i0 - 96:i0].mean()
    if prev_mean_v > 0:
        out["vol_spike"] = v[i0:i1].sum() / (4 * prev_mean_v)
    m3 = v[i0:i1 - 1].mean()
    if m3 > 0:
        out["vol_last_ratio"] = v[i1 - 1] / m3

    last = i1 - 1
    out["ret_last15"] = sign * (c[last] - o[last]) / o[last] * 100
    med_rng = np.median(h[i0 - 96:i0] - l[i0 - 96:i0])
    if med_rng > 0:
        out["rng_last15"] = (h[last] - l[last]) / med_rng
    rng = h[last] - l[last]
    if rng > 0:
        pos = (c[last] - l[last]) / rng
        out["pos_last15"] = pos if sign > 0 else 1 - pos
    if sign > 0:
        ext = l[i0:i1].min()
        out["bounce_pct"] = (c[last] - ext) / ext * 100
    else:
        ext = h[i0:i1].max()
        out["bounce_pct"] = (ext - c[last]) / ext * 100

    if fund is not None:
        fr = asof(fund["t"], fund["r"], end, 9 * NS_H)
        if np.isfinite(fr):
            out["fund_orient"] = -sign * fr * 100
    if met is not None:
        oi_now = asof(met["t"], met["oi"], end, 20 * 60 * 10 ** 9)
        for name, hrs in (("oi_drop_1h", 1), ("oi_drop_4h", 4)):
            oi_prev = asof(met["t"], met["oi"], end - hrs * NS_H, 20 * 60 * 10 ** 9)
            if np.isfinite(oi_now) and np.isfinite(oi_prev) and oi_prev > 0:
                out[name] = -(oi_now / oi_prev - 1) * 100
        tk = asof(met["t"], met["tkr"], end, 20 * 60 * 10 ** 9)
        if np.isfinite(tk) and tk > 0:
            out["tkr_orient"] = sign * np.log(tk)
    return out


# ───────────────────────── statistik ─────────────────────────
def auc(x, y):
    m = x.notna()
    x, y = x[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 < 20 or n0 < 20:
        return np.nan
    rk = x.rank()
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pf(r):
    gl = -r[r <= 0].sum()
    return r[r > 0].sum() / gl if gl > 0 else np.inf


def group_metrics(g):
    r = g["net_return"]
    return pd.Series({
        "n": len(g), "tp_%": g["y_tp"].mean() * 100, "slgap_%": g["y_slgap"].mean() * 100,
        "avg_ret_%": r.mean() * 100, "pf": pf(r),
    })


def bucket_table(df, feat, q=5):
    d = df[df[feat].notna()].copy()
    if d[feat].nunique() < q:
        return None
    d["b"] = pd.qcut(d[feat].rank(method="first"), q, labels=False)
    rows = {}
    for b, g in d.groupby("b"):
        m = group_metrics(g)
        m["range"] = f"{g[feat].min():.3g}..{g[feat].max():.3g}"
        rows[b] = m
    t = pd.DataFrame(rows).T
    t = t[["range", "n", "tp_%", "slgap_%", "avg_ret_%", "pf"]]
    for c in ["n", "tp_%", "slgap_%", "avg_ret_%", "pf"]:
        t[c] = t[c].astype(float)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--bv-dir", required=True, help="folder binance_vision (isi: klines_15m, funding, metrics)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    bv = Path(a.bv_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(a.trades, parse_dates=["entry_time", "exit_time"])
    tr = tr[tr["entry_time"] <= pd.Timestamp(a.train_end)].sort_values("entry_time").reset_index(drop=True)
    tr["y_slgap"] = (tr["exit_type"] == "SLGAP").astype(int)
    tr["y_tp"] = (tr["exit_type"] == "TP").astype(int)
    print(f"[*] {len(tr)} trade | {tr['pair'].nunique()} pair | "
          f"TP={tr['y_tp'].mean()*100:.1f}% SLGAP={tr['y_slgap'].mean()*100:.1f}% "
          f"avg_ret={tr['net_return'].mean()*100:+.3f}%")

    feats = []
    cache, miss_pairs = {}, set()
    for r in tr.itertuples(index=False):
        base = r.pair
        if base not in cache:
            sym = f"{base}USDT"
            kp = bv / f"klines_{a.interval}" / f"{sym}.csv"
            if not kp.exists():
                cache[base] = None
                miss_pairs.add(base)
            else:
                cache[base] = (load_klines(kp), load_funding(bv / "funding" / f"{sym}.csv"),
                               load_metrics(bv / "metrics" / f"{sym}.csv"))
        if cache[base] is None:
            feats.append({f: np.nan for f in FEATURES})
            continue
        k, fund, met = cache[base]
        t0 = int(pd.Timestamp(r.entry_time).to_datetime64().astype("datetime64[ns]").astype("int64"))
        sign = 1.0 if r.direction == "long" else -1.0
        feats.append(trade_features(k, fund, met, t0, sign))

    df = pd.concat([tr, pd.DataFrame(feats)], axis=1)
    df.to_csv(out_dir / "trade_orderflow_features.csv", index=False)
    if miss_pairs:
        print(f"[!] klines 15m tidak ada untuk: {sorted(miss_pairs)} (trade-nya tanpa fitur)")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:,.3f}"

    print("\n" + "=" * 90 + "\nCAKUPAN FITUR\n" + "=" * 90)
    cov = pd.DataFrame({"n_ada": df[FEATURES].notna().sum(),
                        "%": df[FEATURES].notna().mean() * 100})
    print(cov.to_string(float_format=lambda v: f"{v:,.1f}"))

    half = len(df) // 2
    h1, h2 = df.iloc[:half], df.iloc[half:]
    rows = []
    for f in FEATURES:
        rows.append({
            "feature": f, "n": int(df[f].notna().sum()),
            "auc_slgap": auc(df[f], df["y_slgap"]),
            "auc_slgap_h1": auc(h1[f], h1["y_slgap"]), "auc_slgap_h2": auc(h2[f], h2["y_slgap"]),
            "auc_tp": auc(df[f], df["y_tp"]),
            "auc_tp_h1": auc(h1[f], h1["y_tp"]), "auc_tp_h2": auc(h2[f], h2["y_tp"]),
        })
    A = pd.DataFrame(rows).set_index("feature")
    A["edge"] = np.fmax((A["auc_slgap"] - 0.5).abs(), (A["auc_tp"] - 0.5).abs())
    A = A.sort_values("edge", ascending=False)
    A.to_csv(out_dir / "orderflow_auc.csv")

    print("\n" + "=" * 90 + "\nAUC per fitur (0.5 = tidak informatif) + kestabilan paruh waktu (h1 vs h2)\n" + "=" * 90)
    print(A.to_string(float_format=fmt))

    print("\n" + "=" * 90 + f"\nBUCKET {a.top} fitur teratas (5 kuantil; bucket 0 = nilai terkecil)\n" + "=" * 90)
    base_row = group_metrics(df)
    print(f"BASELINE: n={int(base_row['n'])} tp={base_row['tp_%']:.1f}% slgap={base_row['slgap_%']:.1f}% "
          f"avg_ret={base_row['avg_ret_%']:+.3f}% pf={base_row['pf']:.3f}")
    top = list(A.index[:a.top])
    for f in top:
        t = bucket_table(df, f)
        if t is None:
            continue
        print(f"\n--- {f} (auc_slgap={A.loc[f,'auc_slgap']:.3f}, auc_tp={A.loc[f,'auc_tp']:.3f}) ---")
        print(t.to_string(float_format=fmt))
        t.to_csv(out_dir / f"bucket_{f}.csv")

    print("\n" + "=" * 90 + "\nUJI FILTER SEDERHANA (in-sample, hanya indikasi): simpan separuh 'aman' tiap fitur teratas\n" + "=" * 90)
    rows = [("BASELINE", base_row)]
    for f in top[:3]:
        d = df[df[f].notna()]
        med = d[f].median()
        safer_high = (A.loc[f, "auc_slgap"] < 0.5) if np.isfinite(A.loc[f, "auc_slgap"]) else (A.loc[f, "auc_tp"] > 0.5)
        keep = d[d[f] >= med] if safer_high else d[d[f] < med]
        rows.append((f"{f} {'>=' if safer_high else '<'} median({med:.3g})", group_metrics(keep)))
    tab = pd.DataFrame({k: v for k, v in rows}).T
    tab["kept_%"] = tab["n"] / len(df) * 100
    print(tab[["n", "kept_%", "tp_%", "slgap_%", "avg_ret_%", "pf"]].to_string(float_format=fmt))

    print("\nCara baca: lanjut ke pengujian eksekusi 15m HANYA jika ada fitur dengan |AUC-0.5| >= 0.05,")
    print("arah sama di h1 dan h2, dan bucket-nya monoton. Kalau semua AUC dekat 0.5, order flow 15m")
    print("tidak menambah informasi dan nomor 4 sebaiknya dihentikan di sini.")
    print(f"[+] Hasil: {out_dir}")


if __name__ == "__main__":
    main()