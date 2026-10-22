"""
Penyaring ide strategi (TRAIN saja): cari edge BESAR dan volume trade BESAR sekaligus.

Semua ide dievaluasi dengan engine yang SAMA dan sederhana supaya adil dibandingkan:
  - entry di open bar setelah sinyal, exit di open setelah H bar (hold tetap, tanpa SL/TP),
  - fee round trip per trade (--fee, default 0.12% = fee + slippage konservatif),
  - satu posisi per pair pada satu waktu, BTC tidak diperdagangkan (hanya dipakai hedge).
Risiko dikontrol lewat diversifikasi, bukan SL ketat (SL ketat terbukti merusak di data kita).

Keluarga ide (semua param ada di grid; timeframe 1h/4h/1d di-resample dari data 1h):
  A. Cross-sectional: long koin terlemah / short terkuat berdasarkan return RESIDUAL (hedge BTC)
     L bar terakhir, rebalance tiap H bar. mode rev (reversal) atau mom (momentum).
  C. Z-score residual per koin (threshold), mode rev atau mom, hold H bar.
  D. Breakout Donchian N bar: trend (ikut) atau fade (lawan), hold H bar.
  E. Funding ekstrem (butuh --bv-dir): fade sisi yang padat atau follow, hold H jam.

Ukuran "edge besar + volume besar" (per segmen):
  net_%/trade   rata-rata return bersih per trade
  trades/yr     jumlah trade per tahun (volume)
  ann_%         return tahunan pada modal rata-rata yang terpakai = sum(net) / rata-rata posisi
                konkuren / tahun. INI yang menggabungkan edge x volume.
  sharpe        Sharpe harian dari pnl yang dinormalisasi modal terpakai
  t             t-stat per-hari (cluster; trade antar pair berkorelasi)

Anti-overfit: periode train dibagi DISCOVERY (< --split) dan VALIDATION (>= --split).
Shortlist dipilih HANYA dari discovery; validation dipakai sebagai konfirmasi independen.
Setelah shortlist, OOS (di luar train) baru boleh disentuh, SEKALI saja.

Contoh:
  py screen_ideas.py --pairs <pairs_train_selected.txt> --data-dir <futures_1h> ^
     --train-end 2023-12-31 --split 2022-07-01 --out-dir <screen> --bv-dir <binance_vision>
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8766.0
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}
BETA_W = {"1h": 500, "4h": 250, "1d": 90}


# ───────────────────────── data ─────────────────────────
def load_panels(pairs, data_dir, train_end):
    O, H, L, C, Q = {}, {}, {}, {}, {}
    end = train_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for b in pairs:
        p = data_dir / f"{b}USDT.csv"
        if not p.exists():
            print(f"  [!] {b}: tidak ada")
            continue
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
        df = df[df.index <= end]
        if len(df) < 1000:
            continue
        O[b], H[b], L[b], C[b] = df["open"], df["high"], df["low"], df["close"]
        Q[b] = df["quote_volume"] if "quote_volume" in df.columns else pd.Series(np.nan, index=df.index)
    P = {k: pd.DataFrame(v).sort_index() for k, v in (("O", O), ("H", H), ("L", L), ("C", C), ("Q", Q))}
    if "BTC" not in P["C"].columns:
        raise SystemExit("BTC wajib ada di --pairs/--data-dir (dipakai sebagai hedge).")
    return P


def resample_panels(P, tf):
    if tf == "1h":
        return P
    rule = {"4h": "4h", "1d": "1D"}[tf]
    return {"O": P["O"].resample(rule).first(), "H": P["H"].resample(rule).max(),
            "L": P["L"].resample(rule).min(), "C": P["C"].resample(rule).last()}


def resid_returns(C, beta_w):
    r = np.log(C).diff()
    btc = r["BTC"]
    cov = r.rolling(beta_w, min_periods=beta_w // 2).cov(btc)
    var = btc.rolling(beta_w, min_periods=beta_w // 2).var()
    beta = cov.div(var, axis=0)
    res = r - beta.shift(1).mul(btc, axis=0)
    res["BTC"] = np.nan
    return res


def load_funding(pairs, bv_dir, index_1h):
    F = {}
    for b in pairs:
        p = bv_dir / "funding" / f"{b}USDT.csv"
        if not p.exists() or b == "BTC":
            continue
        df = pd.read_csv(p)
        ts = pd.to_datetime(df["timestamp"]).dt.round("h")
        s = pd.Series(df["funding_rate"].to_numpy(float), index=ts)
        s = s[~s.index.duplicated()]
        F[b] = s.reindex(index_1h)
    return pd.DataFrame(F, index=index_1h) if F else None


# ───────────────────────── konstruksi trade ─────────────────────────
def fwd_ret(O, H):
    return O.shift(-(1 + H)) / O.shift(-1) - 1


def dedupe(ev, H):
    """Satu posisi per pair: buang sinyal yang jatuh selama hold posisi sebelumnya."""
    arr = ev.to_numpy().copy()
    for j in range(arr.shape[1]):
        nxt = -1
        for i in np.flatnonzero(arr[:, j]):
            if i >= nxt:
                nxt = i + H + 1
            else:
                arr[i, j] = False
    return pd.DataFrame(arr, index=ev.index, columns=ev.columns)


def make_trades(sel_long, sel_short, Rf, hold_h):
    out = []
    for mask, d in ((sel_long, 1), (sel_short, -1)):
        mask = mask.reindex(index=Rf.index, columns=Rf.columns, fill_value=False).astype(bool)
        m = (mask & Rf.notna()).to_numpy()
        r, c = np.nonzero(m)
        if len(r):
            out.append(pd.DataFrame({"entry_time": Rf.index[r], "pair": Rf.columns[c],
                                     "gross": d * Rf.to_numpy()[r, c]}))
    if not out:
        return pd.DataFrame(columns=["entry_time", "pair", "gross", "hold_h"])
    df = pd.concat(out, ignore_index=True)
    df["hold_h"] = hold_h
    return df


def no_btc(df):
    df = df.copy()
    if "BTC" in df.columns:
        df["BTC"] = False
    return df


def idea_A(P, res, L, H, q, mode, tf_h):
    score = res.rolling(L, min_periods=L).sum()
    rank = score.rank(axis=1, pct=True)
    ok = (score.notna().sum(axis=1) >= 10) & (np.arange(len(score)) % H == 0)
    lo, hi = (rank <= q), (rank >= 1 - q)
    lo = lo.mul(ok.astype(int), axis=0).astype(bool)
    hi = hi.mul(ok.astype(int), axis=0).astype(bool)
    sl, ss = (lo, hi) if mode == "rev" else (hi, lo)
    return make_trades(no_btc(sl), no_btc(ss), fwd_ret(P["O"], H), H * tf_h)


def idea_C(P, res, zw, thr, H, mode, tf_h):
    X = res.cumsum()
    z = (X - X.rolling(zw, min_periods=zw).mean()) / X.rolling(zw, min_periods=zw).std(ddof=0)
    lo, hi = no_btc(z <= -thr), no_btc(z >= thr)
    sl, ss = (lo, hi) if mode == "rev" else (hi, lo)
    acc = dedupe(sl | ss, H)
    return make_trades(sl & acc, ss & acc, fwd_ret(P["O"], H), H * tf_h)


def idea_D(P, N, H, mode, tf_h):
    hh = P["H"].rolling(N, min_periods=N).max().shift(1)
    ll = P["L"].rolling(N, min_periods=N).min().shift(1)
    up, dn = no_btc(P["C"] > hh), no_btc(P["C"] < ll)
    sl, ss = (up, dn) if mode == "trend" else (dn, up)
    acc = dedupe(sl | ss, H)
    return make_trades(sl & acc, ss & acc, fwd_ret(P["O"], H), H * tf_h)


def idea_E(P, F, thr, H, mode):
    pos, neg = no_btc(F >= thr), no_btc(F <= -thr)       # pos = long padat, neg = short padat
    sl, ss = (neg, pos) if mode == "fade" else (pos, neg)  # fade: long saat short padat
    acc = dedupe(sl | ss, H)
    return make_trades(sl & acc, ss & acc, fwd_ret(P["O"], H), H)


# ───────────────────────── metrik ─────────────────────────
def seg_metrics(tr, start, end, fee):
    keys = ["n", "trades/yr", "gross_%", "net_%", "win_%", "ann_%", "sharpe", "t"]
    nan = {k: np.nan for k in keys}
    hours = (end - start) / pd.Timedelta(hours=1)
    years = hours / HOURS_PER_YEAR
    s = tr[(tr["entry_time"] >= start) & (tr["entry_time"] < end)]
    n = len(s)
    if n < 30:
        nan["n"] = n
        return nan
    net = s["gross"] - fee
    K = max(s["hold_h"].sum() / hours, 1.0)                    # rata-rata posisi konkuren (min 1 unit)
    exit_day = (s["entry_time"] + pd.to_timedelta(s["hold_h"], unit="h")).dt.floor("D")
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    daily = (net.groupby(exit_day).sum() / K).reindex(days, fill_value=0.0)
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(365) if daily.std(ddof=1) > 0 else np.nan
    dm = net.groupby(s["entry_time"].dt.floor("D")).mean()
    t = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 2 and dm.std(ddof=1) > 0 else np.nan
    return {"n": n, "trades/yr": n / years, "gross_%": s["gross"].mean() * 100, "net_%": net.mean() * 100,
            "win_%": (net > 0).mean() * 100, "ann_%": net.sum() / K / years * 100,
            "sharpe": sharpe, "t": t}


# ───────────────────────── main ─────────────────────────
def build_configs(grid, have_funding):
    cfg = []
    if grid == "small":
        A = dict(L=[6, 24], H=[3, 12], q=[0.2], mode=["rev", "mom"])
        C = dict(zw=[50], thr=[2.5, 3.5], H=[3, 12], mode=["rev", "mom"])
        D = dict(N=[24, 96], H=[6, 24], mode=["trend", "fade"])
        E = dict(thr=[0.0003, 0.001], H=[8, 24], mode=["fade", "follow"])
    else:
        A = dict(L=[3, 6, 12, 24], H=[1, 3, 6, 12], q=[0.1, 0.2], mode=["rev", "mom"])
        C = dict(zw=[24, 50, 100], thr=[2.5, 3.5], H=[1, 3, 6, 12], mode=["rev", "mom"])
        D = dict(N=[12, 24, 48, 96], H=[3, 6, 12, 24], mode=["trend", "fade"])
        E = dict(thr=[0.0002, 0.0005, 0.001], H=[8, 24, 72], mode=["fade", "follow"])
    for tf in TF_HOURS:
        for L, H, q, m in itertools.product(A["L"], A["H"], A["q"], A["mode"]):
            cfg.append(("A_xsec", tf, dict(L=L, H=H, q=q, mode=m)))
        for zw, thr, H, m in itertools.product(C["zw"], C["thr"], C["H"], C["mode"]):
            cfg.append(("C_zres", tf, dict(zw=zw, thr=thr, H=H, mode=m)))
        for N, H, m in itertools.product(D["N"], D["H"], D["mode"]):
            cfg.append(("D_break", tf, dict(N=N, H=H, mode=m)))
    if have_funding:
        for thr, H, m in itertools.product(E["thr"], E["H"], E["mode"]):
            cfg.append(("E_fund", "1h", dict(thr=thr, H=H, mode=m)))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--split", default="2022-07-01", help="batas discovery | validation")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--bv-dir", default=None, help="folder binance_vision (untuk ide E funding)")
    ap.add_argument("--fee", type=float, default=0.0012, help="fee+slippage round trip per trade")
    ap.add_argument("--grid", choices=["small", "full"], default="full")
    ap.add_argument("--min-trades-yr", type=float, default=100, help="volume minimum untuk shortlist")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    train_end = pd.Timestamp(a.train_end)
    split = pd.Timestamp(a.split)
    start = pd.Timestamp(a.start)
    end = train_end + pd.Timedelta(days=1)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(a.pairs, "r", encoding="utf-8") as f:
        pairs = [x.strip() for x in f if x.strip() and not x.strip().startswith("#")]
    pairs = list(dict.fromkeys(["BTC"] + pairs))

    print(f"[*] load {len(pairs)} pair ...")
    P1 = load_panels(pairs, Path(a.data_dir), train_end)
    print(f"[*] panel 1h: {P1['C'].shape[1]} pair x {P1['C'].shape[0]:,} bar | fee={a.fee*100:.2f}% | "
          f"discovery <{split.date()} | validation >={split.date()}")

    F = load_funding(list(P1["C"].columns), Path(a.bv_dir), P1["C"].index) if a.bv_dir else None
    print(f"[*] funding: {'tersedia' if F is not None else 'tidak dipakai (ide E dilewati)'}")

    panels, resids = {}, {}
    for tf in TF_HOURS:
        panels[tf] = resample_panels(P1, tf)
        resids[tf] = resid_returns(panels[tf]["C"], BETA_W[tf])

    cfgs = build_configs(a.grid, F is not None)
    print(f"[*] {len(cfgs)} konfigurasi ...")
    rows = []
    for k, (idea, tf, p) in enumerate(cfgs, 1):
        P, res, th = panels[tf], resids[tf], TF_HOURS[tf]
        if idea == "A_xsec":
            tr = idea_A(P, res, p["L"], p["H"], p["q"], p["mode"], th)
        elif idea == "C_zres":
            tr = idea_C(P, res, p["zw"], p["thr"], p["H"], p["mode"], th)
        elif idea == "D_break":
            tr = idea_D(P, p["N"], p["H"], p["mode"], th)
        else:
            tr = idea_E(P, F, p["thr"], p["H"], p["mode"])
        d = seg_metrics(tr, start, split, a.fee)
        v = seg_metrics(tr, split, end, a.fee)
        row = {"idea": idea, "tf": tf, "params": " ".join(f"{k_}={v_}" for k_, v_ in p.items())}
        row.update({f"d_{k_}": v_ for k_, v_ in d.items()})
        row.update({f"v_{k_}": v_ for k_, v_ in v.items()})
        rows.append(row)
        if k % 50 == 0:
            print(f"  {k}/{len(cfgs)}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "screen_results.csv", index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 40)
    fmt = lambda v: f"{v:,.2f}"
    show = ["idea", "tf", "params", "d_trades/yr", "d_gross_%", "d_net_%", "d_ann_%", "d_sharpe", "d_t",
            "v_trades/yr", "v_net_%", "v_ann_%", "v_sharpe", "v_t"]

    N = len(res)
    print("\n" + "=" * 100)
    print(f"{N} konfigurasi diuji. Di bawah hipotesis NOL (tanpa edge), t-stat TERBAIK dari {N} ide")
    print(f"diharapkan sekitar {np.sqrt(2 * np.log(N)):.1f}. Jadi t_discovery di bawah itu = kemungkinan kebetulan.")
    print("=" * 100)

    cand = res[(res["d_trades/yr"] >= a.min_trades_yr) & (res["d_net_%"] > 0)].copy()
    cand = cand.sort_values("d_sharpe", ascending=False)
    print(f"\nSHORTLIST dari DISCOVERY saja (volume >= {a.min_trades_yr:.0f} trade/thn, net>0), urut Sharpe discovery:")
    print(cand.head(a.top)[show].to_string(index=False, float_format=fmt))

    surv = cand[(cand["v_net_%"] > 0) & (cand["v_t"] >= 1.5) & (cand["d_t"] >= 2.0)]
    print(f"\nSURVIVOR (discovery t>=2, validation net>0 dan t>=1.5): {len(surv)} dari {len(cand)} shortlist")
    if len(surv):
        print(surv.sort_values("v_sharpe", ascending=False).head(a.top)[show].to_string(index=False, float_format=fmt))
    else:
        print("  Tidak ada. Artinya tidak ada ide di grid ini yang edge-nya bertahan di data validation.")

    print("\nRINGKASAN per keluarga ide (median atas konfigurasi; discovery | validation):")
    g = res.groupby("idea")[["d_gross_%", "d_net_%", "d_ann_%", "v_net_%", "v_ann_%"]].median()
    print(g.to_string(float_format=fmt))

    print("\nCara baca: pilih yang punya net_% dan ann_% BESAR dengan trades/yr BESAR di discovery DAN")
    print("validation, bukan salah satu saja. gross_% harus >= ~3x fee. Survivor tetap belum terbukti:")
    print("itu hanya kandidat untuk satu kali OOS dengan parameter terkunci.")
    print(f"[+] Semua hasil: {out_dir / 'screen_results.csv'}")


if __name__ == "__main__":
    main()