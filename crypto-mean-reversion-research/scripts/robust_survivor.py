"""
Uji ketahanan survivor  C_zres 1h  (z-score RESIDUAL terhadap BTC, mode rev, exit by waktu)  -- TRAIN saja.

Strategi: z = z-score (window zw) dari akumulasi return residual (r_koin - beta*r_BTC).
  z <= -thr -> LONG, z >= +thr -> SHORT. Entry di open bar setelah sinyal, exit di open
  setelah H bar (time stop). Opsional catastrophic stop (% dari harga entry, gap-aware).

Bagian:
  1. DISTRIBUSI kerugian (tanpa SL): worst/p1/p5, share trade rugi >5/10/20%, trade terburuk.
  2. CATASTROPHIC stop lebar: berapa edge hilang untuk memotong ekor.
  3. STRES fee/slippage (0.09% .. 0.25%).
  4. PARAMETER TETANGGA (zw x thr x H) pada fee 0.12% dan fee stres.
  5. SIMULASI PORTOFOLIO mirip demo: batas slot, prioritas |z| terbesar, MTM per jam.
     Grid: H x catastrophic x jumlah slot. Untuk tiap sel dicari ukuran posisi f* (notional
     per posisi, fraksi equity awal) TERBESAR yang masih memenuhi:
        - maxDD (mark-to-market) <= --dd-target
        - total exposure (slot x f) <= --max-exposure   (1.0 = tanpa leverage, seperti demo 1x)
     lalu dilaporkan return tahunan pada f* ("grid paling efisien" = return tertinggi pada
     risiko yang sama). Sebagai pembanding ada baris REF dengan setting mirip demo
     (--ref-slots, --ref-f).

Catatan metodologi:
  - Return portofolio NON-compounding (% dari equity awal), jadi return/DD tidak bergantung
    skala; yang dioptimasi adalah slot/exit, sizing mengikuti batas DD & exposure.
  - MTM memakai harga close per jam; DD intrabar (low/high) tidak tertangkap.
  - Aturan non-overlap per pair sedikit berbeda dari screen_ideas.py, jadi jumlah trade bisa
    selisih beberapa persen dari tabel screening.
  - Semua di periode train. Jangan menyentuh OOS dari script ini.

Butuh di folder yang sama: screen_ideas.py.
Contoh:
  py robust_survivor.py --pairs <pairs_train_selected.txt> --data-dir <futures_1h> ^
     --train-end 2023-12-31 --split 2022-07-01 --out-dir <robust_survivor>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import screen_ideas as si

HOURS_PER_YEAR = si.HOURS_PER_YEAR
SIDE = "both"   # both | long | short (diatur dari --side)
ELIG = None     # panel bool (bar x pair): pair boleh diperdagangkan pada bar itu (universe point-in-time)


def make_elig(P, min_qvol=None, min_age=0, top_n=None, win_days=30):
    """Universe point-in-time: hanya pakai info SEBELUM hari berjalan (median volume 30 hari
    sebelumnya, usia listing). Tidak ada info performa dan tidak ada survivorship (pair delisting
    tetap ada selama masih memenuhi syarat)."""
    if min_qvol is None and not top_n and not min_age:
        return None
    index, cols = P["C"].index, P["C"].columns
    Q = P["Q"]
    if (min_qvol is not None or top_n) and Q.isna().all().all():
        raise SystemExit("Data tidak punya kolom quote_volume; filter volume butuh klines hasil download_binance_vision.py")
    qd = Q.resample("D").sum(min_count=1)
    med = qd.rolling(win_days, min_periods=max(5, win_days // 2)).median().shift(1)
    ok = pd.DataFrame(True, index=med.index, columns=med.columns)
    if min_qvol is not None:
        ok &= (med >= min_qvol)
    if top_n:
        ok &= (med.rank(axis=1, ascending=False) <= top_n)
    if min_age:
        first = P["C"].apply(lambda s_: s_.first_valid_index())
        day = med.index.values.astype("datetime64[D]")
        fv = pd.DatetimeIndex(first.values).values.astype("datetime64[D]")
        age = (day[:, None] - fv[None, :]).astype("timedelta64[D]").astype(float)
        ok &= pd.DataFrame(age >= min_age, index=med.index, columns=med.columns)
    ok = ok.reindex(index.normalize()).fillna(False).astype(bool)
    return ok.to_numpy()


# ───────────────────────── event builder ─────────────────────────
def build_events(P, res, zw, thr, H, cat, fee):
    """Semua sinyal mentah (tanpa batas slot/overlap) beserta hasil exit-nya."""
    On, Hn, Ln = P["O"].to_numpy(), P["H"].to_numpy(), P["L"].to_numpy()
    n = On.shape[0]
    X = res.cumsum()
    z = (X - X.rolling(zw, min_periods=zw).mean()) / X.rolling(zw, min_periods=zw).std(ddof=0)
    zn = z.to_numpy().copy()
    btc = list(P["C"].columns).index("BTC")
    zn[:, btc] = np.nan

    with np.errstate(invalid="ignore"):
        rl, cl = np.nonzero(zn <= -thr)
        rs, cs = np.nonzero(zn >= thr)
    t = np.concatenate([rl, rs])
    col = np.concatenate([cl, cs])
    d = np.concatenate([np.ones(len(rl)), -np.ones(len(rs))])
    if SIDE == "long":
        keep = d > 0
        t, col, d = t[keep], col[keep], d[keep]
    elif SIDE == "short":
        keep = d < 0
        t, col, d = t[keep], col[keep], d[keep]
    absz = np.abs(zn[t, col])
    ei = t + 1
    ok = ei + H < n
    t, col, d, absz, ei = t[ok], col[ok], d[ok], absz[ok], ei[ok]
    if ELIG is not None:
        keep = ELIG[t, col]
        t, col, d, absz, ei = t[keep], col[keep], d[keep], absz[keep], ei[keep]

    entry = On[ei, col]
    xpx = On[ei + H, col]
    ok = np.isfinite(entry) & np.isfinite(xpx) & (entry > 0)
    t, col, d, absz, ei, entry, xpx = t[ok], col[ok], d[ok], absz[ok], ei[ok], entry[ok], xpx[ok]

    if cat is None or len(t) == 0:
        anyhit = np.zeros(len(t), bool)
        first = np.zeros(len(t), int)
        px = xpx
    else:
        k = np.arange(H)
        rows = ei[:, None] + k[None, :]
        cc = col[:, None]
        Lm, Hm, Om = Ln[rows, cc], Hn[rows, cc], On[rows, cc]
        level = np.where(d > 0, entry * (1 - cat), entry * (1 + cat))
        with np.errstate(invalid="ignore"):
            hit = np.where(d[:, None] > 0, Lm <= level[:, None], Hm >= level[:, None])
        anyhit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        ob = Om[np.arange(len(ei)), first]
        stop_px = np.where(d > 0, np.minimum(level, ob), np.maximum(level, ob))
        px = np.where(anyhit & np.isfinite(stop_px), stop_px, xpx)

    sb = ei + first
    xb = np.where(anyhit, sb, ei + H)            # bar tempat posisi dianggap terealisasi
    free = np.where(anyhit, sb + 1, ei + H)      # bar paling awal slot/pair boleh dipakai lagi
    gross = d * (px / entry - 1)
    order = np.lexsort((-absz, t))
    ev = {"t": t, "col": col, "dir": d, "absz": absz, "ei": ei, "entry": entry, "px": px,
          "xb": xb, "free": free, "gross": gross, "stopped": anyhit}
    ev = {k_: v[order] for k_, v in ev.items()}
    ev["net"] = ev["gross"] - fee
    ev["H"] = H
    return ev


def nonoverlap(ev):
    """Satu posisi per pair, tanpa batas slot."""
    last, acc = {}, []
    for i in range(len(ev["t"])):
        c = ev["col"][i]
        if ev["ei"][i] >= last.get(c, -1):
            last[c] = ev["free"][i]
            acc.append(i)
    return np.array(acc, dtype=int)


def to_frame(ev, acc, index, cols):
    return pd.DataFrame({
        "entry_time": index[ev["t"][acc]], "pair": np.array(cols)[ev["col"][acc]],
        "dir": ev["dir"][acc], "gross": ev["gross"][acc],
        "hold_h": (ev["free"][acc] - ev["ei"][acc]).astype(float),
        "stopped": ev["stopped"][acc],
    })


# ───────────────────────── statistik trade ─────────────────────────
def pf(r):
    r = np.asarray(r)
    gl = -r[r <= 0].sum()
    return r[r > 0].sum() / gl if gl > 0 else np.inf


def dist_row(net):
    net = np.asarray(net)
    if len(net) == 0:
        return {}
    srt = np.sort(net)
    k = max(1, int(len(net) * 0.01))
    return {
        "n": len(net), "mean_%": net.mean() * 100, "median_%": np.median(net) * 100,
        "worst_%": srt[0] * 100, "p1_%": np.percentile(net, 1) * 100, "p5_%": np.percentile(net, 5) * 100,
        "p95_%": np.percentile(net, 95) * 100,
        "<-5%": (net < -0.05).mean() * 100, "<-10%": (net < -0.10).mean() * 100,
        "<-20%": (net < -0.20).mean() * 100,
        "mean_ex_worst1%": srt[k:].mean() * 100, "pf": pf(net),
    }


# ───────────────────────── portofolio ─────────────────────────
def simulate_slots(ev, slots):
    held, acc = {}, []
    ei, col, free = ev["ei"], ev["col"], ev["free"]
    for i in range(len(ei)):
        e = ei[i]
        if held:
            for c_ in [c for c, f in held.items() if f <= e]:
                del held[c_]
        c = col[i]
        if c in held or len(held) >= slots:
            continue
        held[c] = free[i]
        acc.append(i)
    return np.array(acc, dtype=int)


def unit_curve(ev, acc, Cn, n_bars, net):
    """Kurva pnl (fraksi) untuk notional 1.0 per posisi, MTM per close jam."""
    realized = np.zeros(n_bars + 1)
    unreal = np.zeros(n_bars)
    conc = np.zeros(n_bars + 1)
    for i in acc:
        e, xb, c, d_ = ev["ei"][i], ev["xb"][i], ev["col"][i], ev["dir"][i]
        realized[xb] += net[i]
        if xb > e:
            seg = Cn[e:xb, c] / ev["entry"][i] - 1
            unreal[e:xb] += d_ * np.nan_to_num(seg, nan=0.0)
        conc[e] += 1
        conc[xb] -= 1
    curve = np.cumsum(realized[:n_bars]) + unreal
    return curve, np.cumsum(conc[:n_bars])


def max_dd(curve, f):
    eq = 1 + f * curve
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def solve_f(curve, f_cap, dd_target):
    """f terbesar <= f_cap dengan |maxDD| <= dd_target."""
    if -max_dd(curve, f_cap) <= dd_target:
        return f_cap, "exposure"
    lo, hi = 0.0, f_cap
    for _ in range(40):
        mid = (lo + hi) / 2
        if -max_dd(curve, mid) <= dd_target:
            lo = mid
        else:
            hi = mid
    return lo, "DD"


def portfolio_row(curve, conc, f, index, idx_split, n_acc, n_ev, years, years_v, label):
    eq = 1 + f * curve
    total = (eq[-1] - 1) * 100
    dd = max_dd(curve, f) * 100
    daily = pd.Series(eq, index=index).groupby(index.floor("D")).last()
    r = daily.pct_change().dropna()
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(365) if r.std(ddof=1) > 0 else np.nan
    v = 1 + f * (curve[idx_split:] - curve[idx_split - 1])
    v_ret = (v[-1] - 1) * 100
    v_dd = float((v / np.maximum.accumulate(v) - 1).min()) * 100
    out = dict(label)
    out.update({
        "f": f, "n": n_acc, "skip_%": max(0.0, (1 - n_acc / max(n_ev, 1)) * 100),
        "avg_conc": conc.mean(), "peak_exp_%": conc.max() * f * 100,
        "ann_%": total / years, "maxDD_%": dd, "ret/DD": total / abs(dd) if dd < 0 else np.nan,
        "sharpe": sharpe, "v_ann_%": v_ret / years_v, "v_maxDD_%": v_dd,
        "v_ret/DD": v_ret / abs(v_dd) if v_dd < 0 else np.nan,
    })
    return out


# ───────────────────────── main ─────────────────────────
def parse_cats(s):
    return [None if x.strip().lower() == "none" else float(x) for x in s.split(",")]


def cat_name(c):
    return "none" if c is None else f"{c*100:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--split", default="2022-07-01")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--zw", type=int, default=100)
    ap.add_argument("--thr", type=float, default=3.5)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--fee", type=float, default=0.0012)
    ap.add_argument("--cats", default="none,0.20,0.12,0.08,0.05", help="catastrophic stop (fraksi harga) untuk bagian 2")
    ap.add_argument("--fees", default="0.0009,0.0012,0.0015,0.0020,0.0025")
    ap.add_argument("--h-list", default="3,6", help="H yang diuji di grid portofolio")
    ap.add_argument("--pf-cats", default="none,0.20,0.12", help="catastrophic yang diuji di grid portofolio")
    ap.add_argument("--slots", default="2,3,5,8,10,15,20")
    ap.add_argument("--dd-target", type=float, default=0.25)
    ap.add_argument("--max-exposure", type=float, default=1.0, help="total notional/equity maksimum (1.0 = tanpa leverage)")
    ap.add_argument("--ref-slots", type=int, default=2)
    ap.add_argument("--ref-f", type=float, default=0.25, help="notional per posisi ala demo (margin 25%% x 1x)")
    ap.add_argument("--skip-neighbors", action="store_true")
    ap.add_argument("--side", choices=["both", "long", "short"], default="both",
                    help="long = hanya sinyal LONG (z <= -thr), short = hanya SHORT")
    ap.add_argument("--exclude-file", default=None, help="txt berisi base yang DIKELUARKAN (mis. pairs_train_selected.txt)")
    ap.add_argument("--min-qvol", type=float, default=None, help="median volume harian USDT 30 hari sebelumnya >= nilai ini (point-in-time)")
    ap.add_argument("--min-age-days", type=int, default=0, help="usia listing minimal (hari) pada saat sinyal")
    ap.add_argument("--top-n", type=int, default=None, help="hanya N pair dengan volume terbesar pada hari itu")
    ap.add_argument("--selected-file", default=None, help="pairs_train_selected.txt: untuk membandingkan pair terpilih vs lainnya")
    ap.add_argument("--stress-slots", type=int, default=None, help="stres fee pada slot ini (bukan pilihan return tertinggi)")
    ap.add_argument("--stress-h", type=int, default=None)
    ap.add_argument("--stress-cat", default="none")
    ap.add_argument("--nb-zw", default="50,75,100,150,200", help="grid window untuk bagian 4")
    ap.add_argument("--nb-thr", default="3.0,3.5,4.0", help="grid threshold untuk bagian 4")
    ap.add_argument("--nb-h", default="2,3,4,6,8", help="grid hold untuk bagian 4")
    a = ap.parse_args()

    global SIDE, ELIG
    SIDE = a.side
    train_end, split, start = pd.Timestamp(a.train_end), pd.Timestamp(a.split), pd.Timestamp(a.start)
    end = train_end + pd.Timedelta(days=1)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.pairs, "r", encoding="utf-8") as f_:
        pairs = [x.strip() for x in f_ if x.strip() and not x.strip().startswith("#")]
    pairs = list(dict.fromkeys(["BTC"] + pairs))
    if a.exclude_file:
        with open(a.exclude_file, "r", encoding="utf-8") as fh:
            excl = {x.strip() for x in fh if x.strip() and not x.strip().startswith("#")}
        n0 = len(pairs)
        pairs = [p_ for p_ in pairs if p_ == "BTC" or p_ not in excl]
        print(f"[*] --exclude-file: {n0 - len(pairs)} pair dikeluarkan dari universe (hanya pair yang TIDAK pernah dipilih yang diperdagangkan)")

    print(f"[*] load {len(pairs)} pair ...")
    P = si.load_panels(pairs, Path(a.data_dir), train_end)
    res = si.resid_returns(P["C"], si.BETA_W["1h"])
    index, cols = P["C"].index, list(P["C"].columns)
    Cn = P["C"].to_numpy()
    ELIG = make_elig(P, a.min_qvol, a.min_age_days, a.top_n)
    if ELIG is not None:
        print(f"[*] universe point-in-time: rata-rata {ELIG.sum(axis=1).mean():.1f} pair eligible per jam "
              f"(min_qvol={a.min_qvol}, min_age={a.min_age_days}, top_n={a.top_n})")
    n_bars = len(index)
    idx_split = int(index.searchsorted(split))
    years = (index[-1] - index[0]) / pd.Timedelta(days=365.25)
    years_v = (index[-1] - split) / pd.Timedelta(days=365.25)
    print(f"[*] {len(cols)} pair x {n_bars:,} bar | survivor: zw={a.zw} thr={a.thr} H={a.H} | fee={a.fee*100:.2f}%")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    fmt = lambda v: f"{v:,.3f}"

    # ───── 1. distribusi (tanpa SL) ─────
    ev0 = build_events(P, res, a.zw, a.thr, a.H, None, a.fee)
    acc0 = nonoverlap(ev0)
    tf0 = to_frame(ev0, acc0, index, cols)
    net0 = ev0["net"][acc0]
    print("\n" + "=" * 100 + f"\n1. DISTRIBUSI HASIL PER TRADE (tanpa SL/TP, hold {a.H} jam, fee {a.fee*100:.2f}%)\n" + "=" * 100)
    masks = {"SEMUA": np.ones(len(tf0), bool),
             "discovery": (tf0["entry_time"] < split).to_numpy(),
             "validation": (tf0["entry_time"] >= split).to_numpy(),
             "long": (tf0["dir"] > 0).to_numpy(), "short": (tf0["dir"] < 0).to_numpy()}
    dist = pd.DataFrame({k: dist_row(net0[m]) for k, m in masks.items()}).T
    print(dist.to_string(float_format=fmt))
    if a.selected_file:
        with open(a.selected_file, "r", encoding="utf-8") as fh:
            sel = {x.strip() for x in fh if x.strip() and not x.strip().startswith("#")}
        tf0["grp"] = np.where(tf0["pair"].isin(sel), "terpilih (pairs_train_selected)", "lainnya (tidak pernah dipilih)")
        tf0["net"] = net0
        print("\nPER GRUP UNIVERSE (apakah edge ada juga di pair yang TIDAK pernah dipilih?):")
        rows = {}
        for g, d_ in tf0.groupby("grp"):
            dd_, vv_ = d_[d_["entry_time"] < split], d_[d_["entry_time"] >= split]
            rows[g] = {"pair": d_["pair"].nunique(), "n": len(d_), "net_%": d_["net"].mean() * 100,
                       "disc_net_%": dd_["net"].mean() * 100, "val_net_%": vv_["net"].mean() * 100,
                       "pf": pf(d_["net"]), "worst_%": d_["net"].min() * 100}
        print(pd.DataFrame(rows).T.to_string(float_format=fmt))
    worst = tf0.assign(net_pct=net0 * 100).nsmallest(10, "net_pct")[["entry_time", "pair", "dir", "net_pct"]]
    print("\n10 trade terburuk:")
    print(worst.to_string(index=False, float_format=fmt))

    # ───── 2. catastrophic stop ─────
    print("\n" + "=" * 100 + "\n2. CATASTROPHIC STOP LEBAR: berapa edge hilang untuk memotong ekor\n" + "=" * 100)
    rows = []
    for c in parse_cats(a.cats):
        ev = build_events(P, res, a.zw, a.thr, a.H, c, a.fee)
        acc = nonoverlap(ev)
        tfm = to_frame(ev, acc, index, cols)
        net = ev["net"][acc]
        d_ = si.seg_metrics(tfm, start, split, a.fee)
        v_ = si.seg_metrics(tfm, split, end, a.fee)
        r = dist_row(net)
        rows.append({"cat": cat_name(c), "n": len(net), "stopped_%": tfm["stopped"].mean() * 100,
                     "disc_net_%": d_["net_%"], "val_net_%": v_["net_%"], "val_t": v_["t"],
                     "mean_%": r["mean_%"], "worst_%": r["worst_%"], "p1_%": r["p1_%"], "pf": r["pf"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))

    # ───── 3. stres fee ─────
    print("\n" + "=" * 100 + "\n3. STRES FEE + SLIPPAGE (round trip per trade)\n" + "=" * 100)
    rows = []
    for c in (None, 0.12):
        ev = build_events(P, res, a.zw, a.thr, a.H, c, a.fee)
        tfm = to_frame(ev, nonoverlap(ev), index, cols)
        for fee in [float(x) for x in a.fees.split(",")]:
            d_ = si.seg_metrics(tfm, start, split, fee)
            v_ = si.seg_metrics(tfm, split, end, fee)
            rows.append({"cat": cat_name(c), "fee_%": fee * 100, "disc_net_%": d_["net_%"], "disc_t": d_["t"],
                         "val_net_%": v_["net_%"], "val_t": v_["t"], "val_ann_%": v_["ann_%"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))
    print("Fee break-even: kolom val_net_% yang menyentuh 0 menunjukkan batas fee+slippage yang masih tertahan.")

    # ───── 4. tetangga parameter ─────
    if not a.skip_neighbors:
        print("\n" + "=" * 100 + "\n4. PARAMETER TETANGGA (mode rev, hold waktu, tanpa SL), seluruhnya dari screen_ideas.idea_C\n" + "=" * 100)
        zws = [int(x) for x in a.nb_zw.split(",")]
        thrs = [float(x) for x in a.nb_thr.split(",")]
        Hs = [int(x) for x in a.nb_h.split(",")]
        rows = []
        for zw in zws:
            for thr in thrs:
                for H in Hs:
                    if SIDE == "both" and ELIG is None:
                        tr = si.idea_C(P, res, zw, thr, H, "rev", 1)
                    else:
                        evn = build_events(P, res, zw, thr, H, None, a.fee)
                        tr = to_frame(evn, nonoverlap(evn), index, cols)
                    r = {"zw": zw, "thr": thr, "H": H}
                    for tag, fee in (("12", 0.0012), ("20", 0.0020)):
                        d_ = si.seg_metrics(tr, start, split, fee)
                        v_ = si.seg_metrics(tr, split, end, fee)
                        r[f"d_net{tag}"], r[f"v_net{tag}"] = d_["net_%"], v_["net_%"]
                        r[f"v_t{tag}"] = v_["t"]
                    r["n_per_yr"] = si.seg_metrics(tr, start, end, 0.0012)["trades/yr"]
                    rows.append(r)
        nb = pd.DataFrame(rows)
        nb.to_csv(out_dir / "neighbors.csv", index=False)
        for tag in ("12", "20"):
            both = ((nb[f"d_net{tag}"] > 0) & (nb[f"v_net{tag}"] > 0)).mean() * 100
            print(f"fee 0.{tag}%: {both:.0f}% dari {len(nb)} kombinasi positif di discovery DAN validation "
                  f"| median val net = {nb[f'v_net{tag}'].median():.3f}%")
        print("\nval_net_% (fee 0.12%) : zw x thr  (H=6)")
        print(nb[nb["H"] == 6].pivot(index="zw", columns="thr", values="v_net12").to_string(float_format=fmt))
        print("\nval_net_% (fee 0.12%) : H x thr  (zw=100)")
        print(nb[nb["zw"] == 100].pivot(index="H", columns="thr", values="v_net12").to_string(float_format=fmt))
        print("\nval_net_% (fee 0.20%) : zw x thr  (H=6)")
        print(nb[nb["H"] == 6].pivot(index="zw", columns="thr", values="v_net20").to_string(float_format=fmt))

    # ───── 5. portofolio ─────
    print("\n" + "=" * 100)
    print(f"5. PORTOFOLIO: slot x catastrophic x H  | DD target {a.dd_target*100:.0f}% | exposure maks {a.max_exposure*100:.0f}% "
          f"| prioritas |z| terbesar")
    print("=" * 100)
    slots_list = [int(x) for x in a.slots.split(",")]
    rows, ref_rows = [], []
    for H in [int(x) for x in a.h_list.split(",")]:
        for c in parse_cats(a.pf_cats):
            ev = build_events(P, res, a.zw, a.thr, H, c, a.fee)
            n_ev = len(nonoverlap(ev))   # peluang entry tanpa batas slot (basis skip_%)
            for S in slots_list:
                acc = simulate_slots(ev, S)
                curve, conc = unit_curve(ev, acc, Cn, n_bars, ev["net"])
                f_cap = a.max_exposure / S
                f, bind = solve_f(curve, f_cap, a.dd_target)
                lab = {"H": H, "cat": cat_name(c), "slots": S, "bind": bind}
                rows.append(portfolio_row(curve, conc, f, index, idx_split, len(acc), n_ev, years, years_v, lab))
                if S == a.ref_slots:
                    lab2 = {"H": H, "cat": cat_name(c), "slots": S, "bind": "REF(demo)"}
                    ref_rows.append(portfolio_row(curve, conc, a.ref_f, index, idx_split, len(acc), n_ev, years, years_v, lab2))
    pf_df = pd.DataFrame(rows)
    pf_df.to_csv(out_dir / "portfolio_grid.csv", index=False)
    cols_show = ["H", "cat", "slots", "bind", "f", "n", "skip_%", "avg_conc", "peak_exp_%", "ann_%", "maxDD_%",
                 "ret/DD", "sharpe", "v_ann_%", "v_maxDD_%", "v_ret/DD"]
    ranked = pf_df.sort_values("ann_%", ascending=False)
    print("\nTOP 15 (return tahunan tertinggi pada DD target; f = notional per posisi, fraksi equity):")
    print(ranked.head(15)[cols_show].to_string(index=False, float_format=fmt))
    print("\nREFERENSI mirip demo (slot & f tetap, tanpa optimasi):")
    if ref_rows:
        print(pd.DataFrame(ref_rows)[cols_show].to_string(index=False, float_format=fmt))

    best = ranked.iloc[0]
    print(f"\nPILIHAN GRID: H={int(best['H'])} cat={best['cat']} slots={int(best['slots'])} f={best['f']:.3f} "
          f"(binding: {best['bind']}) -> ann {best['ann_%']:.1f}%/thn, maxDD {best['maxDD_%']:.1f}%, "
          f"validation: ann {best['v_ann_%']:.1f}%/thn, maxDD {best['v_maxDD_%']:.1f}%")

    # stres fee pada pilihan grid
    print("\nStres fee pada pilihan grid (f tetap):")
    if a.stress_slots:
        sH = a.stress_h or int(best["H"])
        sC = parse_cats(a.stress_cat)[0]
        sS = a.stress_slots
        ev = build_events(P, res, a.zw, a.thr, sH, sC, a.fee)
        acc = simulate_slots(ev, sS)
        c_, _ = unit_curve(ev, acc, Cn, n_bars, ev["net"])
        sf, _b = solve_f(c_, a.max_exposure / sS, a.dd_target)
        print(f"(konfigurasi stres: H={sH} cat={cat_name(sC)} slots={sS} f={sf:.3f})")
    else:
        sC = None if best["cat"] == "none" else float(best["cat"].strip("%")) / 100
        sH, sS, sf = int(best["H"]), int(best["slots"]), float(best["f"])
        ev = build_events(P, res, a.zw, a.thr, sH, sC, a.fee)
        acc = simulate_slots(ev, sS)
    rows = []
    for fee in [float(x) for x in a.fees.split(",")]:
        net = ev["gross"] - fee
        curve, conc = unit_curve(ev, acc, Cn, n_bars, net)
        r = portfolio_row(curve, conc, sf, index, idx_split, len(acc), len(nonoverlap(ev)), years, years_v,
                          {"fee_%": fee * 100})
        rows.append(r)
    print(pd.DataFrame(rows)[["fee_%", "ann_%", "maxDD_%", "ret/DD", "sharpe", "v_ann_%", "v_maxDD_%", "v_ret/DD"]]
          .to_string(index=False, float_format=fmt))

    print("\nCara baca / kriteria lulus (tulis SEBELUM OOS):")
    print("  - net validation > 0 pada fee 0.20% (bagian 3) dan >=60% tetangga positif (bagian 4)")
    print("  - worst trade/p1 masih ditoleransi atau terpotong catastrophic yang tidak merusak edge (bagian 1-2)")
    print("  - portofolio: ret/DD > 1 di full DAN validation, ann_% validation > 0 pada fee 0.20%")
    print("  - jangan pilih sel grid hanya karena ann_% tertinggi: cari plateau (slot/cat sebelahnya mirip)")
    print(f"[+] Hasil: {out_dir}")


if __name__ == "__main__":
    main()