"""
Uji PERSISTENSI kualitas pair, walk-forward, TRAIN saja:
"apakah pair yang bagus di masa lalu masih bagus di periode berikutnya?"
Kalau tidak, seleksi pair berbasis performa (seperti pairs_train_selected.txt) hanya fitting noise.

Masalah statistik yang dipecahkan: sinyal survivor (thr 4.0, long-only) hanya menghasilkan sekitar
beberapa trade per pair per tahun, jadi PnL per pair terlalu berisik untuk menilai pair. Karena itu
KUALITAS PAIR diukur dengan sinyal yang sama tapi lebih padat (--dense-thr, default 2.5):
  Q(pair, window) = rata-rata net return per trade dari sinyal long z-residual <= -dense_thr, hold H bar.
Lalu diuji dua hal:
  A. PERSISTENSI: korelasi peringkat (Spearman IC) antara Q di window lookback (sebelum periode)
     dan Q di periode berikutnya, per periode, dengan t-stat antar periode.
  B. NILAI SELEKSI: pair tercile atas (berdasarkan Q lookback) vs tercile bawah, diukur pada trade
     survivor (thr --sparse-thr) di periode berikutnya.
  C. FITUR STRUKTURAL tanpa PnL (autokorelasi lag-1 return residual, variance ratio, volatilitas,
     volume): apakah salah satunya memprediksi Q periode berikutnya (IC).
Semua keputusan pair pada tiap periode hanya memakai data SEBELUM periode itu (point-in-time).

Cara baca:
  - mean IC > ~0.10 dengan t > 2 antar periode  -> ada persistensi, seleksi bergulir layak dipakai
  - IC ~ 0 atau tidak stabil tandanya                -> "pair terbaik" itu noise; jangan seleksi pair
  - Fitur struktural dengan IC stabil                 -> kandidat aturan seleksi non-PnL

Butuh di folder yang sama: screen_ideas.py, robust_survivor.py.
Contoh:
  py pair_persistence.py --pairs <pairs_all_candidates.txt> --data-dir <klines_1h> ^
     --train-end 2023-12-31 --min-qvol 5000000 --min-age-days 180 --out-dir <persist>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import screen_ideas as si
import robust_survivor as rs


def trades_table(P, res, index, cols, zw, thr, H, fee):
    ev = rs.build_events(P, res, zw, thr, H, None, fee)
    acc = rs.nonoverlap(ev)
    return pd.DataFrame({"entry_time": index[ev["t"][acc]], "pair": np.array(cols)[ev["col"][acc]],
                         "net": ev["net"][acc]})


def structural_features(res, Q, start, lookback):
    w0 = start - lookback
    W = res[(res.index >= w0) & (res.index < start)]
    f = pd.DataFrame(index=res.columns)
    f["autocorr1"] = W.apply(lambda c: c.autocorr(1) if c.notna().sum() > 200 else np.nan)
    v1 = W.var()
    v3 = W.rolling(3).sum().var()
    f["vr3"] = v3 / (3 * v1)
    f["vol_daily"] = W.std() * np.sqrt(24)
    qd = Q[(Q.index >= w0) & (Q.index < start)].resample("D").sum(min_count=1)
    f["log_qvol"] = np.log(qd.median().replace(0, np.nan))
    return f


def spearman(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--start-eval", default="2021-01-01", help="awal periode evaluasi pertama")
    ap.add_argument("--period-months", type=int, default=6)
    ap.add_argument("--lookback-months", type=int, default=12)
    ap.add_argument("--dense-thr", type=float, default=2.5, help="threshold sinyal padat untuk mengukur kualitas pair")
    ap.add_argument("--sparse-thr", type=float, default=4.0, help="threshold sinyal survivor")
    ap.add_argument("--zw", type=int, default=100)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--side", choices=["both", "long", "short"], default="long")
    ap.add_argument("--fee", type=float, default=0.0012)
    ap.add_argument("--min-n", type=int, default=15, help="minimal trade padat per pair di window")
    ap.add_argument("--min-qvol", type=float, default=None)
    ap.add_argument("--min-age-days", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=None)
    a = ap.parse_args()

    rs.SIDE = a.side
    train_end = pd.Timestamp(a.train_end)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.pairs, "r", encoding="utf-8") as fh:
        pairs = [x.strip() for x in fh if x.strip() and not x.strip().startswith("#")]
    pairs = list(dict.fromkeys(["BTC"] + pairs))

    print(f"[*] load {len(pairs)} pair ...")
    P = si.load_panels(pairs, Path(a.data_dir), train_end)
    index, cols = P["C"].index, list(P["C"].columns)
    res = si.resid_returns(P["C"], si.BETA_W["1h"])
    rs.ELIG = rs.make_elig(P, a.min_qvol, a.min_age_days, a.top_n)
    if rs.ELIG is not None:
        print(f"[*] universe point-in-time: rata-rata {rs.ELIG.sum(axis=1).mean():.1f} pair eligible per jam")

    dense = trades_table(P, res, index, cols, a.zw, a.dense_thr, a.H, a.fee)
    sparse = trades_table(P, res, index, cols, a.zw, a.sparse_thr, a.H, a.fee)
    print(f"[*] trade padat (thr {a.dense_thr}): {len(dense):,} | trade survivor (thr {a.sparse_thr}): {len(sparse):,} "
          f"| side={a.side} | H={a.H} | fee={a.fee*100:.2f}%")
    per_pair_sparse = len(sparse) / max(sparse['pair'].nunique(), 1)
    print(f"    rata-rata trade survivor per pair selama train: {per_pair_sparse:.1f}"
          + (" (terlalu sedikit untuk menilai pair lewat PnL survivor langsung)" if per_pair_sparse < 30 else ""))

    L = pd.DateOffset(months=a.lookback_months)
    starts = list(pd.date_range(a.start_eval, train_end, freq=f"{a.period_months}MS"))
    rows, sel_rows, feat_rows = [], [], []
    for s in starts:
        e = min(s + pd.DateOffset(months=a.period_months), train_end + pd.Timedelta(days=1))
        lb = dense[(dense["entry_time"] >= s - L) & (dense["entry_time"] < s)].groupby("pair")["net"].agg(["count", "mean"])
        te = dense[(dense["entry_time"] >= s) & (dense["entry_time"] < e)].groupby("pair")["net"].agg(["count", "mean"])
        lb = lb[lb["count"] >= a.min_n]
        te = te[te["count"] >= a.min_n]
        common = lb.index.intersection(te.index)
        if len(common) < 12:
            print(f"  periode {s.date()}: pair memenuhi syarat terlalu sedikit ({len(common)}), dilewati")
            continue

        # A. persistensi
        ic = spearman(lb.loc[common, "mean"], te.loc[common, "mean"])
        rows.append({"period": f"{s.date()}..{e.date()}", "n_pair": len(common), "IC_Q": ic})

        # B. nilai seleksi (tercile) -> diukur pada trade survivor di periode berikutnya
        q = lb.loc[lb.index, "mean"].sort_values()
        k = len(q) // 3
        bottom, top = set(q.index[:k]), set(q.index[-k:])
        sp = sparse[(sparse["entry_time"] >= s) & (sparse["entry_time"] < e)]
        dn = dense[(dense["entry_time"] >= s) & (dense["entry_time"] < e)]
        r = {"period": f"{s.date()}..{e.date()}"}
        for tag, grp in (("top", top), ("bottom", bottom), ("all", set(q.index))):
            r[f"{tag}_sparse_n"] = int(sp["pair"].isin(grp).sum())
            r[f"{tag}_sparse_net_%"] = sp.loc[sp["pair"].isin(grp), "net"].mean() * 100
            r[f"{tag}_dense_net_%"] = dn.loc[dn["pair"].isin(grp), "net"].mean() * 100
        sel_rows.append(r)

        # C. fitur struktural
        F = structural_features(res, P["Q"], s, L)
        qte = te["mean"]
        for name in F.columns:
            feat_rows.append({"period": f"{s.date()}..{e.date()}", "feature": name,
                              "IC": spearman(F.loc[te.index.intersection(F.index), name], qte)})

    if not rows:
        raise SystemExit("Tidak ada periode yang bisa dievaluasi. Turunkan --min-n atau perbesar universe.")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:,.3f}"

    A = pd.DataFrame(rows)
    print("\n" + "=" * 90 + "\nA. PERSISTENSI: IC (Spearman) kualitas pair lookback vs periode berikutnya\n" + "=" * 90)
    print(A.to_string(index=False, float_format=fmt))
    print(f"\nmean IC = {A['IC_Q'].mean():.3f} | t antar periode = {tstat(A['IC_Q']):.2f} | "
          f"periode IC>0: {(A['IC_Q'] > 0).sum()}/{len(A)}")
    A.to_csv(out_dir / "persistence_ic.csv", index=False)

    B = pd.DataFrame(sel_rows)
    print("\n" + "=" * 90 + "\nB. NILAI SELEKSI: tercile atas vs bawah (Q lookback), diukur di periode berikutnya\n" + "=" * 90)
    print(B.to_string(index=False, float_format=fmt))
    for kind in ("sparse", "dense"):
        diff = B[f"top_{kind}_net_%"] - B[f"bottom_{kind}_net_%"]
        print(f"selisih top-bottom ({kind}, net %/trade): mean={diff.mean():.3f} | t antar periode={tstat(diff):.2f} "
              f"| top>bottom di {(diff > 0).sum()}/{diff.notna().sum()} periode")
    tot = {t: B[f"{t}_sparse_n"].sum() for t in ("top", "bottom", "all")}
    print(f"jumlah trade survivor terpool: top={tot['top']}, bottom={tot['bottom']}, all={tot['all']}")
    B.to_csv(out_dir / "selection_value.csv", index=False)

    C = pd.DataFrame(feat_rows)
    print("\n" + "=" * 90 + "\nC. FITUR STRUKTURAL (tanpa PnL): IC terhadap Q periode berikutnya\n" + "=" * 90)
    summ = C.groupby("feature")["IC"].agg(mean_IC="mean", t=tstat, pos=lambda x: f"{(x > 0).sum()}/{x.notna().sum()}")
    print(summ.to_string(float_format=fmt))
    C.to_csv(out_dir / "structural_ic.csv", index=False)

    print("\nKesimpulan otomatis:")
    mic, t_ic = A["IC_Q"].mean(), tstat(A["IC_Q"])
    if mic > 0.10 and t_ic > 2:
        print("  [ADA persistensi] kualitas pair lalu memprediksi periode berikutnya. Seleksi bergulir layak dipakai.")
    elif mic > 0.05:
        print("  [LEMAH] ada sedikit persistensi tapi belum meyakinkan. Jangan menyaring pair secara agresif.")
    else:
        print("  [TIDAK ada persistensi] performa pair lalu tidak memprediksi masa depan. Seleksi pair berbasis PnL = noise.")
    print("  Untuk fitur struktural: layak dipertimbangkan hanya jika mean IC > ~0.10 DAN t > 2 DAN tanda sama di mayoritas periode.")
    print(f"[+] Hasil: {out_dir}")


if __name__ == "__main__":
    main()