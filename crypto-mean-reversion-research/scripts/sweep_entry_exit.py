"""
Sweep dua uji (TRAIN saja), SL aktif dari bar 1 (min_hold_bars=0):

  Uji 1 - Entry tertunda: sinyal |z|>=z_threshold tetap pemicu, tapi entry baru
          dilakukan jika dalam <= W bar setelah sinyal ada KONFIRMASI reversal.
          Konfirmasi dicek di close bar d, entry di open[d+1] (tanpa lookahead).
          Tidak ada konfirmasi dalam W bar -> sinyal dibatalkan.
          confirm: z_turn   = z bergerak ke arah 0 dibanding bar sebelumnya
                   close_dir= candle bar d menutup searah posisi (close vs open)
                   both     = keduanya
  Uji 2 - SL/TP: sweep sl_mult x tp_mult (ATR di bar keputusan).

Baseline pembanding (harus cocok dengan CSV train kamu):
  immediate, sl=0.5, tp=2.5, min_hold=3  -> n_trades & avg return harus sama
  immediate, sl=0.5, tp=2.5, min_hold=0  -> dampak menghapus kebal 3 bar

Engine mereplikasi generate_train_trades_locked.py: 1 posisi per pair, sinyal
baru diabaikan sampai posisi sebelumnya exit, tie-break SL menang, fee round trip.

PERINGATAN overfit: grid ini cukup banyak. Pilih konfigurasi yang stabil
(banyak tahun positif, sedikit sensitif ke tetangga grid), bukan cuma yang
avg return tertinggi. OOS hanya boleh disentuh SEKALI setelah kandidat dikunci.

Contoh:
  py sweep_entry_exit.py --pairs pairs_train_selected.txt --data-dir <futures_1h> ^
     --train-end 2023-12-31 --out-dir <diag>
"""

import argparse
import itertools
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
LOCKED = dict(z_threshold=3.8, hl_threshold=20.0, atr_threshold=0.85)
FAST_WINDOW = 300  # scan exit dulu di 300 bar; kalau belum kena, scan sisa data


# ───────────────────────── data ─────────────────────────
def load_pair(base, data_dir, a):
    p = data_dir / f"{base}USDT.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < 1000:
        return None

    z = calculate_zscore(df, window=a.z_window)
    atr_raw = calculate_atr(df, n=a.atr_n)
    atr_pct = calculate_atr_percentile(df, atr_n=a.atr_n, lookback=a.atr_lookback)
    adx = calculate_adx(df, n=a.adx_n)["adx"]
    hl = calculate_half_life(df, window=a.hl_window)["half_life"]

    sig = combine_single_pair_entry(
        z, atr_pct, adx, hl,
        z_threshold=LOCKED["z_threshold"], atr_threshold=LOCKED["atr_threshold"],
        adx_threshold=NO_ADX_THRESHOLD, hl_threshold=LOCKED["hl_threshold"])
    el, es = sig["entry_long"].to_numpy(), sig["entry_short"].to_numpy()

    return {
        "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
        "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
        "ts": pd.DatetimeIndex(df["timestamp"]),
        "z": z.to_numpy(float), "atr": atr_raw.to_numpy(float),
        "el": el, "es": es, "sig_idx": np.flatnonzero(el | es),
    }


# ───────────────────────── engine ─────────────────────────
def resolve_exit(direction, tp, sl, high, low, min_hold):
    """(bar_offset, price, type) atau None. SL aktif setelah min_hold; tie -> SL."""
    n = len(high)
    if n == 0:
        return None
    if direction == 1:
        tp_hit, sl_hit = high >= tp, low <= sl
    else:
        tp_hit, sl_hit = low <= tp, high >= sl
    if min_hold > 0:
        sl_hit = sl_hit.copy()
        sl_hit[:min(min_hold, n)] = False
    hit = tp_hit | sl_hit
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    return (i + 1, sl, "SL") if sl_hit[i] else (i + 1, tp, "TP")


def resolve_exit_live(direction, tp, sl, o, h, l, c, min_hold):
    """
    Model eksekusi mirip live (manage_open_position):
      - TP native aktif dari bar 1 (exit di harga tp).
      - min_hold == 0: SL native dipasang saat entry -> aktif dari bar 1.
      - min_hold > 0 : SL dipasang SETELAH bar ke-(min_hold+1) close (bars_since_entry
        > MIN_HOLD_BARS). Saat itu kalau close bar tsb sudah melewati level SL
        (error -2021 di live) -> close MARKET di harga close bar itu (type "SLGAP").
        Kalau belum, SL aktif mulai bar berikutnya.
      - SL yang kena: kalau open bar sudah melewati SL (gap) -> fill di open, bukan di sl.
    Return (bar_offset, price, type) atau None.
    """
    n = len(h)
    if n == 0:
        return None
    long_ = direction == 1
    tp_hit = (h >= tp) if long_ else (l <= tp)

    start = 0
    if min_hold > 0:
        m = min(min_hold, n - 1)
        pre = tp_hit[:m + 1]
        if pre.any():
            return int(np.argmax(pre)) + 1, tp, "TP"
        if n <= min_hold:
            return None
        beyond = (c[min_hold] <= sl) if long_ else (c[min_hold] >= sl)
        if beyond:
            return min_hold + 1, c[min_hold], "SLGAP"
        start = min_hold + 1
        if start >= n:
            return None

    hh, ll, oo, tph = h[start:], l[start:], o[start:], tp_hit[start:]
    sl_hit = (ll <= sl) if long_ else (hh >= sl)
    hit = tph | sl_hit
    if not hit.any():
        return None
    j = int(np.argmax(hit))
    if sl_hit[j]:
        gap = (oo[j] <= sl) if long_ else (oo[j] >= sl)
        return start + j + 1, (oo[j] if gap else sl), "SL"
    return start + j + 1, tp, "TP"


def find_decision_bar(d, t, direction, wait, confirm):
    """Bar keputusan (close-nya dipakai), entry di open[bar+1]. None = batal."""
    if wait == 0:
        return t
    z, o, c = d["z"], d["o"], d["c"]
    n = len(o)
    for k in range(1, wait + 1):
        b = t + k
        if b + 1 >= n:
            return None
        z_ok = (z[b] > z[b - 1]) if direction == 1 else (z[b] < z[b - 1])
        c_ok = (c[b] > o[b]) if direction == 1 else (c[b] < o[b])
        ok = {"z_turn": z_ok, "close_dir": c_ok, "both": z_ok and c_ok}[confirm]
        if ok:
            return b
    return None


def _resolve(model, direction, tp, sl, d, a, b, min_hold):
    if model == "ideal":
        return resolve_exit(direction, tp, sl, d["h"][a:b], d["l"][a:b], min_hold)
    return resolve_exit_live(direction, tp, sl, d["o"][a:b], d["h"][a:b],
                             d["l"][a:b], d["c"][a:b], min_hold)


def backtest(pairs_data, train_end, wait, confirm, sl_mult, tp_mult, min_hold, model="live"):
    recs, n_sig, n_fill = [], 0, 0
    for base, d in pairs_data.items():
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
            sl = entry - direction * sl_mult * atr

            res = _resolve(model, direction, tp, sl, d, ex, ex + FAST_WINDOW, min_hold)
            if res is None:
                res = _resolve(model, direction, tp, sl, d, ex, None, min_hold)
            if res is None:
                last_exit = n
                continue
            off, px, typ = res
            last_exit = ex + off - 1
            raw = direction * (px - entry) / entry
            recs.append((d["ts"][t], base, raw - FEE_ROUND_TRIP, typ))

    return pd.DataFrame(recs, columns=["entry_time", "pair", "ret", "type"]), n_sig, n_fill


def summarize(tr, n_sig, n_fill):
    if len(tr) == 0:
        return {"n_trades": 0}
    r = tr["ret"]
    gl = -r[r <= 0].sum()
    std = r.std(ddof=1)
    yr = tr.groupby(tr["entry_time"].dt.year)["ret"].mean()
    return {
        "n_trades": len(tr),
        "fill_%": n_fill / n_sig * 100 if n_sig else np.nan,
        "win_%": (r > 0).mean() * 100,
        "tp_%": (tr["type"] == "TP").mean() * 100,
        "gapexit_%": (tr["type"] == "SLGAP").mean() * 100,
        "avg_ret_%": r.mean() * 100,
        "median_%": r.median() * 100,
        "pf": r[r > 0].sum() / gl if gl > 0 else np.inf,
        "t_stat": r.mean() / (std / np.sqrt(len(r))) if std > 0 else np.nan,
        "years_pos": f"{int((yr > 0).sum())}/{len(yr)}",
        "worst_year_%": yr.min() * 100,
    }


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--waits", default="0,1,2,3", help="lama tunggu konfirmasi (bar); 0 = entry langsung")
    ap.add_argument("--confirms", default="z_turn,close_dir,both")
    ap.add_argument("--sl-grid", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--tp-grid", default="1.5,2.0,2.5,3.0")
    ap.add_argument("--exit-model", choices=["live", "ideal"], default="live",
                    help="live = SL gap/market-close seperti kode live (default); ideal = exit selalu di level SL")
    ap.add_argument("--min-trades", type=int, default=300, help="minimal trade agar config diranking")
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

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:,.3f}"

    # baseline (validasi vs CSV train)
    print("\n" + "=" * 78 + "\nBASELINE\n" + "=" * 78)
    base_rows = {}
    for name, mh, mdl in [("ideal_mh3 (harus = CSV train)", 3, "ideal"),
                          ("live_mh3  (gap SL realistis)", 3, "live"),
                          ("ideal_mh0 (tanpa kebal)", 0, "ideal"),
                          ("live_mh0  (SL dari bar 1)", 0, "live")]:
        tr, ns, nf = backtest(data, train_end, 0, "z_turn", 0.5, 2.5, mh, mdl)
        base_rows[name] = summarize(tr, ns, nf)
    print(pd.DataFrame(base_rows).T.to_string(float_format=fmt))

    waits = [int(x) for x in a.waits.split(",")]
    confirms = a.confirms.split(",")
    sls = [float(x) for x in a.sl_grid.split(",")]
    tps = [float(x) for x in a.tp_grid.split(",")]

    entry_modes = []
    for w in waits:
        if w == 0:
            entry_modes.append((0, "-"))
        else:
            entry_modes += [(w, c) for c in confirms]

    total = len(entry_modes) * len(sls) * len(tps)
    print(f"\n[*] sweep {total} konfigurasi (min_hold=0, exit-model={a.exit_model}) ...")
    rows, k = [], 0
    for (w, c), sl, tp in itertools.product(entry_modes, sls, tps):
        k += 1
        tr, ns, nf = backtest(data, train_end, w, c if w else "z_turn", sl, tp, 0, a.exit_model)
        s = summarize(tr, ns, nf)
        s.update({"wait": w, "confirm": c, "sl": sl, "tp": tp})
        rows.append(s)
        if k % 20 == 0:
            print(f"  {k}/{total}")

    res = pd.DataFrame(rows)
    cols = ["wait", "confirm", "sl", "tp", "n_trades", "fill_%", "win_%", "tp_%", "avg_ret_%",
            "median_%", "pf", "t_stat", "years_pos", "worst_year_%"]
    res = res[cols]
    res.to_csv(out_dir / "sweep_results.csv", index=False)

    ok = res[res["n_trades"] >= a.min_trades].sort_values("avg_ret_%", ascending=False)

    print("\n" + "=" * 78 + f"\nTOP {a.top} by avg_ret_% (n_trades >= {a.min_trades})\n" + "=" * 78)
    print(ok.head(a.top).to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78 + "\nUJI 2 SAJA: entry langsung (wait=0), grid SL x TP, avg_ret_%\n" + "=" * 78)
    w0 = res[res["wait"] == 0]
    if len(w0):
        print(w0.pivot(index="sl", columns="tp", values="avg_ret_%").to_string(float_format=fmt))
        print("\nn_trades:")
        print(w0.pivot(index="sl", columns="tp", values="n_trades").to_string(float_format=lambda v: f"{v:,.0f}"))

    print("\n" + "=" * 78 + "\nUJI 1: rata-rata per mode entry (sl/tp dirata-ratakan, avg_ret_%)\n" + "=" * 78)
    m = res.groupby(["wait", "confirm"])[["n_trades", "fill_%", "avg_ret_%", "pf"]].mean()
    print(m.to_string(float_format=fmt))

    print("\nCara baca: kandidat bagus = avg_ret_% > baseline_mh3 (kebal 3 bar), pf > 1.2,")
    print("years_pos mayoritas, n_trades tidak anjlok, dan tetangga grid (sl/tp sebelah)")
    print("juga positif. Satu sel tinggi di tengah sel jelek = noise, bukan edge.")
    print(f"[+] Semua hasil: {out_dir / 'sweep_results.csv'}")


if __name__ == "__main__":
    main()