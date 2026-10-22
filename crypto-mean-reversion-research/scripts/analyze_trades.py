"""
Analisa trade-level CSV (kolom: entry_time, exit_time, direction, net_return, pair).
- PF / win rate keseluruhan & per pair
- Konsentrasi return (top N trade / top N bulan)
- Korelasi antar posisi yang overlap waktu (searah / hasil bareng)
- Asimetri long vs short
- Simulasi portfolio equity (opsional, kalau simulate_portfolio_directional.py
  ada di folder yang sama) -- TIERED sizing + LEVERAGE opsional

Jalankan:
    python analyze_trades.py --csv "D:\\...\\train_trades_locked_sl_delayed_only.csv"

Sweep (train, cari kandidat rasio):
    python analyze_trades.py --csv <path> --simulate --capital-pct 0.5 --max-positions 2

Validasi rasio terkunci (OOS, dengan/tanpa leverage):
    python analyze_trades.py --csv <path> --simulate --short-mult 1.5 --long-mult 0.5 --leverage 2.0
"""

import argparse
import numpy as np
import pandas as pd


def profit_factor(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return np.nan
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r <= 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else np.nan


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def overall_stats(df):
    print_header("1. RINGKASAN KESELURUHAN")
    n = len(df)
    wins = (df["net_return"] > 0).sum()
    print(f"Total trade       : {n}")
    print(f"Periode           : {df['entry_time'].min()}  ->  {df['entry_time'].max()}")
    print(f"Pair unik         : {df['pair'].nunique()}")
    print(f"Win rate          : {wins/n*100:.1f}%  ({wins}/{n})")
    print(f"Profit factor     : {profit_factor(df['net_return']):.3f}")
    print(f"Avg return/trade  : {df['net_return'].mean()*100:+.3f}%")
    print(f"Sum net_return    : {df['net_return'].sum():.3f}  (unweighted, bukan equity)")
    hold_h = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
    print(f"Hold time (jam)   : median={hold_h.median():.1f}  mean={hold_h.mean():.1f}  max={hold_h.max():.1f}")


def direction_stats(df):
    print_header("2. LONG vs SHORT")
    g = df.groupby("direction")["net_return"].agg(
        n="count",
        win_rate=lambda x: (x > 0).mean() * 100,
        pf=profit_factor,
        sum_return="sum",
    )
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    total = df["net_return"].sum()
    for d in g.index:
        pct = g.loc[d, "sum_return"] / total * 100 if total != 0 else np.nan
        print(f"  kontribusi {d:5s}: {pct:.1f}% dari total summed return")


def concentration_stats(df):
    print_header("3. KONSENTRASI RETURN")
    total = df["net_return"].sum()
    df_sorted = df.sort_values("net_return", ascending=False).reset_index(drop=True)
    for n in [5, 10, 20]:
        n = min(n, len(df_sorted))
        pct = df_sorted.iloc[:n]["net_return"].sum() / total * 100
        print(f"Top {n:>3d} trade ({n/len(df)*100:4.1f}% dari trade) = {pct:6.1f}% dari total summed return")
    top5pct_n = max(1, int(len(df) * 0.05))
    pct = df_sorted.iloc[:top5pct_n]["net_return"].sum() / total * 100
    print(f"Top 5% trade ({top5pct_n} trade)            = {pct:6.1f}% dari total summed return")

    monthly = df.set_index("entry_time").resample("ME")["net_return"].sum().sort_values(ascending=False)
    print("\nTop 5 bulan (summed net_return):")
    print(monthly.head(5).to_string(float_format=lambda x: f"{x:.3f}"))
    for k in [3, 5]:
        k = min(k, len(monthly))
        print(f"Top {k} bulan = {monthly.head(k).sum()/total*100:.1f}% dari total")


def overlap_correlation(df):
    print_header("4. OVERLAP / KORELASI ANTAR POSISI KONKUREN")
    d = df.sort_values("entry_time").reset_index(drop=True)
    n = len(d)
    same_sign = 0
    same_dir = 0
    total_overlap = 0
    rets1, rets2 = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if d.loc[j, "entry_time"] >= d.loc[i, "exit_time"]:
                break
            a0, a1 = d.loc[i, "entry_time"], d.loc[i, "exit_time"]
            b0, b1 = d.loc[j, "entry_time"], d.loc[j, "exit_time"]
            if a0 < b1 and b0 < a1:
                total_overlap += 1
                r1, r2 = d.loc[i, "net_return"], d.loc[j, "net_return"]
                if (r1 > 0) == (r2 > 0):
                    same_sign += 1
                if d.loc[i, "direction"] == d.loc[j, "direction"]:
                    same_dir += 1
                rets1.append(r1)
                rets2.append(r2)

    if total_overlap == 0:
        print("Tidak ada trade yang overlap waktu.")
        return
    corr = np.corrcoef(rets1, rets2)[0, 1] if total_overlap > 1 else np.nan
    print(f"Jumlah pasangan trade yang overlap waktu : {total_overlap}")
    print(f"  -> searah (long+long / short+short)     : {same_dir/total_overlap*100:.1f}%")
    print(f"  -> hasil sama (sama untung/sama rugi)   : {same_sign/total_overlap*100:.1f}%")
    print(f"  -> korelasi return antar pasangan overlap: {corr:.3f}")


def per_pair_stats(df, top_n=10):
    print_header("5. PER-PAIR")
    g = df.groupby("pair")["net_return"].agg(n="count", pf=profit_factor, sum_return="sum").sort_values(
        "sum_return", ascending=False
    )
    print(f"Top {top_n} pair (sum_return):")
    print(g.head(top_n).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nBottom {top_n} pair (sum_return):")
    print(g.tail(top_n).to_string(float_format=lambda x: f"{x:.3f}"))
    n_neg = (g["sum_return"] < 0).sum()
    print(f"\nJumlah pair dengan kontribusi negatif: {n_neg} / {len(g)}")


def max_drawdown_pct(equity_curve):
    """equity_curve: list of (timestamp, equity). Return max drawdown dalam %, negatif."""
    peak = -np.inf
    mdd = 0.0
    for _, v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            mdd = min(mdd, dd)
    return mdd * 100


# Rasio short_mult/long_mult yang di-sweep. Baseline (1.0/1.0) SELALU disertakan
# sebagai pembanding. Tambah/kurangi titik di sini kalau perlu.
SWEEP_RATIOS = [
    (1.0, 1.0),
    (1.1, 0.9),
    (1.2, 0.8),
    (1.3, 0.7),
    (1.5, 0.5),
    (2.0, 0.2),
]


def weighted_profit_factor(executed_trades):
    """PF yang di-weight sesuai allocated_capital tiap trade (bukan PF
    mentah trade-level) -- merefleksikan PF riil setelah sizing/tiering
    diterapkan, konsisten dengan equity curve yang dihasilkan simulasi."""
    if not executed_trades:
        return np.nan
    pnl = np.array([t["allocated_capital"] * t["net_return"] for t in executed_trades])
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl <= 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else np.nan


def run_simulation(df, capital_pct, max_positions, initial_equity, leverage, locked_ratio=None,
                    max_exposure_pct=None, fallback_leverage=1.0):
    print_header("6. SIMULASI PORTFOLIO EQUITY (simulate_portfolio_directional)")
    try:
        from simulate_portfolio_directional import simulate_portfolio
    except ImportError:
        print("[!] simulate_portfolio_directional.py tidak ditemukan di folder yang sama -- skip.")
        return

    trades = df.to_dict("records")
    rows = []

    if max_exposure_pct is not None:
        print(f"[*] max_total_exposure_pct DI-OVERRIDE manual ke {max_exposure_pct:.1f}% "
              f"(cap default <=100% TIDAK berlaku -- ini exposure/leverage riil, "
              f"pahami risikonya sebelum percaya hasilnya).")

    if locked_ratio is not None:
        print(f"[*] Mode validasi -- rasio dikunci: short_mult={locked_ratio[0]}, "
              f"long_mult={locked_ratio[1]}, leverage={leverage}, "
              f"fallback_leverage={fallback_leverage} "
              f"(dibandingkan ke baseline leverage=1.0 saja, TIDAK di-sweep)")
        ratios_to_run = [(1.0, 1.0, 1.0, 1.0), (locked_ratio[0], locked_ratio[1], leverage, fallback_leverage)]
    else:
        if leverage != 1.0 or fallback_leverage != 1.0:
            print(f"[*] Mode sweep dengan leverage={leverage}, fallback_leverage={fallback_leverage} "
                  f"diterapkan ke SEMUA rasio (termasuk baseline 1.0/1.0) -- kalau mau "
                  f"baseline murni, jalankan terpisah dengan --leverage 1.0 --fallback-leverage 1.0.")
        ratios_to_run = [(s, l, leverage, fallback_leverage) for s, l in SWEEP_RATIOS]

    for short_mult, long_mult, lev, fb_lev in ratios_to_run:
        result = simulate_portfolio(
            trades, capital_pct=capital_pct, max_positions=max_positions,
            initial_equity=initial_equity, short_mult=short_mult, long_mult=long_mult,
            leverage=lev, fallback_leverage=fb_lev, max_total_exposure_pct=max_exposure_pct,
        )
        eq = result["final_equity"]
        mdd = max_drawdown_pct(result["equity_curve"])
        ret_pct = (eq / initial_equity - 1) * 100
        n_exec = len(result["executed_trades"])
        n_skip_exp = len(result["skipped_exposure_trades"])
        return_dd_ratio = ret_pct / abs(mdd) if mdd != 0 else np.nan
        pf_weighted = weighted_profit_factor(result["executed_trades"])
        rows.append({
            "short_mult": short_mult, "long_mult": long_mult, "leverage": lev,
            "fallback_leverage": fb_lev,
            "final_equity": eq, "return_pct": ret_pct, "max_dd_pct": mdd,
            "return_over_dd": return_dd_ratio, "pf_weighted": pf_weighted,
            "executed": n_exec, "skipped_by_exposure": n_skip_exp,
        })

    sweep_df = pd.DataFrame(rows)
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if locked_ratio is not None:
        base_row = sweep_df.iloc[0]
        lock_row = sweep_df.iloc[1]
        print(f"\n[*] Baseline (leverage=1.0) return_over_dd = {base_row['return_over_dd']:.2f}, "
              f"max_dd = {base_row['max_dd_pct']:.2f}%")
        print(f"[*] Rasio dikunci (leverage={leverage}) return_over_dd = {lock_row['return_over_dd']:.2f}, "
              f"max_dd = {lock_row['max_dd_pct']:.2f}%")
        if lock_row["return_over_dd"] > base_row["return_over_dd"]:
            print("[*] Rasio yang dikunci MASIH lebih baik dari baseline di data ini -- "
                  "indikasi bagus (bukan bukti final, tapi konsisten dengan train).")
        else:
            print("[!] Rasio yang dikunci TIDAK lebih baik dari baseline di data ini -- "
                  "kemungkinan hasil train tadi overfit, pertimbangkan turunkan rasio/leverage "
                  "atau kembali ke 1.0/1.0.")
        if leverage != 1.0:
            dd_ratio = abs(lock_row['max_dd_pct']) / abs(base_row['max_dd_pct']) if base_row['max_dd_pct'] != 0 else np.nan
            print(f"[*] Max drawdown naik {dd_ratio:.2f}x dibanding baseline TANPA leverage -- "
                  "pastikan ini masih dalam batas toleransi risiko sebelum dipakai di demo/live.")
    else:
        print("\n[*] Baca ini dengan hati-hati:")
        print("    - Kalau return_over_dd naik TERUS/monoton sampai rasio paling ekstrem")
        print("      di list SWEEP_RATIOS, itu tanda overfitting -- BUKAN alasan untuk")
        print("      pilih rasio paling ekstrem.")
        print("    - Cari titik di mana return_over_dd mulai MENDATAR/plateau -- itu")
        print("      kandidat rasio yang lebih aman, bukan titik tertinggi di tabel.")
        print("    - Rasio final tetap wajib divalidasi di luar sample yang dipakai")
        print("      untuk memilihnya.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path ke trade-level CSV.")
    parser.add_argument("--simulate", action="store_true",
                         help="Jalankan simulasi portfolio (butuh simulate_portfolio_directional.py).")
    parser.add_argument("--capital-pct", type=float, default=0.5)
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--short-mult", type=float, default=None,
                         help="Kalau diisi bersama --long-mult: mode VALIDASI, cuma "
                              "bandingkan rasio ini vs baseline 1.0/1.0 (tidak sweep).")
    parser.add_argument("--long-mult", type=float, default=None)
    parser.add_argument("--leverage", type=float, default=1.0,
                         help="Pengali NOTIONAL per posisi (default 1.0 = tanpa leverage). "
                              "Di mode validasi, leverage diterapkan HANYA ke rasio yang "
                              "dikunci, baseline tetap leverage=1.0 untuk pembanding adil.")
    parser.add_argument("--fallback-leverage", type=float, default=1.0,
                         help="Leverage KHUSUS yang hanya berlaku saat SHORT jatuh ke tier "
                              "fallback (long_mult). Tier short penuh dan tier long TIDAK "
                              "terpengaruh parameter ini -- keduanya tetap pakai --leverage "
                              "biasa. Default 1.0 = tidak ada leverage tambahan di fallback.")
    parser.add_argument("--max-exposure-pct", type=float, default=None,
                         help="Override manual batas total exposure (persen). Default None "
                              "= otomatis dihitung & di-cap ke maksimal 100%% (aman, tanpa "
                              "leverage riil). Set angka besar (mis. 100000) untuk EFEKTIF "
                              "menghapus cap -- HATI-HATI, ini exposure/leverage riil "
                              "melebihi modal, bukan cuma alokasi within-100%%.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["entry_time", "exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    print(f"[*] Loaded: {args.csv}  ({len(df)} baris)")

    overall_stats(df)
    direction_stats(df)
    concentration_stats(df)
    overlap_correlation(df)
    per_pair_stats(df)

    if args.simulate:
        locked_ratio = None
        if args.short_mult is not None and args.long_mult is not None:
            locked_ratio = (args.short_mult, args.long_mult)
        run_simulation(df, args.capital_pct, args.max_positions, args.initial_equity,
                       args.leverage, locked_ratio, args.max_exposure_pct, args.fallback_leverage)
    else:
        print("\n[*] Lewati simulasi portfolio (pakai --simulate untuk mengaktifkan).")


if __name__ == "__main__":
    main()