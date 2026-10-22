"""
Bangun universe LUAS tanpa memakai pairs_train_selected.txt, bebas survivorship:

  1. Ambil daftar SEMUA simbol USDT-M futures dari listing bucket Binance Vision
     (termasuk yang sudah delisting, berbeda dengan exchangeInfo yang hanya memuat yang aktif).
  2. Unduh klines 1h (semua kolom, termasuk quote_volume) per simbol.
  3. Tulis ringkasan likuiditas/riwayat dan daftar kandidat.

Seleksi pair TIDAK dilakukan di sini dan TIDAK memakai performa strategi. Filter likuiditas
dan usia diterapkan point-in-time di robust_survivor.py / oos_survivor.py lewat
--min-qvol / --min-age-days / --top-n, jadi tiap bar hanya memakai informasi sebelum bar itu.

Output (di --out-dir):
    klines_1h\\{BASE}USDT.csv     data 1h per simbol (kolom: timestamp, open, high, low, close, volume,
                                   quote_volume, count, taker_buy_volume, taker_buy_quote_volume)
    universe_summary.csv          first/last timestamp, jumlah bar, median volume harian
    pairs_all_candidates.txt      daftar base (>= 1000 bar), siap untuk --pairs

Contoh (PowerShell, satu baris):
  py build_universe.py --out-dir "D:\\...\\binance_vision_all" --start 2019-12 --end 2026-09

Download berisi ribuan file (sekitar 300 simbol x ~80 bulan). Aman dijalankan ulang:
simbol yang CSV-nya sudah ada dilewati.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import download_binance_vision as dv

S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/klines/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_EXCLUDE = "USDC,BUSD,TUSD,USDP,FDUSD,DAI,EUR,GBP,AEUR,USTC_"


def list_symbols(session):
    """Semua prefix simbol di bawah monthly/klines/ (paginated)."""
    syms, marker = [], ""
    while True:
        url = f"{S3_LIST}?delimiter=/&prefix={PREFIX}" + (f"&marker={marker}" if marker else "")
        r = session.get(url, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        prefixes = [e.text for e in root.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        syms += [p[len(PREFIX):].strip("/") for p in prefixes]
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=NS) == "true")
        if not truncated or not prefixes:
            break
        marker = root.findtext("s3:NextMarker", default=prefixes[-1], namespaces=NS)
    return sorted(set(syms))


def filter_symbols(syms, exclude_bases):
    out = []
    for s in syms:
        if not s.endswith("USDT") or "_" in s:      # buang kontrak delivery (BTCUSDT_240628)
            continue
        base = s[:-4]
        if not base or base in exclude_bases:
            continue
        out.append(s)
    return out


def summarize(csv_path, train_end):
    df = pd.read_csv(csv_path, usecols=["timestamp", "quote_volume"])
    ts = pd.to_datetime(df["timestamp"])
    s = pd.Series(df["quote_volume"].to_numpy(float), index=ts)
    daily = s.resample("D").sum(min_count=1)
    tr = daily[daily.index <= train_end]
    return {
        "first_ts": ts.iloc[0], "last_ts": ts.iloc[-1], "n_bars": len(df),
        "med_qvol_all": daily.median(),
        "med_qvol_train_last90": tr.tail(90).median() if len(tr) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", default="2019-12")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--train-end", default="2023-12-31", help="hanya untuk kolom ringkasan likuiditas")
    ap.add_argument("--exclude-bases", default=DEFAULT_EXCLUDE, help="base yang dibuang (stablecoin dll), pisah koma")
    ap.add_argument("--only", default=None, help="opsional: file txt base untuk membatasi (tes kecil)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out_root = Path(a.out_dir)
    kdir = out_root / "klines_1h"
    kdir.mkdir(parents=True, exist_ok=True)
    session = dv.make_session(a.workers)

    print("[*] mengambil daftar simbol dari bucket Binance Vision ...")
    syms_all = list_symbols(session)
    exclude = {x.strip() for x in a.exclude_bases.split(",") if x.strip()}
    syms = filter_symbols(syms_all, exclude)
    if a.only:
        with open(a.only, "r", encoding="utf-8") as fh:
            keep = {f"{x.strip()}USDT" for x in fh if x.strip() and not x.strip().startswith("#")}
        syms = [s for s in syms if s in keep]
    print(f"[*] {len(syms_all)} prefix -> {len(syms)} simbol USDT-M perpetual (tanpa kontrak delivery/stablecoin)")

    months = dv.months_between(a.start, a.end)
    bad = []
    for i, sym in enumerate(syms, 1):
        out_path = kdir / f"{sym}.csv"
        if out_path.exists() and not a.overwrite:
            continue
        urls = [dv.url_klines(sym, "1h", ym) for ym in months]
        frames, missing, failed = dv.fetch_many(urls, dv.parse_klines, a.workers, a.retries, session)
        if failed:
            print(f"[{i}/{len(syms)}] {sym}: GAGAL {len(failed)} file -> dilewati, jalankan ulang. {failed[0]}")
            bad.append(sym)
            continue
        if not frames:
            print(f"[{i}/{len(syms)}] {sym}: tidak ada data")
            continue
        print(f"[{i}/{len(syms)}] {sym}")
        dv.finish_and_write(frames, out_path, f"1h ({len(frames)} file ok, {missing} tidak ada)", 60)

    print("\n[*] menyusun ringkasan ...")
    train_end = pd.Timestamp(a.train_end)
    rows = []
    for sym in syms:
        p = kdir / f"{sym}.csv"
        if not p.exists():
            continue
        r = summarize(p, train_end)
        r["symbol"] = sym
        rows.append(r)
    summ = pd.DataFrame(rows).set_index("symbol").sort_values("med_qvol_train_last90", ascending=False)
    summ.to_csv(out_root / "universe_summary.csv")
    good = [s[:-4] for s in summ.index if summ.loc[s, "n_bars"] >= 1000]
    with open(out_root / "pairs_all_candidates.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(good))

    pd.set_option("display.width", 200)
    print(f"\n{len(summ)} simbol terunduh | {len(good)} dengan >= 1000 bar | gagal jaringan: {len(bad)}")
    print("\n20 simbol dengan volume (median harian, 90 hari terakhir train) terbesar:")
    print(summ[["first_ts", "last_ts", "n_bars", "med_qvol_train_last90"]].head(20).to_string(
        float_format=lambda v: f"{v:,.0f}"))
    for thr in (1e6, 5e6, 2e7, 1e8):
        print(f"  simbol dengan volume harian >= ${thr:,.0f} di akhir train: "
              f"{int((summ['med_qvol_train_last90'] >= thr).sum())}")
    if bad:
        print(f"\nJalankan ulang perintah yang sama untuk mengulang yang gagal: {bad[:10]}")
    print(f"\n[+] {out_root}")
    print("Berikutnya: gunakan pairs_all_candidates.txt sebagai --pairs dan klines_1h sebagai --data-dir.")


if __name__ == "__main__":
    main()