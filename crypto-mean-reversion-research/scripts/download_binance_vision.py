"""
Unduh data Binance Vision (USDT-M futures) untuk riset order flow:
  1. klines   : 15m (default), semua kolom termasuk taker_buy_volume  -> klines_15m/{BASE}USDT.csv
  2. funding  : funding rate (file bulanan)                           -> funding/{BASE}USDT.csv
  3. metrics  : open interest, long/short ratio, taker ratio (file HARIAN) -> metrics/{BASE}USDT.csv

Struktur output (di bawah --out-dir):
    klines_15m\\  funding\\  metrics\\      (satu CSV per pair, nama sama dengan folder 1h)

Perilaku:
  - File yang tidak ada di server (404/403) dianggap "tidak ada data" (normal untuk bulan
    sebelum listing / setelah delisting, atau metrics sebelum Binance mulai menerbitkannya).
  - Error jaringan di-retry. Kalau masih gagal, CSV pair/tipe itu TIDAK ditulis supaya tidak
    ada data bolong yang diam-diam; jalankan ulang perintah yang sama (pair yang sudah
    selesai otomatis dilewati).
  - metrics: titik mulai dideteksi otomatis dari BTCUSDT (probe tiap bulan) supaya tidak
    membuang ribuan request ke tanggal yang memang belum ada datanya.

Contoh (PowerShell, satu baris):
  py download_binance_vision.py --pairs "D:\\...\\pairs_train_selected.txt" --out-dir "D:\\...\\binance_vision" --start 2019-12 --end 2023-12

Hanya klines + funding (lebih cepat):
  ... --types klines,funding
"""

import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
METRIC_COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
               "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
               "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


SPOT_BASE = "https://data.binance.vision/data/spot"


def spot_symbol(base):
    """Perp '1000SHIB' -> spot ('SHIBUSDT', skala 1000); perp biasa -> (BASEUSDT, 1.0)."""
    for pre in ("1000000", "1000"):
        if base.startswith(pre) and len(base) > len(pre):
            return base[len(pre):] + "USDT", float(pre)
    return base + "USDT", 1.0


def url_klines_spot(sym, interval, ym):
    return f"{SPOT_BASE}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{ym}.zip"


class HardError(Exception):
    pass


# ───────────────────────── jaringan ─────────────────────────
def http_get(url, retries, session):
    """bytes jika ada, None jika file tidak ada (404/403). HardError jika gagal setelah retry."""
    last = None
    for i in range(retries + 1):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(1.5 * (i + 1))
    raise HardError(f"{url}: {last}")


def url_exists(url, session):
    try:
        r = session.head(url, timeout=30)
        return r.status_code == 200
    except requests.RequestException:
        return False


def make_session(workers):
    s = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    s.mount("https://", ad)
    return s


# ───────────────────────── URL ─────────────────────────
def url_klines(sym, interval, ym):
    return f"{BASE}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{ym}.zip"


def url_funding(sym, ym):
    return f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"


def url_metrics(sym, day):
    return f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{day}.zip"


# ───────────────────────── parsing ─────────────────────────
def _read_zip_csv(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        raw = z.read(name).decode("utf-8", errors="replace")
    first = raw.split("\n", 1)[0].split(",")[0].strip()
    has_header = any(ch.isalpha() for ch in first)  # header: open_time/calc_time/create_time
    df = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
    return df


def _to_ts(series):
    v = pd.to_numeric(series, errors="coerce").astype("int64")
    if v.max() > 10 ** 14:  # mikrodetik -> milidetik
        v = v // 1000
    return pd.to_datetime(v, unit="ms")


def parse_klines(content):
    df = _read_zip_csv(content).iloc[:, :12]
    df.columns = KLINE_COLS
    out = df.drop(columns=["close_time", "ignore", "open_time"]).copy()
    out.insert(0, "timestamp", _to_ts(df["open_time"]))
    return out


def parse_funding(content):
    df = _read_zip_csv(content)
    out = pd.DataFrame({"timestamp": _to_ts(df.iloc[:, 0]),
                        "funding_rate": pd.to_numeric(df.iloc[:, -1], errors="coerce")})
    if df.shape[1] >= 3:
        out["interval_hours"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return out


def parse_metrics(content):
    df = _read_zip_csv(content)
    if "create_time" not in df.columns:  # tanpa header
        df = df.iloc[:, :len(METRIC_COLS)]
        df.columns = METRIC_COLS[:df.shape[1]]
    df = df.drop(columns=[c for c in ("symbol",) if c in df.columns])
    df = df.rename(columns={"create_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ───────────────────────── unduh satu pair/tipe ─────────────────────────
def fetch_many(urls, parser, workers, retries, session):
    frames, missing, failed = [], 0, []

    def work(u):
        content = http_get(u, retries, session)
        if content is None:
            return u, None
        return u, parser(content)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, u) for u in urls]
        for f in futures:
            try:
                u, df = f.result()
            except HardError as e:
                failed.append(str(e))
                continue
            except Exception as e:  # file rusak / format tak terduga
                failed.append(f"parse error: {e}")
                continue
            if df is None:
                missing += 1
            else:
                frames.append(df)
    return frames, missing, failed


def finish_and_write(frames, out_path, label, interval_min=None):
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(out_path, index=False)
    first, last = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    note = ""
    if interval_min:
        span = (pd.Timestamp(last) - pd.Timestamp(first)) / pd.Timedelta(minutes=interval_min)
        gaps = int(round(span)) + 1 - len(df)
        note = f" | bar hilang (gap): {gaps}"
    print(f"    {label}: {len(df):,} baris | {first} -> {last}{note}")


def months_between(start, end):
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def days_between(start_ts, end_ts):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start_ts, end_ts, freq="D")]


def detect_metrics_start(start, end, session):
    """Probe BTCUSDT tgl 15 tiap bulan -> bulan pertama yang ada datanya."""
    for ym in months_between(start, end):
        day = f"{ym}-15"
        if url_exists(url_metrics("BTCUSDT", day), session):
            return pd.Timestamp(f"{ym}-01")
    return None


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="txt berisi base per baris (BTC otomatis ditambahkan)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", default="2019-12", help="YYYY-MM")
    ap.add_argument("--end", default="2023-12", help="YYYY-MM")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--types", default="klines,funding,metrics")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--metrics-start", default=None, help="YYYY-MM-DD, override deteksi otomatis")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--market", choices=["um", "spot"], default="um",
                    help="spot: unduh klines SPOT untuk basis carry (hanya klines; folder spot_klines_<interval>)")
    a = ap.parse_args()

    out_root = Path(a.out_dir)
    dirs = {"klines": out_root / f"klines_{a.interval}", "funding": out_root / "funding",
            "metrics": out_root / "metrics"}
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    if a.market == "spot":
        types = ["klines"]
        dirs["klines"] = out_root / f"spot_klines_{a.interval}"
    for t in types:
        dirs[t].mkdir(parents=True, exist_ok=True)

    with open(a.pairs, "r", encoding="utf-8") as f:
        bases = [x.strip() for x in f if x.strip() and not x.strip().startswith("#")]
    bases = list(dict.fromkeys(["BTC"] + bases))

    session = make_session(a.workers)
    interval_min = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(a.interval)
    months = months_between(a.start, a.end)

    metrics_days = None
    if "metrics" in types:
        if a.metrics_start:
            m_start = pd.Timestamp(a.metrics_start)
        else:
            print("[*] mendeteksi awal data metrics (probe BTCUSDT) ...")
            m_start = detect_metrics_start(a.start, a.end, session)
        if m_start is None:
            print("[!] metrics tidak ditemukan dalam rentang --start..--end, metrics dilewati.")
            types.remove("metrics")
        else:
            end_ts = pd.Period(a.end, freq="M").end_time.normalize()
            metrics_days = days_between(m_start, end_ts)
            print(f"[*] metrics tersedia mulai ~{m_start.date()} ({len(metrics_days)} hari/pair). "
                  f"Periode sebelum itu TIDAK akan punya data open interest.")

    summary = []
    print(f"[*] {len(bases)} pair | tipe={types} | {a.start} -> {a.end} | workers={a.workers}")
    for i, base in enumerate(bases, 1):
        sym = spot_symbol(base)[0] if a.market == "spot" else f"{base}USDT"
        print(f"\n[{i}/{len(bases)}] {sym}")
        for t in types:
            out_path = dirs[t] / f"{sym}.csv"
            if out_path.exists() and not a.overwrite:
                print(f"    {t}: sudah ada, dilewati")
                continue
            if t == "klines":
                urls = [(url_klines_spot if a.market == "spot" else url_klines)(sym, a.interval, ym) for ym in months]
                parser = parse_klines
            elif t == "funding":
                urls = [url_funding(sym, ym) for ym in months]
                parser = parse_funding
            else:
                urls = [url_metrics(sym, d) for d in metrics_days]
                parser = parse_metrics

            frames, missing, failed = fetch_many(urls, parser, a.workers, a.retries, session)
            if failed:
                print(f"    {t}: GAGAL {len(failed)} file -> CSV tidak ditulis, jalankan ulang. Contoh: {failed[0]}")
                summary.append((sym, t, "GAGAL"))
                continue
            if not frames:
                print(f"    {t}: tidak ada data di server ({missing} file 404)")
                summary.append((sym, t, "KOSONG"))
                continue
            finish_and_write(frames, out_path, f"{t} ({len(frames)} file ok, {missing} tidak ada)",
                             interval_min if t == "klines" else None)
            summary.append((sym, t, "OK"))

    bad = [s for s in summary if s[2] != "OK"]
    print("\n" + "=" * 70)
    print(f"Selesai. OK={len(summary) - len(bad)} | bermasalah={len(bad)}")
    for sym, t, st in bad:
        print(f"  {sym:12s} {t:8s} {st}")
    if any(s[2] == "GAGAL" for s in bad):
        print("Ada yang GAGAL (jaringan): jalankan ulang perintah yang sama, yang sudah selesai dilewati.")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()