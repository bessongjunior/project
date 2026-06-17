# -*- coding: utf-8 -*-
"""
Acquire the LaDe-D delivery dataset and stage it locally in both formats.

  research_impl/dataset/parquets/delivery_<code>.parquet -> PySpark ingestion (dataset.py / data.py)
  research_impl/dataset/csv/delivery_<code>.csv         -> pandas path (pre_processing/preprocess.py)

Two modes:
  download  : pull each city from HuggingFace, write parquet + csv      (needs network)
  convert   : turn already-downloaded parquets/ into csv/               (offline)

Requires: pandas, pyarrow, huggingface_hub, fsspec  (run locally, not on EMR).
HF source: https://huggingface.co/datasets/Cainiao-AI/LaDe-D
"""
import os
import argparse

import pandas as pd

from research_impl.pre_processing.utils import ws, dir_check, CITY_CODE

HF_BASE = "hf://datasets/Cainiao-AI/LaDe-D/"

# HF split filenames, keyed by city code (cq, hz, jl, sh, yt).
HF_SPLITS = {
    'cq': 'data/delivery_cq-00000-of-00001-465887add76aeabc.parquet',
    'hz': 'data/delivery_hz-00000-of-00001-8090c86f64781f71.parquet',
    'jl': 'data/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet',
    'sh': 'data/delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet',
    'yt': 'data/delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet',
}


def _dirs(parquet_dir, csv_dir):
    parquet_dir = parquet_dir or os.path.join(ws, 'research_impl', 'dataset', 'parquets')
    csv_dir = csv_dir or os.path.join(ws, 'research_impl', 'dataset', 'csv')
    return parquet_dir, csv_dir


def download(cities=None, parquet_dir=None, csv_dir=None, to_csv=True):
    """Pull each city from HuggingFace -> parquets/ (+ csv/)."""
    cities = cities or list(CITY_CODE.keys())
    parquet_dir, csv_dir = _dirs(parquet_dir, csv_dir)

    for city in cities:
        code = CITY_CODE.get(city, city)
        if code not in HF_SPLITS:
            print(f"[!] no HF split for {city} ({code}); skipping")
            continue
        print(f"[*] {city}: reading {HF_BASE + HF_SPLITS[code]}")
        df = pd.read_parquet(HF_BASE + HF_SPLITS[code])

        pq = os.path.join(parquet_dir, f'delivery_{code}.parquet')
        dir_check(pq)
        df.to_parquet(pq, index=False)
        print(f"[+] {city}: {len(df)} rows -> {pq}")

        if to_csv:
            csv = os.path.join(csv_dir, f'delivery_{code}.csv')
            dir_check(csv)
            df.to_csv(csv, index=False)
            print(f"[+] {city}: -> {csv}")


def convert(cities=None, parquet_dir=None, csv_dir=None):
    """Offline: existing parquets/ -> csv/ (no network)."""
    cities = cities or list(CITY_CODE.keys())
    parquet_dir, csv_dir = _dirs(parquet_dir, csv_dir)

    for city in cities:
        code = CITY_CODE.get(city, city)
        pq = os.path.join(parquet_dir, f'delivery_{code}.parquet')
        if not os.path.exists(pq):
            print(f"[!] {city}: {pq} not found; skipping")
            continue
        csv = os.path.join(csv_dir, f'delivery_{code}.csv')
        dir_check(csv)
        pd.read_parquet(pq).to_csv(csv, index=False)
        print(f"[+] {city}: {pq} -> {csv}")


def main():
    p = argparse.ArgumentParser(description="Download / convert LaDe-D delivery data")
    p.add_argument('--mode', choices=['download', 'convert'], default='download')
    p.add_argument('--cities', nargs='+', default=list(CITY_CODE.keys()))
    p.add_argument('--parquet_dir', default=None, help="default: research_impl/dataset/parquets")
    p.add_argument('--csv_dir', default=None, help="default: research_impl/dataset/csv")
    p.add_argument('--no-csv', action='store_true', help="download: parquet only")
    args = p.parse_args()

    if args.mode == 'download':
        download(args.cities, args.parquet_dir, args.csv_dir, to_csv=not args.no_csv)
    else:
        convert(args.cities, args.parquet_dir, args.csv_dir)


if __name__ == "__main__":
    main()
