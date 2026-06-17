# -*- coding: utf-8 -*-
"""
AWS EMR entry point for the delivery data ingestion pipeline.

Thin, readable spark-submit driver: it builds the SparkSession the EMR/YARN way
(no hard-coded master), parses I/O locations (local paths or S3 prefixes), and
delegates the actual batch ingestion to dataset.py.

Each city is ingested independently and written to its own prefix, so cities
stay fully separated.

EMR (cluster mode) -- ship the pipeline module with --py-files:
    spark-submit \
        --deploy-mode cluster \
        --py-files s3://<bucket>/code/dataset.py \
        s3://<bucket>/code/data.py \
        --input  s3://<bucket>/lade-d/data \
        --output s3://<bucket>/research_impl/dataset

EMR step (alternative): set Main JAR = command-runner.jar, args =
    spark-submit --py-files s3://<bucket>/code/dataset.py
    s3://<bucket>/code/data.py --input s3://... --output s3://...

Local:
    spark-submit research_impl/dataset/data.py \
        --input data/raw_data/delivery --output research_impl/dataset
"""
import argparse

from pyspark.sql import SparkSession

# Works both as a package module and on EMR where dataset.py is shipped flat
# via --py-files (so it imports as a top-level module).
try:
    from research_impl.dataset.dataset import run, CITY_CODE
except ImportError:  # pragma: no cover - EMR --py-files layout
    from dataset import run, CITY_CODE


def build_spark(app_name="LaDe-Delivery-Ingestion"):
    """SparkSession suitable for EMR: master/resources come from the cluster."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def parse_args():
    p = argparse.ArgumentParser(description="LaDe-D delivery ingestion (EMR batch)")
    p.add_argument('--input', required=True,
                   help="dir or S3 prefix containing delivery_<code>.parquet files")
    p.add_argument('--output', required=True,
                   help="dir or S3 prefix for ingested per-city tables")
    p.add_argument('--cities', nargs='+', default=list(CITY_CODE.keys()),
                   help="subset of cities to ingest (default: all in scope)")
    p.add_argument('--no-csv', action='store_true',
                   help="skip the human-readable CSV copy (parquet only)")
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark()
    try:
        print(f"[*] Spark {spark.version} | ingesting {args.cities}")
        run(spark, args.cities, args.input, args.output, write_csv=not args.no_csv)
        print("[+] Ingestion complete.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
