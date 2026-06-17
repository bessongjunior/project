# -*- coding: utf-8 -*-
"""
Delivery data ingestion pipeline  (PySpark, batch).

Pipeline stage 1. Reads the raw LaDe-D delivery parquet files (one per city) and
produces the cleaned, feature-engineered per-city tables that feed the
map-matching bridge (pre_processing/map_matcher.py) and, after that, the tensor
builder (dataset/tensorize.py).

Spark-native by design so it scales on a cluster: the trajectory features
(time-to-last, distance-to-last) are computed with window functions instead of
the per-courier Python loops used in the original pandas preprocess.py.

  Input  : <input_dir>/delivery_<code>.parquet                (code = cq,hz,jl,sh,yt)
  Output : <output_dir>/tmp/<city>/package_feature.parquet     (+ _csv/ for inspection)

Run locally via spark-submit, or on AWS EMR via dataset/data.py.
LaDe-D: https://huggingface.co/datasets/Cainiao-AI/LaDe-D

NOTE: Spark cannot read `hf://` URLs. Download the parquet files first (e.g. with
the `datasets`/`huggingface_hub` client) and land them on a Spark-readable path
(local dir, S3, or HDFS), then point --input there.
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# Self-contained registry (mirrors pre_processing.utils.CITY_CODE) so this module
# ships to an EMR cluster without the rest of the research_impl package.
CITY_CODE = {
    'shanghai': 'sh', 'hangzhou': 'hz', 'chongqing': 'cq',
    'jilin': 'jl', 'yantai': 'yt',
}

# Schema the downstream map_matcher / tensorize stages rely on.
OUTPUT_COLS = [
    'order_id', 'courier_id', 'ds', 'lat', 'lng', 'aoi_type',
    'accept_time_minute', 'finish_time_minute', 'expect_finish_time_minute',
    'time_to_last_package', 'dis_to_last_package',
]


# ---------------------------------------------------------------------------
# Column expressions (LaDe time strings are 'MM-dd HH:mm:ss')
# ---------------------------------------------------------------------------

def _minute_of_day(col):
    """'MM-dd HH:mm:ss' -> minute-of-day (float)."""
    hms = F.split(F.split(col, ' ').getItem(1), ':')
    return (hms.getItem(0).cast('int') * 60
            + hms.getItem(1).cast('int')
            + hms.getItem(2).cast('int') / 60.0)


def _day_key(col):
    """'MM-dd ...' -> integer MMdd, for cross-day accept detection."""
    md = F.split(F.split(col, ' ').getItem(0), '-')
    return md.getItem(0).cast('int') * 100 + md.getItem(1).cast('int')


def _haversine(lat1, lng1, lat2, lng2):
    """Great-circle distance in metres (Spark column expression)."""
    R = 6371000.0
    a = (F.sin(F.radians(lat2 - lat1) / 2) ** 2
         + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2))
         * F.sin(F.radians(lng2 - lng1) / 2) ** 2)
    return 2 * R * F.asin(F.sqrt(F.least(F.lit(1.0), a)))


# ---------------------------------------------------------------------------
# Per-city ingestion
# ---------------------------------------------------------------------------

def ingest_city(spark, city, input_dir, output_dir, write_csv=True):
    code = CITY_CODE.get(city, city)
    # glob matches both clean (delivery_cq.parquet) and raw HF (delivery_cq-00000-...) names
    src = f"{input_dir.rstrip('/')}/delivery_{code}*.parquet"
    print(f"[*] Ingesting {city} <- {src}")
    df = spark.read.parquet(src)

    # --- time features (minute-of-day, cross-day accept adjustment) ---
    df = (df
          .withColumn('finish_time_minute', _minute_of_day(F.col('delivery_time')))
          .withColumn('_accept_raw', _minute_of_day(F.col('accept_time')))
          .withColumn('_dday', _day_key(F.col('delivery_time')))
          .withColumn('_aday', _day_key(F.col('accept_time')))
          .withColumn('accept_time_minute',
                      F.when(F.col('_dday') != F.col('_aday'),
                             F.col('_accept_raw') - 1440.0)
                       .otherwise(F.col('_accept_raw'))))

    # LaDe-D delivery has no promised window -> default a full day.
    if 'expect_finish_time_minute' not in df.columns:
        df = df.withColumn('expect_finish_time_minute', F.lit(1440.0))
    if 'aoi_type' not in df.columns:
        df = df.withColumn('aoi_type', F.lit(0))

    # --- trajectory features: previous stop within the same courier-day ---
    w = Window.partitionBy('courier_id', 'ds').orderBy('finish_time_minute')
    df = (df
          .withColumn('_plat', F.lag('lat').over(w))
          .withColumn('_plng', F.lag('lng').over(w))
          .withColumn('_pft', F.lag('finish_time_minute').over(w))
          .withColumn('time_to_last_package',
                      F.when(F.col('_pft').isNull(), F.lit(0.0))
                       .otherwise(F.col('finish_time_minute') - F.col('_pft')))
          .withColumn('dis_to_last_package',
                      F.when(F.col('_plat').isNull(), F.lit(0.0))
                       .otherwise(_haversine(F.col('_plat'), F.col('_plng'),
                                             F.col('lat'), F.col('lng')))))

    out = df.select(*[c for c in OUTPUT_COLS if c in df.columns])

    dest = f"{output_dir.rstrip('/')}/tmp/{city}/package_feature"
    out.write.mode('overwrite').parquet(dest + '.parquet')
    if write_csv:
        out.coalesce(1).write.mode('overwrite').option('header', True).csv(dest + '_csv')
    print(f"[+] {city}: wrote {dest}.parquet")
    return out


def run(spark, cities, input_dir, output_dir, write_csv=True):
    """Batch-ingest every requested city, keeping each city's output separate."""
    for city in cities:
        try:
            ingest_city(spark, city, input_dir, output_dir, write_csv=write_csv)
        except Exception as e:                       # noqa: BLE001 - keep batch going
            print(f"[!] {city}: {e}")


if __name__ == "__main__":
    # Standalone local run; data.py is the configurable EMR entry point.
    import argparse
    p = argparse.ArgumentParser(description="LaDe-D delivery ingestion (PySpark)")
    p.add_argument('--input', default='research_impl/dataset/parquets')
    p.add_argument('--output', default='research_impl/dataset')
    p.add_argument('--cities', nargs='+', default=list(CITY_CODE.keys()))
    p.add_argument('--no-csv', action='store_true')
    args = p.parse_args()

    spark = SparkSession.builder.appName("LaDe-Delivery-Ingestion").getOrCreate()
    try:
        run(spark, args.cities, args.input, args.output, write_csv=not args.no_csv)
    finally:
        spark.stop()
