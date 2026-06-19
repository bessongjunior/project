# An Optimal Big Data Architecture for Data Management and Route Optimisation for Parcel and Mail Delivery

A reproducible big-data pipeline that unifies **data management** and **route /
travel-time optimisation** for last-mile parcel and mail delivery. Raw delivery
logs are ingested at scale with **Apache Spark on Amazon EMR**, stored in an
**Amazon S3 data lake**, grounded in the real road network via **OpenStreetMap
map-matching**, turned into per-trajectory spatio-temporal graphs, and used to
train and evaluate **graph neural-network** route/ETA models against classical and
learned baselines.

Dataset: **LaDe-D** (Cainiao last-mile delivery) — cities `cq, sh, hz, jl, yt`.

---

## 1. Architecture

The system is a layered AWS big-data architecture (see
`research_impl/big_data_architecture.svg`):

```
Data Sources (LaDe-D · OpenStreetMap)
        │  acquire & stage
Amazon S3 — Data Lake (raw/ · processed/ · dataset/ · artifacts/)
        │  read / write (Spark)
Amazon EMR — Hadoop / Spark (YARN · HDFS)
   Stage 1 Ingestion → Stage 2 Graph Extraction → Stage 3 Map-Matching → Stage 4 Tensorisation
        │  train / val / test tensors
Analytics & ML (PyTorch): models · training · baselines · evaluation
        │  results
Outputs (weights · metrics · grades)  → persisted to S3
Orchestration & Governance: EMR Steps · spark-submit · CloudWatch · IAM · S3 encryption
```

## 2. Repository layout (`research_impl/`)

| Path | Stage / role |
|------|--------------|
| `dataset/download.py` | acquire LaDe-D from Hugging Face → `dataset/csv/` + `dataset/parquets/` |
| `dataset/dataset.py` · `dataset/data.py` | **Spark ingestion** + the **EMR spark-submit driver** |
| `pre_processing/preprocess.py` | pandas ingestion (small/local alternative) |
| `extraction/map_data.py` | OSM road-graph extraction (OSMnx) |
| `pre_processing/map_matcher.py` | snap to OSM nodes + true road distance |
| `dataset/tensorize.py` | build `V, A, L, labels, ETA, masks` tensors |
| `algorithms/` | ST-GCN, T-GCN, Graph2Route, M2G4RTP, DRL4Route, FDNet, shared `pointer.py`, `baseline/` |
| `train.py` | training harness (beam search, best-epoch checkpoint) |
| `evaluation/eval.py` | metrics (HR@K, KRC, LSD, ED, MAE/RMSE/MAPE/ACC@T) + A–D grading |

## 3. Prerequisites

- Python 3.12, dependencies in `requirements.txt` (CPU PyTorch, Spark, OSMnx,
  NetworkX, scikit-learn, OR-Tools, LightGBM, …).
- For EMR: an AWS account, an S3 bucket, and EMR with Spark.

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

## 4. Quick start (local, single city)

See `how-to-run.md` for the full single-city walk-through (e.g. Jilin). In short:

```bash
venv/bin/python -m research_impl.dataset.download --cities jilin
venv/bin/python -m research_impl.pre_processing.preprocess --cities jilin
# extract the city OSM graph (bounding box of the delivery points) — see how-to-run.md
venv/bin/python -c "from research_impl.pre_processing.map_matcher import run; run('jilin')"
venv/bin/python -m research_impl.dataset.tensorize --cities jilin
venv/bin/python -m research_impl.train --model graph2route --city jilin
venv/bin/python -m research_impl.evaluation.eval --city jilin --beam 5
```

---

## 5. Running on an AWS EMR cluster

The Spark ingestion is the distributed, EMR-native stage; the remaining stages run
as Python jobs on the cluster primary node, reading/writing the S3 data lake.

### 5.1 Stage code and data to S3
```bash
aws s3 cp research_impl s3://<bucket>/code/research_impl --recursive
# raw LaDe-D parquet (download once, then upload)
venv/bin/python -m research_impl.dataset.download           # writes dataset/parquets/
aws s3 cp research_impl/dataset/parquets s3://<bucket>/lade-d/ --recursive
```

### 5.2 Launch an EMR cluster (Spark) with a dependency bootstrap
Create `bootstrap.sh` (installs the Python deps on every node) and upload it:
```bash
#!/bin/bash
sudo python3 -m pip install -r /tmp/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```
```bash
aws s3 cp requirements.txt s3://<bucket>/code/requirements.txt
aws s3 cp bootstrap.sh    s3://<bucket>/code/bootstrap.sh
aws emr create-cluster --name "delivery-bigdata" --release-label emr-7.1.0 \
  --applications Name=Spark Name=Hadoop \
  --instance-type m5.xlarge --instance-count 3 \
  --ec2-attributes KeyName=<key>,SubnetId=<subnet> \
  --service-role EMR_DefaultRole --use-default-roles \
  --bootstrap-actions Path=s3://<bucket>/code/bootstrap.sh \
  --log-uri s3://<bucket>/emr-logs/
```

### 5.3 Stage 1 — Spark ingestion (distributed)
Submit the EMR driver `dataset/data.py`, shipping the pipeline module with
`--py-files`; input/output are S3 prefixes:
```bash
aws emr add-steps --cluster-id <cluster-id> --steps Type=CUSTOM_JAR,\
Jar=command-runner.jar,Name=ingest,Args=[spark-submit,--deploy-mode,cluster,\
--py-files,s3://<bucket>/code/research_impl/dataset/dataset.py,\
s3://<bucket>/code/research_impl/dataset/data.py,\
--input,s3://<bucket>/lade-d,--output,s3://<bucket>/research_impl/dataset]
```
This writes `s3://<bucket>/research_impl/dataset/tmp/<city>/package_feature.parquet`.

### 5.4 Stages 2–6 — Python pipeline (primary node)
SSH to the primary node (or add EMR Steps with `command-runner.jar` calling
`python3 -m …`). Sync the ingested data down, run the stages, then sync results up:
```bash
aws s3 sync s3://<bucket>/research_impl/dataset research_impl/dataset
python3 -m research_impl.extraction.map_data            # OSM graphs  (needs internet egress)
python3 -m research_impl.pre_processing.map_matcher     # snap + road distance
python3 -m research_impl.dataset.tensorize              # tensors
python3 -m research_impl.train --model graph2route --city shanghai --epochs 40 --hidden 128
python3 -m research_impl.evaluation.eval --city shanghai --beam 5 --rush-hour
aws s3 sync research_impl/evaluation/outputs s3://<bucket>/research_impl/outputs
```

### 5.5 Notes
- **Scale:** Spark ingestion handles the large cities (e.g. Shanghai, 1.48 M rows)
  that the pandas path cannot; training/eval run on the primary node (GPU optional —
  CPU is the default).
- **Governance:** use IAM roles for S3 access, enable S3 encryption (SSE), and view
  driver/executor logs in the EMR `--log-uri` bucket / CloudWatch.
- **OSM extraction** needs outbound internet from the cluster (NAT gateway) to query
  OpenStreetMap; alternatively pre-extract graphs locally and upload to
  `s3://<bucket>/research_impl/processed/`.

---

## 6. Outputs

Per city, evaluation writes to `research_impl/evaluation/outputs/<city>/latest/`
(and a timestamped copy):
- `metrics.csv` — every model and baseline × all metrics
- `grades.json` — A–D grade and gate results
- `summary.txt` — the console report

## 7. Models and baselines
- **Route / ETA models:** ST-GCN, T-GCN (travel-time), Graph2Route, FDNet
  (route + ETA), M2G4RTP, DRL4Route.
- **Baselines:** Distance-Greedy, Time-Greedy, OR-Tools (TSP), LightGBM.
- **Metrics:** route — HR@K, KRC, LSD, ED; ETA — MAE, RMSE, MAPE, ACC@T; with an
  A–D grade relative to the baselines.
