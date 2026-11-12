# TabICLv2 on Google Cloud TPU v7x (Ironwood)

Single-host and multi-host inference and synthetic pretraining benchmarks for [TabICLv2](https://github.com/soda-inria/tabicl) on Google Cloud TPU v7x (Ironwood) with PyTorch and TorchTPU.

**Summary:** Using initial power-of-two shape bucketing (`fp32` column stage, `bf16` 3-pass row stage, `bf16` in-context learning), **one TPU v7x TensorCore (half a physical chip) runs 50k-row model-forward inference in 21.13 s uncached (2.8× slower than the ~7.5 s published 1× H100 NVL end-to-end reference) and 2.32 s per repeated query from a prebuilt KV cache (9.10× faster than an uncached forward pass)**. Scaling across all 8 TensorCores (4 physical chips) on a single `v7x` host reaches **0.38 datasets/s uncached and 3.43 datasets/s (17,128 test rows/s) for KV-cached queries (7.95× scaling over one TensorCore)**, while passing numerical parity across all 15 OpenML classification and regression datasets.

---

## 1. Model, Framework, and Hardware

- **Model:** TabICLv2 (*Tabular In-Context Learning v2*, [arXiv:2602.11139](https://arxiv.org/abs/2602.11139)) is an open-source tabular foundation model for classification and regression that predicts test labels from labeled training rows in a single forward pass.
  - **Parameters & Weights:** **27.55M parameters** for the classifier (28.54M for the quantile regressor). Inference runs an **8-estimator ensemble** in mixed precision: `float32` (`highest`) for the column Set Transformer, 3-pass `bfloat16` (`high`) for row interaction, and `bfloat16` autocast for the 12 in-context learning blocks (storing KV caches in `bfloat16`).
  - **Architecture:** Three sequential stages — (1) a 3-block column-wise Set Transformer (`width 128`, `8 heads × 16 dim`, `128` inducing points), (2) a 3-block row-wise interaction transformer (`width 128`, `8 heads × 16 dim`, RoPE + `4` CLS tokens), and (3) a 12-block dataset-wise in-context transformer (`width 512`, `8 heads × 64 dim`).
- **Framework:** Python 3.12, PyTorch 2.15.0, `torch_tpu` 0.2.0, `libtpu` 0.0.50 (`tpu_dist` ICI backend), and `tabicl` 2.2.0 compiled with shape-bucketed `torch.compile(backend="tpu", fullgraph=True)` ([`scripts/tabicl_tpu_bucketed.py`](./scripts/tabicl_tpu_bucketed.py)) and Pallas Splash Attention for context lengths `≥16,384` ([`results/tpu_patch.diff`](./results/tpu_patch.diff)).
- **Hardware:** Google Cloud TPU v7x (Ironwood) `2x2x2` ICI slice (`2` hosts × `4` dual-core physical chips = **8 physical chips / 16 TensorCores**, `192 GiB` HBM3e per physical chip / **`96 GiB` HBM3e per TensorCore**). TorchTPU exposes each TensorCore as one PyTorch device (`tpu:0..7` per host).

---

## 2. Main Results

### A. Inference Latency & Host Scaling (`45k` Train + `5k` Test × `100` Features, `8` Estimators)

All configurations below pass numerical parity against CPU `fp32` across 15 OpenML datasets (`max |ΔAUC| = 0.00197`, `max relative RMSE diff = 0.076%`; [`results/accuracy/`](./results/accuracy/)).

| Topology | Mode | Warm `p50` Latency | Throughput | Peak HBM / TensorCore | Comparison |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **1 TensorCore (`0.5` `v7x` chip)** | One-shot (no KV cache) | **21.13 s** | 0.047 datasets/s (`237 rows/s`) | 49.4 GiB | **2.8× slower** than ~7.5 s published `1× H100 NVL` end-to-end plot (see notes) |
| **1 TensorCore (`0.5` `v7x` chip)** | Repeated query (`kv_cache=True`) | **2.32 s** | 0.431 datasets/s (`2,154 rows/s`) | 64.1 GiB | **9.10× faster** per query than a 21.13 s uncached forward pass (excludes ~20.6 s cache build) |
| **1 Host (`8` TensorCores = `4` `v7x` chips)** | One-shot (`8` datasets in parallel) | **21.11 s** | **0.379 datasets/s** (`1,894 rows/s`) | 53.6 GiB | **8.00× throughput** vs. 1 TensorCore (`100%` within-host scaling) |
| **1 Host (`8` TensorCores = `4` `v7x` chips)** | Repeated query (`kv_cache=True`) | **2.33 s** | **3.43 datasets/s** (**`17,128 rows/s`**) | 75.4 GiB | **7.95× throughput** vs. 1 TensorCore; **9.04× faster** than uncached host forward |
| **2 Hosts via global `tpu_dist` (`16` TensorCores = `8` `v7x` chips)** | One-shot / KV query (`16` datasets) | 60.58 s / 8.45 s | 0.265 / 1.91 datasets/s | 66.3 / 68.5 GiB | **30–44% lower throughput** than 1 independent host due to `tpu_dist` barrier & host-memory staging |

*Measurement notes:*
1. **Model-forward scope & bucket padding:** Reported `v7x` latencies measure the compiled model forward pass on pre-allocated tensors (excluding CPU scikit-learn `fit`) and use power-of-two buckets (`50,000` rows padded to `65,536`, `100` features padded to `128`) with `fp32` column matmuls. Upstream H100 references ("under 10 s" in README text; `~7.5 s` on the `1× H100 NVL` curve in [`soda-inria/tabicl`](https://github.com/soda-inria/tabicl)) measure end-to-end `fit + predict`. *(See [`tabiclv2-torchtpu-v5p-benchmark`](https://github.com/ankdeshm/tabiclv2-torchtpu-v5p-benchmark) for subsequent finer bucketing and custom small-head Pallas kernels.)*
2. **KV cache is for repeated queries only:** Building the KV cache once costs roughly the same as a full uncached pass (`20.6 s` build + `0.35 s` query vs `21.1 s` one-shot at `50k + 1k × 100`).

### B. Context-Length Scaling on One TensorCore (`1,000` Test Rows, `100` Features, `8` Estimators)

| Training Rows | One-Shot Forward (`nocache`) | KV Cache Build (`prefill`) | Repeated Query (`kv_cache=True`) | Query Speedup vs. Uncached Forward |
| ---: | ---: | ---: | ---: | :--- |
| 1,000 | 0.45 s | 0.24 s | **0.19 s** | **2.38× faster** than uncached forward |
| 10,000 | 3.95 s | 3.92 s | **0.22 s** | **18.3× faster** than uncached forward |
| 50,000 | 21.15 s | 20.63 s | **0.35 s** | **60.9× faster** than uncached forward |
| 100,000 | 53.50 s | 51.43 s | **0.49 s** | **109× faster** than uncached forward |
| 250,000 (`2` classes) | **159.09 s** (`61.7 GiB` HBM) | Exceeds `96 GiB`/core | — | Full-attention inference scales to 250k rows without KV cache |

### C. Distributed Synthetic Pretraining (`16` TensorCores = `8` `v7x` Chips, `2x2x2` ICI AllReduce)

Measured on random synthetic tables (`bfloat16` mixed precision, `tpu_dist` ICI gradient AllReduce; [`results/multi_host/`](./results/multi_host/)) to benchmark step time and multi-host scaling:

| Sequence Rows (`Train + Test`) | Features | Global Batch (`per TensorCore`) | Step `p50` | Aggregate Throughput | Per-Chip Throughput (`8` Chips) | Peak HBM / Core |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| `1,024` (`768 + 256`) | 32 | 128 (`8`/core) | **669.4 ms** | **189.5 tables/s** (`194,013 rows/s`) | 23.7 tables/s/chip | 94.7 GiB |
| `2,048` (`1,536 + 512`) | 64 | 64 (`4`/core) | **764.3 ms** | **83.4 tables/s** (`170,733 rows/s`) | 10.4 tables/s/chip | 38.9 GiB |
| `4,096` (`3,072 + 1,024`) | 100 | 32 (`2`/core) | **857.0 ms** | **37.2 tables/s** (`152,365 rows/s`) | 4.65 tables/s/chip | 51.8 GiB |

---

## 3. Reproducing the Main Results

Build the container image from [`docker/Dockerfile`](./docker/Dockerfile) and apply [`results/tpu_patch.diff`](./results/tpu_patch.diff) once to `/workspace/tabicl`:

```bash
git -C /workspace/tabicl apply /path/to/results/tpu_patch.diff
```

### Accuracy Verification (15 OpenML Classification & Regression Datasets)
Scripts: [`scripts/build_openml_reference_bundle.py`](./scripts/build_openml_reference_bundle.py) and [`scripts/check_accuracy.py`](./scripts/check_accuracy.py):

```bash
python3 scripts/build_openml_reference_bundle.py --out-dir /tmp/tabicl_cpu_ref
python3 scripts/check_accuracy.py --bundle-path /tmp/tabicl_cpu_ref/ref_cpu_bundle.pt --out-dir /tmp/tabicl_accuracy
```

### Single-Host Inference Grid (`1` TensorCore & `8` TensorCores on One `v7x` Host)
Script: [`scripts/benchmark_single_host.py`](./scripts/benchmark_single_host.py) (deployable via [`k8s/single_host_worker_pod.yaml`](./k8s/single_host_worker_pod.yaml)):

```bash
python3 scripts/benchmark_single_host.py \
  --bundle-path /tmp/tabicl_cpu_ref/ref_cpu_bundle.pt \
  --out-csv /tmp/bench_single_host.csv \
  --out-summary /tmp/bench_single_host_summary.json
```

### Multi-Host `2x2x2` ICI Inference (`DP=16`) & Synthetic Pretraining (`DDP=16`)
Script: [`scripts/benchmark_multi_host.py`](./scripts/benchmark_multi_host.py) (deployable across 2 hosts via [`k8s/multi_host_v7x8_job.yaml`](./k8s/multi_host_v7x8_job.yaml)):

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${JOB_COMPLETION_INDEX}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:29500" \
  scripts/benchmark_multi_host.py \
  --output-dir /tmp/tabicl_multihost_results
```

---

## 4. Detailed Scaling Tables, Engineering Notes, and Raw Outputs

- **[`APPENDIX.md`](./APPENDIX.md):** Full 14-point `tpu:8` and `tpu:16` scaling tables, KV-cache build-plus-query breakdowns, TorchTPU porting details (`tpu_patch.diff` and two-sided TPU DMA synchronization barriers), precision accuracy-gate findings, and multi-host serving lessons.
- **[`results/`](./results/):** Compatibility patch (`tpu_patch.diff`) and raw CSV/JSON measurements in [`accuracy/`](./results/accuracy/), [`single_host/`](./results/single_host/), and [`multi_host/`](./results/multi_host/).
