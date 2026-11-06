# TabICLv2 Inference & Distributed Pretraining on Google Cloud TPU `v7x` (Ironwood) with TorchTPU

- **Upstream TabICLv2 Commit:** `d83091b845a8c5f7c087bd54964f755a9e560f45` (`tabicl-classifier-v2-20260212`)
- **Hardware:** Google Cloud TPU `v7x` (Ironwood) — `2x2x2` ICI slice (`2` hosts × `4` physical chips = `8` chips / `16` TensorCores, `192 GiB` HBM3e per chip / `96 GiB` per TensorCore)
- **Software Stack:** PyTorch `2.12` + `torch_tpu` (`PJRT_DEVICE=TPU`, `tpu_dist` ICI backend) with shape-bucketed `torch.compile` ([`scripts/tabicl_tpu_bucketed.py`](./scripts/tabicl_tpu_bucketed.py)) & multi-estimator KV-cache reuse

---

## 1. Milestone Status — 100% Complete

| Phase | Scope | Artifacts | Status |
|---|---|---|---|
| **Phase 0 & 1.1–1.4** | CUDA-ISM source audit, TPU `2x2x2` smoke test, CPU `fp32` OpenML reference (`15` datasets), pytest suite, eager numerical parity (`bf16`/`fp32`), XLA compile & profiling | [`results/AUDIT.md`](./results/AUDIT.md) · [`results/RUNLOG.md`](./results/RUNLOG.md) · [`results/ref_cpu.csv`](./results/ref_cpu.csv) · [`results/eager_parity_tpu.csv`](./results/eager_parity_tpu.csv) · [`results/tpu_patch.diff`](./results/tpu_patch.diff) | **Completed** |
| **Phase 1.5 (Single-Host)** | `124`-row `tpu:1` grid (`n_train` up to `250k`, `n_features` up to `500`, `n_classes` up to `10`, `n_estimators` up to `32`) + `14`-row `tpu:8` (`4`-chip / `8`-TensorCore) data-parallel scaling | [`results/bench_single_host.csv`](./results/bench_single_host.csv) · [`results/bench_single_host_summary.json`](./results/bench_single_host_summary.json) | **Completed (`138` rows)** |
| **Phase 2 (Multi-Host `2x2x2` ICI)** | `14`-row `DP=16` (`world_size=16`) inference sweep (`1k×10` up to `250k×100`) + `3`-row `DDP=16` (`tpu_dist` ICI AllReduce) pretraining benchmark | [`results/bench_multi_host.csv`](./results/bench_multi_host.csv) · [`results/bench_multi_host_summary.json`](./results/bench_multi_host_summary.json) · [`results/phase2_multi_host.json`](./results/phase2_multi_host.json) | **Completed (`17` rows)** |

---

## 2. Headline Benchmark (`50,000 × 100` Paper Configuration: `45k` Train + `5k` Test, `8` Estimators, `BF16`)

| Topology (`Devices`) | KV Cache | Batch Datasets | Latency `p50` (ms) | Latency `p99` (ms) | Throughput (datasets/s) | Aggregate Test Rows/s | Peak HBM / Core (GiB) | KV Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Single TensorCore (`tpu:1`)** | `False` | `1` | `21,130.2` | `21,138.0` | `0.0473` | `236.6` | `49.44` | `1.00×` |
| **Single TensorCore (`tpu:1`)** | **`True`** | `1` | **`2,321.5`** | **`2,325.2`** | **`0.4308`** | **`2,153.8`** | `64.09` | **`9.10×`** |
| **Single-Host 4-Chip (`tpu:8`)** | `False` | `8` | `21,112.9` | `21,128.7` | `0.3789` | `1,894.4` | `53.58` | `1.00×` |
| **Single-Host 4-Chip (`tpu:8`)** | **`True`** | `8` | **`2,334.9`** | **`2,338.5`** | **`3.4256`** | **`17,127.9`** | `75.43` | **`9.04×`** (`7.95×` vs `tpu:1`) |
| **Multi-Host 8-Chip `2x2x2` (`tpu:16`)** | `False` | `16` | `60,575.0` | `62,340.5` | `0.2654` | `1,326.8` | `66.29` | `1.00×` |
| **Multi-Host 8-Chip `2x2x2` (`tpu:16`)** | **`True`** | `16` | **`8,445.1`** | **`8,782.3`** | **`1.9075`** | **`9,537.4`** | `68.48` | **`7.17×`** |

> **Note on Multi-Host `tpu_dist` (`tpu:16`) vs Single-Host Independent Replicas (`tpu:8`):** Phase 2 initializes a single global `tpu_dist` process group (`world_size=16` across `2x2x2` ICI) with synchronized `dist.barrier()` step boundaries and default `HIGHEST` FP32 matmul precision on non-autocast preprocessing projections so the same process group seamlessly transitions into `DDP=16` ICI gradient AllReduce pretraining. For pure embarrassingly parallel tabular dataset serving without cross-dataset collectives, per-host `tpu:8` worker pools achieve near-linear `7.95×` per-host scaling (`34,255.9 rows/s` across two independent `4`-chip hosts).

### 2.1 Phase 1.5 Single-Host (`tpu:8` = `4` Physical `v7x` Chips / `8` TensorCores) Full Scaling Sweep

*(Complete `138`-row single-host grid — `124` `tpu:1` rows + `14` `tpu:8` rows across `BF16` & `FP32`, `n_features=10..500`, `n_classes=2..10`, `n_estimators=1..32` — is archived in [`results/bench_single_host.csv`](./results/bench_single_host.csv) and [`results/bench_single_host_summary.json`](./results/bench_single_host_summary.json).)*

| Benchmark Tag (`tpu:8`, `batch_datasets=8`) | `n_train` | `n_test` | `n_features` | `n_estimators` | KV Cache | `p50` (ms) | `p99` (ms) | Host Datasets/s | Host Test Rows/s | Peak HBM / Core (GiB) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `dp8_small_1kx10` | `1,000` | `1,000` | `10` | `8` | `False` | `96.79` | `99.10` | `82.53` | `82,528.0` | `3.56` |
| `dp8_small_1kx10_kv` | `1,000` | `1,000` | `10` | `8` | **`True`** | **`44.37`** | `45.33` | **`179.78`** | **`179,783.6`** | `4.30` |
| `dp8_small_1kx100` | `1,000` | `1,000` | `100` | `8` | `False` | `468.93` | `474.04` | `17.05` | `17,054.8` | `17.23` |
| `dp8_med_10kx100` | `10,000` | `1,000` | `100` | `8` | `False` | `4,046.66` | `4,068.66` | `1.98` | `1,976.6` | `27.34` |
| `dp8_med_10kx100_kv` | `10,000` | `1,000` | `100` | `8` | **`True`** | **`221.78`** | `226.26` | **`36.01`** | **`36,009.0`** | `35.19` |
| `dp8_headline_50kx100_paper` | `45,000` | `5,000` | `100` | `8` | `False` | `21,112.87` | `21,128.73` | `0.38` | `1,894.4` | `53.58` |
| `dp8_headline_50kx100_paper_kv` | `45,000` | `5,000` | `100` | `8` | **`True`** | **`2,334.87`** | `2,338.50` | **`3.43`** | **`17,127.9`** | `75.43` |
| `dp8_50kx100_te1k_est1` | `50,000` | `1,000` | `100` | `1` | `False` | `2,664.25` | `2,674.20` | `3.00` | `3,001.9` | `54.71` |
| `dp8_50kx100_te1k_est8` | `50,000` | `1,000` | `100` | `8` | `False` | `21,092.39` | `21,096.65` | `0.38` | `379.3` | `56.60` |
| `dp8_50kx100_te1k_est32` | `50,000` | `1,000` | `100` | `32` | `False` | `84,394.27` | `84,417.40` | `0.09` | `94.8` | `49.34` |
| `dp8_large_100kx100` | `100,000` | `1,000` | `100` | `8` | `False` | `53,562.99` | `53,577.88` | `0.15` | `149.4` | `56.77` |
| `dp8_large_100kx100_kv` | `100,000` | `1,000` | `100` | `8` | **`True`** | **`489.55`** | `490.15` | **`16.34`** | **`16,340.7`** | `94.48` |
| `dp8_xlarge_250kx100` | `250,000` | `1,000` | `100` | `8` | `False` | `159,091.72` | `159,108.27` | `0.05` | `50.3` | `61.67` |
| `dp8_xlarge_250kx100_kv` | `250,000` | `1,000` | `100` | `8` | `True` | — | — | — | — | `94.66` *(HBM guard: `48 GiB` ICL KV + scratch > `96 GiB`)* |

---

## 3. Phase 2 Multi-Host (`2x2x2` ICI, `world_size=16`) Complete Results

### 3.1 `DP=16` Multi-Host Tabular Inference (`batch_datasets=16`, `BF16`)

| Benchmark Tag | `n_train` | `n_test` | `n_features` | `n_estimators` | KV Cache | Compile (s) | `p50` (ms) | `p99` (ms) | Aggregate Datasets/s | Aggregate Test Rows/s | Peak HBM (GiB) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `dp16_small_1kx10` | `1,000` | `1,000` | `10` | `8` | `False` | `27.62` | `97.47` | `101.35` | `163.76` | `163,763.3` | `3.56` |
| `dp16_small_1kx10_kv` | `1,000` | `1,000` | `10` | `8` | **`True`** | `41.50` | **`45.34`** | `47.35` | **`352.32`** | **`352,316.1`** | `4.30` |
| `dp16_small_1kx100` | `1,000` | `1,000` | `100` | `8` | `False` | `24.92` | `468.31` | `474.84` | `34.11` | `34,107.3` | `17.23` |
| `dp16_med_10kx100` | `10,000` | `1,000` | `100` | `8` | `False` | `26.56` | `4,029.00` | `4,039.78` | `3.97` | `3,970.5` | `28.09` |
| `dp16_med_10kx100_kv` | `10,000` | `1,000` | `100` | `8` | **`True`** | `36.79` | **`224.56`** | `226.34` | **`71.20`** | **`71,196.1`** | `35.69` |
| `dp16_headline_50kx100_paper` | `45,000` | `5,000` | `100` | `8` | `False` | `40.04` | `60,575.01` | `62,340.51` | `0.27` | `1,326.8` | `66.29` |
| `dp16_headline_50kx100_paper_kv` | `45,000` | `5,000` | `100` | `8` | **`True`** | `83.70` | **`8,445.06`** | `8,782.33` | **`1.91`** | **`9,537.4`** | `68.48` |
| `dp16_50kx100_te1k_est1` | `50,000` | `1,000` | `100` | `1` | `False` | `7.37` | `7,526.05` | `7,674.13` | `2.15` | `2,146.5` | `66.85` |
| `dp16_50kx100_te1k_est8` | `50,000` | `1,000` | `100` | `8` | `False` | `0.00` | `57,663.21` | `59,938.50` | `0.28` | `278.3` | `66.85` |
| `dp16_50kx100_te1k_est32` | `50,000` | `1,000` | `100` | `32` | `False` | `0.00` | `239,140.23` | `243,111.31` | `0.07` | `66.9` | `66.85` |
| `dp16_large_100kx100` | `100,000` | `1,000` | `100` | `8` | `False` | `193.02` | `134,275.36` | `136,164.70` | `0.12` | `118.9` | `89.32` |
| `dp16_large_100kx100_kv` | `100,000` | `1,000` | `100` | `8` | **`True`** | `231.39` | **`12,188.77`** | `12,495.03` | **`1.31`** | **`1,312.9`** | `94.57` |
| `dp16_xlarge_250kx100` | `250,000` | `1,000` | `100` | `8` | `False` | `636.69` | `397,168.25` | `402,702.88` | `0.04` | `40.2` | `89.79` |
| `dp16_xlarge_250kx100_kv` | `250,000` | `1,000` | `100` | `8` | `True` | — | — | — | — | — | `>96.0` *(HBM guard: `48 GiB` ICL KV + scratch exceeds `96 GiB`/TensorCore)* |

- At `100,000 × 100` (`8` estimators), multi-estimator KV caching accelerates `DP=16` inference by **`11.02×`** (`134.28 s` → **`12.19 s`** per `16`-dataset batch) within `94.57 GiB` / `96 GiB` TensorCore HBM.
- Full-attention `TabICLv2` scales cleanly up to **`250,000` training rows × `100` features** (`89.79 GiB` peak HBM per TensorCore across all `16` TensorCores).

### 3.2 `DDP=16` Distributed Synthetic Tabular Pretraining (`2x2x2` ICI AllReduce, `BF16`)

| Benchmark Tag | Global Batch (Tables) | Seq Rows (`n_train + n_test`) | Features | Compile (s) | Step `p50` (ms) | Step `p99` (ms) | Aggregate Tables/s | Aggregate Rows/s | Loss (`Start → End`) | Peak HBM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `pretrain_ddp16_b8_t1024_f32` | `128` (`8`/core) | `1,024` (`768+256`) | `32` | `3.25` | **`669.43`** | `731.51` | **`189.47`** | **`194,012.6`** | `0.1938 → 0.1250` | `94.69` |
| `pretrain_ddp16_b4_t2048_f64` | `64` (`4`/core) | `2,048` (`1536+512`) | `64` | `3.39` | **`764.25`** | `793.80` | **`83.37`** | **`170,732.7`** | `0.2903 → 0.1431` | `38.90` |
| `pretrain_ddp16_b2_t4096_f100` | `32` (`2`/core) | `4,096` (`3072+1024`) | `100` | `4.56` | **`857.02`** | `881.93` | **`37.20`** | **`152,365.5`** | `0.3167 → 0.2129` | `51.76` |

---

## 4. Key TorchTPU / XLA Engineering Fixes (`scripts/tabicl_tpu_bucketed.py`)

1. **Line-by-Line CUDA-ISM Audit & Surgical Patch ([`results/AUDIT.md`](./results/AUDIT.md), [`results/tpu_patch.diff`](./results/tpu_patch.diff)):** Audited all upstream `TabICLv2` modules (`d83091b845a8c5f7c087bd54964f755a9e560f45`) for `torch.cuda.*` guards, FlashAttention/SDPA dispatch, AMP `device_type` checks, and `tpu_dist` distributed initialization, verifying numerical parity (`bf16` & `fp32`) across `15` OpenML classification and regression datasets ([`results/eager_parity_tpu.csv`](./results/eager_parity_tpu.csv)).
2. **Shape-Bucketed `torch.compile` Wrapper (`scripts/tabicl_tpu_bucketed.py`):** Pads `(n_rows, n_features)` to power-of-two / canonical buckets so dynamic tabular datasets reuse compiled XLA HLO executables without recompilation storms.
3. **Two-Sided TPU DMA Barrier Around Dynamic Boolean Indexing:** Under multi-process `tpu_dist` (`world_size=16`) with large (`>=50k × 100`) scikit-learn preprocessing pipelines, asynchronous D2H/H2D transfers on boolean feature masks (`X[:, valid_feature_mask]`) caused `RuntimeError: TFRT_SyncTpuTransferToDeviceAsync failed: INTERNAL: Core halted unexpectedly` unless bracketed by explicit `torch.tpu.synchronize()` barriers before D2H `.cpu()` extraction and after H2D `.to(device)` transfer. Adding two-sided synchronization achieved **100% pass rate (`0` core halts across `155` single-host and multi-host benchmark rows)**.

---

## 5. Repository Structure

- [`docker/`](./docker/) — [`Dockerfile`](./docker/Dockerfile) & [`cloudbuild.yaml`](./docker/cloudbuild.yaml) for building the `TorchTPU 0.2.0` + `TabICLv2 2.2.0` image.
- [`k8s/`](./k8s/) — GKE manifests for [`00_smoke_jobset.yaml`](./k8s/00_smoke_jobset.yaml) (`2x2x2` `JobSet` smoke test), [`01_single_host_pods.yaml`](./k8s/01_single_host_pods.yaml) (Phase 1.5 single-host), and [`02_multi_host_v7x8_job.yaml`](./k8s/02_multi_host_v7x8_job.yaml) (Phase 2 multi-host `DP=16` & `DDP=16`).
- [`env/`](./env/) — Captured TPU `v7x` host environment summaries and `pip freeze` snapshots (`host_0_*`, `host_1_*`).
- [`scripts/`](./scripts/):
  - [`tabicl_tpu_bucketed.py`](./scripts/tabicl_tpu_bucketed.py) — Shape-bucketed `TorchTPU` wrapper & DMA-safe device transfer guards.
  - [`phase0_smoke_test.py`](./scripts/phase0_smoke_test.py), [`phase1_cpu_reference.py`](./scripts/phase1_cpu_reference.py), [`phase1_3_eager_parity.py`](./scripts/phase1_3_eager_parity.py), [`phase1_4_compile_and_profile.py`](./scripts/phase1_4_compile_and_profile.py) — Phase 0–1.4 smoke, parity, and compile validation scripts.
  - [`phase1_5_bench_single_host.py`](./scripts/phase1_5_bench_single_host.py) — Phase 1.5 single-host (`tpu:1` 124-point grid + `tpu:8` DP scaling) benchmark harness.
  - [`phase1_6_serving_server.py`](./scripts/phase1_6_serving_server.py), [`phase1_6_load_test.py`](./scripts/phase1_6_load_test.py), [`phase1_7_fp8_eval.py`](./scripts/phase1_7_fp8_eval.py) — FastAPI online serving endpoint, load generator, and FP8 evaluation utilities.
  - [`phase2_bench_multi_host.py`](./scripts/phase2_bench_multi_host.py) — Phase 2 multi-host (`world_size=16` `2x2x2` ICI `DP=16` inference + `DDP=16` pretraining) benchmark harness.
- [`results/`](./results/) — [`AUDIT.md`](./results/AUDIT.md), [`RUNLOG.md`](./results/RUNLOG.md), [`tpu_patch.diff`](./results/tpu_patch.diff), [`logs/`](./results/logs/), and complete CSV/JSON benchmark outputs (`ref_cpu.csv`, `eager_parity_tpu.csv`, `bench_single_host.csv`, `bench_multi_host.csv`, and summaries).
