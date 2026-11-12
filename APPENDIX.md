# Appendix: Full Scaling Tables, TorchTPU Porting Details, and Analysis (TabICLv2 on TPU v7x)

This document preserves all detailed scaling sweeps, KV-cache prefill-plus-query measurements, TorchTPU porting fixes, accuracy-gate rejections, and multi-host findings from our TPU v7x (Ironwood) evaluation. For headline comparisons and reproduction commands, see [`README.md`](./README.md).

---

## 1. TorchTPU Compatibility Changes & Hardware Guards

Upstream TabICLv2 assumes CUDA (`torch.cuda` checks, FlashAttention-3, `float16` AMP) and uses dynamic Python/host control flow inside its forward pass. Applying [`results/tpu_patch.diff`](./results/tpu_patch.diff) alongside [`scripts/tabicl_tpu_bucketed.py`](./scripts/tabicl_tpu_bucketed.py) and [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) addresses four issues on TPU v7x:

1. **Long-context Pallas Splash Attention (`T_kv ≥ 16,384`):** Replaces CUDA-only FlashAttention-3 with custom TorchTPU Pallas Splash Attention ops (`tabicl::tpu_splash_sdpa_seg` and `tabicl::tpu_splash_sdpa_full`) for long in-context sequences while keeping standard XLA SDPA for shorter sequences.
2. **Graph-pure forward pass & power-of-two shape bucketing:** Removes `.item()`, `torch.unique`, `nonzero`, and data-dependent branches across `embedding.py`, `interaction.py`, `layers.py`, `learning.py`, `rope.py`, `tabicl.py`, `inference.py`, `ssmax.py`, and `quantile_dist.py`. [`scripts/tabicl_tpu_bucketed.py`](./scripts/tabicl_tpu_bucketed.py) pads row counts to powers of two (`50,000 → 65,536`, up to `512K`) and feature counts to `{16, 32, 64, 128, 256, 512}` (`100 → 128`) so `torch.compile(backend="tpu", fullgraph=True)` reuses compiled XLA executables.
3. **Two-sided TPU DMA barrier around dynamic boolean indexing:** Under 16-process `tpu_dist` (`world_size=16`) with large (`≥50k × 100`) preprocessing pipelines, asynchronous D2H/H2D transfers on boolean feature masks (`X[:, valid_feature_mask]`) triggered `RuntimeError: TFRT_SyncTpuTransferToDeviceAsync failed: INTERNAL: Core halted unexpectedly`. Bracketing dynamic mask extraction and device transfers with explicit `torch.tpu.synchronize()` barriers eliminated core halts (`0` halts across all 155 benchmark rows).
4. **Single-host metadata proxy on multi-host slices ([`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py)):** Intercepts GCE TPU topology metadata via an `LD_PRELOAD` socket hook so a single `v7x` host (`4` dual-core chips = `8` TensorCores) on a `2x2x2` nodepool can run standalone 1-device or 8-device workloads while forwarding Workload Identity OAuth requests transparently.

---

## 2. Numerical Accuracy Verification & Precision Selection

Evaluated against CPU `float32` reference predictions across 15 OpenML datasets (10 classification, 5 regression; [`results/accuracy/`](./results/accuracy/)):

| Configuration | Max `|ΔAUC|` (10 Classification Datasets) | Max Relative RMSE Diff (5 Regression Datasets) | Status |
| :--- | ---: | ---: | :--- |
| `fp32` (no KV cache) | 0.000244 | 0.0001% | Pass |
| `fp32` (`fp32` KV cache) | 0.000365 | 0.0001% | Pass |
| `bf16` mixed precision (no KV cache) | 0.001971 | 0.0757% | Pass |
| `bf16` mixed precision (`bf16` KV cache) | ≤ 0.0020 | ≤ 0.08% | Pass |
| Rejected: single-pass `bf16` on row interaction stage | **0.0277** (on `kc1`, vs. `0.005` gate) | — | **Failed accuracy gate** |

**Why `v7x` `bf16` runs are slower than optimal:** Keeping the column Set Transformer in `fp32` (`matmul_precision="highest"`) and the row stage in 3-pass `bf16` (`"high"`), combined with power-of-two bucket padding (`50k × 100 → 65,536 × 128`) and standard XLA head padding on `head_dim=16`, accounts for the `21.13 s` single-TensorCore forward latency. Subsequent work in [`tabiclv2-torchtpu-v5p-benchmark`](https://github.com/ankdeshm/tabiclv2-torchtpu-v5p-benchmark) showed that 3-pass `bf16` column matmuls, `4,096`-row / `4`-feature bucketing, and custom VMEM Pallas kernels for `head_dim=16` cut single-device forward time by more than `3×` while still passing all OpenML accuracy gates.

---

## 3. Full Single-Host 8-TensorCore (`4` Physical `v7x` Chips) Scaling Sweep

Complete 14-point `tpu:8` data-parallel sweep (`batch_datasets=8`, 1 dataset per TensorCore, warm `p50`; [`results/single_host/bench_single_host.csv`](./results/single_host/bench_single_host.csv)). The full 124-point single-TensorCore (`tpu:1`) grid across `bf16`/`fp32`, `10–500` features, `2–10` classes, and `1–32` estimators is in the same CSV.

| `n_train` | `n_test` | Features | Estimators | KV Cache | Warm `p50` (ms) | Warm `p99` (ms) | Host Datasets/s | Host Test Rows/s | Peak HBM / Core (GiB) |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1,000 | 10 | 8 | `False` | 96.79 | 99.10 | 82.53 | 82,528.0 | 3.56 |
| 1,000 | 1,000 | 10 | 8 | `True` (query) | 44.37 | 45.33 | 179.78 | 179,783.6 | 4.30 |
| 1,000 | 1,000 | 100 | 8 | `False` | 468.93 | 474.04 | 17.05 | 17,054.8 | 17.23 |
| 10,000 | 1,000 | 100 | 8 | `False` | 4,046.66 | 4,068.66 | 1.98 | 1,976.6 | 27.34 |
| 10,000 | 1,000 | 100 | 8 | `True` (query) | 221.78 | 226.26 | 36.01 | 36,009.0 | 35.19 |
| 45,000 | 5,000 | 100 | 8 | `False` | 21,112.87 | 21,128.73 | 0.38 | 1,894.4 | 53.58 |
| 45,000 | 5,000 | 100 | 8 | `True` (query) | 2,334.87 | 2,338.50 | 3.43 | 17,127.9 | 75.43 |
| 50,000 | 1,000 | 100 | 1 | `False` | 2,664.25 | 2,674.20 | 3.00 | 3,001.9 | 54.71 |
| 50,000 | 1,000 | 100 | 8 | `False` | 21,092.39 | 21,096.65 | 0.38 | 379.3 | 56.60 |
| 50,000 | 1,000 | 100 | 32 | `False` | 84,394.27 | 84,417.40 | 0.09 | 94.8 | 49.34 |
| 100,000 | 1,000 | 100 | 8 | `False` | 53,562.99 | 53,577.88 | 0.15 | 149.4 | 56.77 |
| 100,000 | 1,000 | 100 | 8 | `True` (query) | 489.55 | 490.15 | 16.34 | 16,340.7 | 94.48 |
| 250,000 | 1,000 | 100 | 8 | `False` | 159,091.72 | 159,108.27 | 0.05 | 50.3 | 61.67 |
| 250,000 | 1,000 | 100 | 8 | `True` | — | — | — | — | 94.66 *(HBM guard: `48 GiB` ICL KV + scratch > `96 GiB`)* |

---

## 4. KV Cache Build (`Prefill`) + Query vs. One-Shot Analysis (`1` TensorCore)

Across all 24 measured shape pairs in [`results/single_host/kv_cache_end_to_end.csv`](./results/single_host/kv_cache_end_to_end.csv), building the KV cache plus running one query takes `0.80–1.19×` the latency of an uncached forward pass. Representative `100`-feature, `8`-estimator `bf16` rows:

| `n_train` | `n_test` | Cache Build (`prefill`, ms) | Query `p50` (ms) | Build + Query Sum (ms) | Uncached One-Shot `p50` (ms) | Query-Only Speedup vs. Uncached | `(Build + Query) ÷ Uncached` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1,000 | 241.33 | 190.19 | 431.52 | 452.29 | 2.38× | 0.95 |
| 10,000 | 1,000 | 3,924.79 | 216.27 | 4,141.06 | 3,952.60 | 18.28× | 1.05 |
| 50,000 | 1,000 | 20,628.47 | 347.28 | 20,975.75 | 21,145.73 | 60.89× | 0.99 |
| 100,000 | 1,000 | 51,427.52 | 490.58 | 51,918.10 | 53,502.15 | 109.06× | 0.97 |
| 50,000 | 10,000 | 20,558.53 | 4,558.79 | 25,117.32 | 21,101.73 | 4.63× | 1.19 |
| 100,000 | 10,000 | 51,409.80 | 6,208.55 | 57,618.35 | 53,538.59 | 8.62× | 1.08 |

---

## 5. Multi-Host `2x2x2` ICI (`16` TensorCores = `8` Physical Chips) Findings

Complete 14-point `DP=16` inference sweep ([`results/multi_host/bench_multi_host.csv`](./results/multi_host/bench_multi_host.csv)):

| `n_train` | `n_test` | Features | Estimators | KV Cache | Cold Compile (s) | Warm `p50` (ms) | Warm `p99` (ms) | Slice Datasets/s | Slice Test Rows/s | Peak HBM / Core (GiB) |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1,000 | 10 | 8 | `False` | 27.62 | 97.47 | 101.35 | 163.76 | 163,763.3 | 3.56 |
| 1,000 | 1,000 | 10 | 8 | `True` (query) | 41.50 | 45.34 | 47.35 | 352.32 | 352,316.1 | 4.30 |
| 1,000 | 1,000 | 100 | 8 | `False` | 24.92 | 468.31 | 474.84 | 34.11 | 34,107.3 | 17.23 |
| 10,000 | 1,000 | 100 | 8 | `False` | 26.56 | 4,029.00 | 4,039.78 | 3.97 | 3,970.5 | 28.09 |
| 10,000 | 1,000 | 100 | 8 | `True` (query) | 36.79 | 224.56 | 226.34 | 71.20 | 71,196.1 | 35.69 |
| 45,000 | 5,000 | 100 | 8 | `False` | 40.04 | 60,575.01 | 62,340.51 | 0.27 | 1,326.8 | 66.29 |
| 45,000 | 5,000 | 100 | 8 | `True` (query) | 83.70 | 8,445.06 | 8,782.33 | 1.91 | 9,537.4 | 68.48 |
| 50,000 | 1,000 | 100 | 1 | `False` | 7.37 | 7,526.05 | 7,674.13 | 2.15 | 2,146.5 | 66.85 |
| 50,000 | 1,000 | 100 | 8 | `False` | 0.00 | 57,663.21 | 59,938.50 | 0.28 | 278.3 | 66.85 |
| 50,000 | 1,000 | 100 | 32 | `False` | 0.00 | 239,140.23 | 243,111.31 | 0.07 | 66.9 | 66.85 |
| 100,000 | 1,000 | 100 | 8 | `False` | 193.02 | 134,275.36 | 136,164.70 | 0.12 | 118.9 | 89.32 |
| 100,000 | 1,000 | 100 | 8 | `True` (query) | 231.39 | 12,188.77 | 12,495.03 | 1.31 | 1,312.9 | 94.57 |
| 250,000 | 1,000 | 100 | 8 | `False` | 636.69 | 397,168.25 | 402,702.88 | 0.04 | 40.2 | 89.79 |
| 250,000 | 1,000 | 100 | 8 | `True` | — | — | — | — | — | >96.0 *(HBM guard)* |

### What Worked vs. What Did Not Work Across Hosts
- **Small and medium tables (`≤10k` rows) scaled linearly across `tpu_dist` (`2.0×` over 1 host):** At `1k × 10`, `1k × 100`, and `10k × 100`, single-graph execution fit in HBM without host staging, achieving `163.8`, `34.1`, and `3.97` datasets/s (`1.98–2.01×` single-host throughput).
- **Large tables (`≥50k` rows) under a global `tpu_dist` process group regressed `2.5–2.9×` per dataset:** For `≥50k` rows, the multi-host runner activated `multi_host_stage` (staging intermediate column/row tiles through host RAM) and synchronized all 16 ranks at `dist.barrier()`, increasing `45k + 5k × 100` latency from `21.11 s` (`tpu:8`) to `60.58 s` (`tpu:16`). For dataset-parallel inference on large tables, running independent per-host workers (via [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py)) avoids cross-host barriers and host-RAM staging.
