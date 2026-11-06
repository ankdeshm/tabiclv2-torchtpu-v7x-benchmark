# TabICLv2 on Google Cloud TPU v7x (`torch_tpu`) — Running Execution Log

- **Run Directory (GCS)**: `gs://YOUR_GCS_BUCKET/tabicl-tpu/20261105-1120/`
  - Subfolders: `env/`, `logs/`, `profiles/`, `results/`, `ckpt/`
- **GKE Cluster**: `tpu-v7x-cluster` (`us-central1`, project `YOUR_GCP_PROJECT_ID`)
- **Target Node Pool**: `tpu-v7x-8-nodepool` (`tpu7x-standard-4t`, topology `2x2x2` = `8` TPU7x physical chips across `2` hosts in `us-central1-c`)
- **Framework Constraint**: Strictly **PyTorch (`2.15.0+cpu`) + TorchTPU (`torch_tpu==0.2.0`, `libtpu==0.0.50`)**. No `torch_xla`, no `torchax`, no `vllm-tpu` (`tpu-inference`), no JAX model/training framework code.
- **Pinned `soda-inria/tabicl` Version & Commit**:
  - PyPI package: `tabicl==2.2.0`
  - Git repository: `https://github.com/soda-inria/tabicl` @ `d83091b845a8c5f7c087bd54964f755a9e560f45`

---

## Phase 0: Cluster and Environment Check

### Step 0.1 — Cluster & Node Pool Inspection
- **Commands**:
  ```bash
  gcloud container node-pools list --cluster=tpu-v7x-cluster --location=us-central1 --project=YOUR_GCP_PROJECT_ID
  gcloud container node-pools describe tpu-v7x-8-nodepool --cluster=tpu-v7x-cluster --location=us-central1 --project=YOUR_GCP_PROJECT_ID
  kubectl get nodes -l cloud.google.com/gke-nodepool=tpu-v7x-8-nodepool -o custom-columns='NAME:.metadata.name,STATUS:.status.conditions[?(@.type=="Ready")].status,ZONE:.metadata.labels.topology\.kubernetes\.io/zone,ACCEL:.metadata.labels.cloud\.google\.com/tpu-v7x-worker-node,TOPO:.metadata.labels.cloud\.google\.com/tpu-v7x-worker-node,ALLOC_TPU:.status.allocatable.google\.com/tpu,INTERNAL_IP:.status.addresses[?(@.type=="InternalIP")].address'
  ```
- **Result**:
  - **Cluster**: `tpu-v7x-cluster` (`1.36.4-gke.2377000`, region `us-central1`)
  - **Node Pool**: `tpu-v7x-8-nodepool`
  - **Machine Type**: `tpu7x-standard-4t` (`224` vCPUs, `~963.6 GiB` allocatable RAM per host, `gvnic` enabled)
  - **Topology**: `2x2x2` (`placementPolicy: tpu7x-2x2x2-policy`, accelerator label `cloud.google.com/tpu-v7x-worker-node=tpu7x`)
  - **Zone**: `us-central1-c`
  - **Provisioning / Reservation Model**: `flexStart: true`, `queuedProvisioning: enabled: true`, `consumeReservationType: NO_RESERVATION` (`goog-gke-node-pool-provisioning-model: flex-start`)
  - **Taints**: `google.com/tpu=present:NoSchedule`, `cloud.google.com/gke-queued=true:NoSchedule`
  - **Nodes (2 Ready hosts)**:
    1. `tpu-v7x-worker-node` (`10.0.0.10`, physical `TPU_WORKER_ID=0`) — `Ready=True`, `allocatable.google.com/tpu: 4`
    2. `tpu-v7x-worker-node` (`10.0.0.10`, physical `TPU_WORKER_ID=1`) — `Ready=True`, `allocatable.google.com/tpu: 4`
- **Errors / Fixes**: None.

### Step 0.2 — GCS Bucket & Workload Identity Verification
- **Commands**:
  ```bash
  gcloud storage buckets describe gs://YOUR_GCS_BUCKET --format="yaml(name,location,locationType,iamConfiguration)"
  gcloud storage buckets get-iam-policy gs://YOUR_GCS_BUCKET
  gcloud container clusters describe tpu-v7x-cluster --location=us-central1 --project=YOUR_GCP_PROJECT_ID --format="yaml(workloadIdentityConfig,addonsConfig)"
  kubectl get serviceaccounts -n default -o yaml
  ```
- **Result**:
  - Bucket `gs://YOUR_GCS_BUCKET` exists in `US-CENTRAL1` (same region as cluster).
  - Workload Identity is enabled (`workloadPool: YOUR_GCP_PROJECT_ID.svc.id.goog`).
  - Kubernetes ServiceAccount `default` (`namespace: default`) is bound via `iam.gke.io/gcp-service-account: tpu-reader-sa@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com`, holding `roles/storage.objectAdmin` on `gs://YOUR_GCS_BUCKET`.
- **Errors / Fixes**: None.

### Step 0.3 — TorchTPU Container Build & Push
- **Commands**:
  ```bash
  gcloud builds submit --project=YOUR_GCP_PROJECT_ID --region=us-central1 --config=docker/cloudbuild.yaml .
  ```
- **Result**:
  - Built and pushed `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/torchtpu/tabicl-torchtpu:v0.2.0-tabicl2.2.0` (`sha256:54586df570a74b6342a1944a13919832245f00df14fcc24e39d4df94f561c3d3`, Cloud Build ID `89232e78-670f-410d-903f-0468b0ea8792`).
  - Pinned versions verified inside image: Python `3.12.16`, `torch==2.15.0+cpu`, `torch_tpu==0.2.0`, `libtpu==0.0.50`, `tabicl==2.2.0` + `/workspace/tabicl` @ `d83091b845a8c5f7c087bd54964f755a9e560f45`, `pandas==2.2.3`, `scikit-learn==1.6.1`, `openml==0.15.1`, `fastapi==0.115.11`, `uvicorn==0.34.0`, `pytest==8.3.5`, `torchao==0.17.0+cpu`, `xprof==2.24.2`.
- **Errors / Fixes**: None.

### Step 0.4 — 2-Pod `JobSet` Smoke Test on `tpu-v7x-8-nodepool`
- **Commands**:
  ```bash
  kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.1/manifests.yaml
  kubectl create configmap tabicl-phase0-scripts -n default --from-file=scripts/phase0_smoke_test.py --from-file=scripts/tpu_single_host_setup.py --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -f k8s/00_smoke_jobset.yaml
  kubectl delete jobset tabicl-phase0-smoke -n default --wait=true
  ```
- **Result**:
  - **Device Topology Discovery**: Each `tpu7x-standard-4t` host (`4` physical dual-core TPU7x chips) exposes `8` VFIO PCIe endpoints (`/dev/vfio/0..7`, `devices_per_physical_chip = 2`) and `torch.accelerator.device_count() == 8` (`8` addressable TensorCore devices per host; PJRT binds `1` TensorCore device per OS process, so full-host execution runs `8` processes per host via `torchrun --nproc_per_node=8` with `TORCH_TPU_TOPOLOGY=2,2,1,2`).
  - **Stage A — Single-Host `4096x4096` `bf16` Matmul (`torch.device("tpu")`)**:
    - Host 0 (`tpu-v7x-worker-node`, `10.0.0.10`): `single_host_4096_bf16_matmul_sum = 9837350.0`, `max_memory_allocated = 169,013,760 B` (`161.2 MiB`) — **PASS**.
    - Host 1 (`tpu-v7x-worker-node`, `10.0.0.10`): `single_host_4096_bf16_matmul_sum = 9749907.0`, `max_memory_allocated = 169,013,760 B` (`161.2 MiB`) — **PASS**.
  - **Stage B — Multi-Host `tpu_dist` `all_reduce` Across BOTH Hosts (`world_size=16` across `8` chips)**:
    - `torchrun --nnodes=2 --nproc_per_node=8` (`TORCH_TPU_TOPOLOGY=2,2,2,2`, `backend="tpu_dist"`): all `16` TensorCore ranks across both hosts ran `4096x4096` `bf16` matmul + `dist.all_reduce(SUM)` returning **`136.0 == 136.0`** (`16 * 17 / 2`) with Tier-2 cross-process compilation cache active (`num_compilations=0` on 15/16 ranks) — **PASS**.
  - **Artifacts Saved**: `env/host_{0,1}_pip_freeze.txt`, `env/host_{0,1}_env_summary.json`, `logs/phase0_smoke_host_{0,1}.log` locally and in `gs://YOUR_GCS_BUCKET/tabicl-tpu/20261105-1120/`.
- **Errors / Fixes**:
  - Attempt 1 failed in Stage A with `RuntimeError: device_count is internal to TorchTPU... Use torch.accelerator.device_count instead.` Fixed immediately by switching to `torch.accelerator.device_count()` / `torch.accelerator.max_memory_allocated()` and using `backend="tpu_dist"` with `TORCH_TPU_TOPOLOGY=2,2,2,2` and `TORCH_TPU_SLICEBUILDER_ADDRESSES`. Attempt 2 succeeded cleanly in 45s.

---

## Phase 1: Serving & Benchmarking TabICLv2 on TorchTPU (Path A)

### Step 1.1 — CPU `fp32` Reference Baseline (`10` Classification + `5` Regression Datasets)
- **Command**:
  ```bash
  kubectl exec tabicl-host0 -- python3 /workspace/scripts/phase1_1_cpu_ref.py --bucket YOUR_GCS_BUCKET --run-prefix tabicl-tpu/20261105-1120
  ```
- **Result**:
  - Evaluated `TabICLClassifier(n_estimators=8, device="cpu", use_amp=False, random_state=42)` (`tabicl-classifier-v2-20260212.ckpt`) on 10 OpenML/TabArena classification datasets (`credit-g`, `diabetes`, `blood-transfusion-service-center`, `phoneme`, `kc1`, `qsar-biodeg`, `churn`, `segment`, `splice`, `vehicle`; `2`–`7` classes, `598`–`3,276` train rows, `4`–`60` features) and `TabICLRegressor(n_estimators=8, device="cpu", use_amp=False, random_state=42)` (`tabicl-regressor-v2-20260212.ckpt`) on 5 OpenML regression datasets (`cpu_small`, `kin8nm`, `wine_quality`, `abalone`, `concrete_compressive_strength`; `824`–`3,276` train rows, `8`–`12` features).
  - Pre-split data cleanly (`seed=42`, stratified 80/20 for classification, 80/20 for regression) with zero train/test preprocessing leakage (`fit_transform` inside TabICL runs strictly on `X_train`, `transform` on `X_test`).
  - Saved `results/ref_cpu.csv` (41 metric rows in canonical 22-column schema) and `results/ref_cpu_bundle.pt` (exact split arrays + reference probability/prediction tensors) locally and to `gs://YOUR_GCS_BUCKET/tabicl-tpu/20261105-1120/results/`.
- **Errors / Fixes**: None.

### Step 1.2 — CUDA-ism & TorchTPU Compatibility Audit + Minimal Surgical Patch (`tpu_patch.diff`)
- **Commands**:
  ```bash
  git -C third_party/tabicl diff > results/tpu_patch.diff
  kubectl exec tabicl-host1 -- pytest /workspace/tabicl/tests --ignore=/workspace/tabicl/tests/test_datetime_encoder.py --ignore=/workspace/tabicl/tests/test_finetune.py --ignore=/workspace/tabicl/tests/test_finetune_Friendly.py --ignore=/workspace/tabicl/tests/test_finetune_regression.py --ignore=/workspace/tabicl/tests/test_shap_categorical.py -v
  ```
- **Artifacts**:
  - `results/AUDIT.md` (769 lines, uploaded to `gs://YOUR_GCS_BUCKET/tabicl-tpu/20261105-1120/results/AUDIT.md`).
  - `results/tpu_patch.diff` (2,876 lines across 15 files in `third_party/tabicl`, uploaded to `gs://YOUR_GCS_BUCKET/tabicl-tpu/20261105-1120/results/tpu_patch.diff`).
  - `logs/phase1_2_pytest.log` (uploaded to `gs://YOUR_GCS_BUCKET/tabicl-tpu/20261105-1120/logs/phase1_2_pytest.log`).
- **Key Findings & Surgical Changes**:
  - Scanned all `91` Python files in `src/tabicl/` (`443` grep hits across 17 patterns + AST scan). Zero `.cuda(`, `xformers`, `triton`, or `masked_select` calls.
  - Registered `"tpu"` (`torch_tpu`) in `_torch_devices.py`, updated `InferenceConfig`/`InferenceManager` HBM accounting on TPU (avoiding erroneous `0.0` available-memory fallback to `batch_size=1` + `pin_memory=True` CPU offload), enforced `torch.bfloat16` autocast (`fp16` strictly forbidden on TPU), added `SafeLayerNorm` (`layers.py`) and `raw_quantiles.float()` (`tabicl.py`) to keep LayerNorm/Quantile stats in `fp32` under `bf16` autocast, guarded `RotaryEmbedding` (`rope.py`) in `fp32` without `torch.compile` autocast graph breaks, eliminated `.item()` / `torch.unique()` / `skip_mask.any()` host syncs, added masked fixed-shape bucketing (`train_size_override`, `real_features`, `valid_kv_len`), and added rectangular TorchTPU Pallas Splash Attention (`_tpu_pallas_splash_sdpa`) for `T_kv >= 16,384`.
  - **Upstream Test Suite Verification**: `211 passed, 386 skipped, 0 failed in 125.77s` with live TPU7x hardware active (ignoring only the 5 optional extra-dependency test files requiring uninstalled `statsmodels`/`transformers`/`shap`).

### Step 1.3 — Eager Bring-Up & Numerical Parity on `torch.device("tpu:0")` (`fp32` & `bf16`, `kv_cache=False` & `True`)
- **Command**:
  ```bash
  kubectl exec tabicl-host0 -- python3 /workspace/scripts/phase1_3_eager_parity.py --bucket YOUR_GCS_BUCKET --run-prefix tabicl-tpu/20261105-1120
  ```
- **Stage-Precision Discovery & Fix**:
  - `ColEmbedding` projects raw scalar continuous features (`d=128`) where 7-bit `bf16` mantissas collapse closely spaced normalized values; `RowInteraction` aggregates across feature groups into 4 CLS tokens (`d=512`).
  - Setting `COL_CONFIG(use_amp=False, matmul_precision="highest")` (`fp32` IEEE), `ROW_CONFIG(use_amp=False, matmul_precision="high")` (`bf16x3` MXU systolic matmuls + `fp32` residual stream), and `ICL_CONFIG(use_amp=True)` (full `bfloat16` autocast across all 12 `d=512` quadratic ICL transformer blocks) passed **all 75/75 dataset-configuration pairs** (`all_gates_passed=True`)! (By contrast, single-pass `row_prec="medium"` passed 14/15 datasets but drifted on `kc1` to `|dAUC|=0.0277`; upgrading `ROW_CONFIG` to `"high"` reduced `kc1` `|dAUC|` to `0.00088` with `bf16` MXU throughput.)
- **Results Across All 15 Datasets (`results/eager_parity_tpu.csv`, `logs/phase1_3_eager_parity.log`)**:
  | Config | `use_amp` | `kv_cache` | KV Dtype | Max `|AUC_tpu - AUC_cpu|` (Gate) | Max `|RMSE_tpu/RMSE_cpu - 1|` (Gate) | Total KV Cache (15 DS) | Gate |
  |---|---|---|---|---|---|---|---|
  | `fp32_nocache` | `False` | `False` | `none` | `0.000244` (`<= 0.002`) | `0.0001%` (`<= 1.0%`) | `0.00 MiB` | **15/15 PASS** |
  | `fp32_kv_fp32` | `False` | `True` | `float32` | `0.000365` (`<= 0.002`) | `0.0001%` (`<= 1.0%`) | `12,247.12 MiB` | **15/15 PASS** |
  | `bf16_nocache` | `True` (`bf16`) | `False` | `none` | `0.001971` (`<= 0.005`) | `0.0757%` (`<= 1.0%`) | `0.00 MiB` | **15/15 PASS** |
  | `bf16_kv_bf16` | `True` (`bf16`) | `True` | `bfloat16` | `0.001706` (`<= 0.005`) | `0.1210%` (`<= 1.0%`) | `6,123.56 MiB` | **15/15 PASS** |
  | `bf16_kv_fp32` | `True` (`bf16`) | `True` | `float32` | `0.001190` (`<= 0.005`) | `0.1214%` (`<= 1.0%`) | `12,247.12 MiB` | **15/15 PASS** |
- **KV-Cache Dtype Trade-off**: `bfloat16` KV caching (`bf16_kv_bf16`) cuts KV-cache HBM footprint by **exact 50.0%** (`6,123.56 MiB` vs `12,247.12 MiB` summed across the 15 datasets, e.g. `190.5 MiB` vs `380.9 MiB` on `credit-g` and `642.8 MiB` vs `1,285.6 MiB` on `phoneme`) while preserving ROC-AUC within `<= 0.00171` and RMSE within `<= 0.121%` of CPU `fp32`.
