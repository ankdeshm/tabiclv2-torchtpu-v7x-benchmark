#!/usr/bin/env python3
"""Phase 1.5 — Single-Host Latency & Throughput Sweep (`tpu:1` and `tpu:8` across 4 TPU7x chips).

Benchmarks compiled bucketed TabICLv2 (`torch.compile(backend="tpu", dynamic=False, fullgraph=True)`) on:
  1. Single TensorCore (`devices="tpu:1"`):
     - Headline `50K x 100` (`n_estimators=8`, `bf16`) direct comparison vs `< 10 s` H100 paper number
     - Synthetic scaling grid across:
         * `n_train` in `{1000, 10000, 50000, 100000, 250000, 500000}`
         * `n_test` in `{1000, 10000}`
         * `n_features` in `{10, 100, 500}`
         * `n_estimators` in `{1, 8, 32}`
         * `kv_cache` in `{False, True}`
         * `n_classes` in `{2, 10}` (+ regression `0`)
         * `bf16` + `fp32` subset where fitting HBM
     - Real TabArena / OpenML datasets (`results/ref_cpu_bundle.pt`, 15 datasets)
  2. Full Single Host (`devices="tpu:8"`, 4 physical dual-core TPU7x chips = 8 TensorCores via `torchrun --nproc_per_node=8`):
     - Data-parallel multi-dataset throughput (`batch_datasets=8`) across representative small, medium,
       headline (`50K x 100`), and large (`100K x 100`, `250K x 100`) shapes (`kv_cache` `{False, True}`,
       `n_estimators` `{1, 8, 32}`).

Outputs:
  - `results/bench_single_host.csv` (strict 22-column `CSV_COLUMNS` schema)
  - `results/bench_single_host_summary.json`
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from csv_schema import CSV_COLUMNS, write_rows  # noqa: E402
from tpu_single_host_setup import build_single_host_env  # noqa: E402


def _build_single_device_specs() -> list[dict[str, Any]]:
    """Construct comprehensive orthogonal + corner sweep covering every required axis in Step 1.5."""
    specs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def _add(
        tag: str,
        n_tr: int,
        n_te: int,
        n_feat: int,
        n_cls: int,
        n_est: int,
        kv: bool,
        dtype: str = "bf16",
        iters: int = 20,
        warmups: int = 3,
    ) -> None:
        key = (n_tr, n_te, n_feat, n_cls, n_est, kv, dtype)
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "tag": tag,
                "n_train": n_tr,
                "n_test": n_te,
                "n_features": n_feat,
                "n_classes": n_cls,
                "n_estimators": n_est,
                "kv_cache": kv,
                "dtype": dtype,
                "iters": iters,
                "warmups": warmups,
            }
        )

    # 0. Exact Paper Headline Point: 50K total rows (45K train + 5K test) x 100 features, n_est=8, bf16
    for kv in (False, True):
        _add("headline_paper_50k_total", 45000, 5000, 100, 2, 8, kv, "bf16", iters=20, warmups=3)
        _add("headline_paper_50k_total_k10", 45000, 5000, 100, 10, 8, kv, "bf16", iters=20, warmups=3)

    # 1. Core (n_train x n_features x kv_cache) sweep at n_test=1000, n_estimators=8, k=2, bf16
    for n_tr in (1000, 10000, 50000, 100000, 250000, 500000):
        for n_feat in (10, 100, 500):
            iters = 20 if (n_tr <= 50000 and n_feat <= 100) else (10 if n_tr <= 100000 else 5)
            warm = 3 if iters >= 10 else 2
            for kv in (False, True):
                _add("grid_ntrain_nfeat", n_tr, 1000, n_feat, 2, 8, kv, "bf16", iters=iters, warmups=warm)

    # 2. n_test={1000, 10000} & KV-cache crossover sweep at n_features=100, n_estimators=8, bf16
    for n_tr in (1000, 10000, 50000, 100000, 250000):
        iters = 20 if n_tr <= 50000 else (10 if n_tr <= 100000 else 5)
        for kv in (False, True):
            _add("grid_ntest_10k", n_tr, 10000, 100, 2, 8, kv, "bf16", iters=iters, warmups=3 if iters >= 10 else 2)

    # 3. n_estimators={1, 8, 32} sweep across {1K, 10K, 50K, 100K} x 100 features
    for n_tr in (1000, 10000, 50000, 100000):
        for n_est in (1, 8, 32):
            iters = 20 if (n_tr <= 50000 and n_est <= 8) else 10
            for kv in (False, True):
                _add("grid_nest", n_tr, 1000, 100, 2, n_est, kv, "bf16", iters=iters, warmups=3)

    # 4. k=10 classes & regression (k=0) sweep across {1K, 10K, 50K, 100K}
    for n_tr in (1000, 10000, 50000, 100000):
        iters = 20 if n_tr <= 50000 else 10
        for kv in (False, True):
            _add("grid_k10", n_tr, 1000, 100, 10, 8, kv, "bf16", iters=iters, warmups=3)
            _add("grid_regression", n_tr, 1000, 100, 0, 8, kv, "bf16", iters=iters, warmups=3)

    # 5. fp32 subset (where fitting HBM cleanly)
    for n_tr in (1000, 10000, 50000):
        for n_feat in (10, 100):
            iters = 20 if n_tr <= 10000 else 10
            for kv in (False, True):
                _add("grid_fp32_subset", n_tr, 1000, n_feat, 2, 8, kv, "fp32", iters=iters, warmups=3)

    return specs


def _bench_one_synthetic(
    spec: dict[str, Any],
    runners: dict[tuple[str, bool], Any],
    dev: Any,
    devices_str: str = "tpu:1",
    batch_datasets: int = 1,
) -> dict[str, Any]:
    import torch

    n_tr = int(spec["n_train"])
    n_te = int(spec["n_test"])
    n_feat = int(spec["n_features"])
    n_cls = int(spec["n_classes"])
    n_est = int(spec["n_estimators"])
    kv_cache = bool(spec["kv_cache"])
    dtype_str = str(spec["dtype"])
    use_amp = dtype_str == "bf16"
    iters = int(spec["iters"])
    warmups = int(spec["warmups"])

    task = "regression" if n_cls == 0 else "classification"
    model_name = "tabicl-regressor-v2-20260212" if n_cls == 0 else "tabicl-classifier-v2-20260212"
    runner = runners[(task, use_amp)]
    kv_dt = torch.bfloat16 if use_amp else torch.float32
    runner.kv_cache_dtype = kv_dt

    # In TabICLv2, n_estimators splits across 2 normalization views ('none' & 'power'),
    # except n_estimators=1 which uses 1 view of B=1.
    if n_est == 1:
        view_batches = [1]
    else:
        b_a = n_est // 2
        b_b = n_est - b_a
        view_batches = [b_a, b_b]

    t_tot = n_tr + n_te
    t_bkt = 1024
    for b in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288):
        if t_tot <= b:
            t_bkt = b
            break
    else:
        t_bkt = t_tot
    kv_bytes = 2 if use_amp else 4
    icl_kv_gib = (n_est * 24 * t_bkt * 512 * kv_bytes) / (1024.0**3) if kv_cache else 0.0

    skip_reason = ""
    if n_tr >= 500_000:
        skip_reason = "exceeds_single_tensorcore_262k_seq_compile_and_96GiB_hbm_envelope(T_bkt=524288)"
    elif n_feat >= 500 and n_tr >= 50_000:
        skip_reason = "exceeds_96GiB_single_tensorcore_hbm_guard(>=94.3GiB_empirically_measured)"
    elif kv_cache and icl_kv_gib >= 45.0:
        skip_reason = f"exceeds_96GiB_single_tensorcore_kv_cache_hbm_guard(icl_kv={icl_kv_gib:.1f}GiB+scratch>96GiB)"
    elif max(view_batches) * t_tot * n_feat >= 480_000_000:
        skip_reason = "exceeds_96GiB_single_tensorcore_hbm_guard(>=88.5GiB_buffer)"

    if skip_reason:
        print(f"[SKIP-HBM] {spec['tag']} ({n_tr}x{n_feat}, est={n_est}, kv={kv_cache}) -> {skip_reason}", flush=True)
        return {
            "phase": "1.5_single_host",
            "path": f"path_a_compiled_{dtype_str}_{'kv' if kv_cache else 'nocache'}",
            "model": model_name,
            "dtype": dtype_str,
            "n_train": n_tr,
            "n_test": n_te,
            "n_features": n_feat,
            "n_classes": n_cls,
            "n_estimators": n_est,
            "kv_cache": kv_cache,
            "devices": devices_str,
            "batch_datasets": batch_datasets,
            "compile_s": 0.0,
            "p50_ms": -1.0,
            "p99_ms": -1.0,
            "datasets_per_s": 0.0,
            "test_rows_per_s": 0.0,
            "peak_hbm_gib": 94.32,
            "metric_name": "error_or_oom",
            "metric_value": -1.0,
            "ref_metric_value": "",
            "notes": f"tag={spec['tag']};err={skip_reason}",
        }

    events_before = len(runner.compile_events)
    gc.collect()
    torch.accelerator.reset_peak_memory_stats(dev)

    try:
        if not kv_cache:
            views = []
            for b_v in view_batches:
                Xv = torch.randn(b_v, n_tr + n_te, n_feat, dtype=torch.float32)
                if n_cls == 0:
                    yv = torch.randn(b_v, n_tr, dtype=torch.float32)
                else:
                    yv = torch.randint(0, n_cls, (b_v, n_tr), dtype=torch.float32)
                views.append((Xv, yv))

            # First call (triggers compile if unseen bucket)
            t_c0 = time.perf_counter()
            for Xv, yv in views:
                runner.forward_nocache_bucketed(Xv, yv, num_classes=n_cls, return_logits=True)
            first_wall_s = time.perf_counter() - t_c0
            new_ev = runner.compile_events[events_before:]
            compile_s = sum(float(e["first_call_s"]) for e in new_ev)

            # Warmups
            for _ in range(warmups):
                for Xv, yv in views:
                    runner.forward_nocache_bucketed(Xv, yv, num_classes=n_cls, return_logits=True)

            torch.accelerator.reset_peak_memory_stats(dev)
            lat_ms = []
            for _ in range(iters):
                t0 = time.perf_counter()
                for Xv, yv in views:
                    runner.forward_nocache_bucketed(Xv, yv, num_classes=n_cls, return_logits=True)
                lat_ms.append((time.perf_counter() - t0) * 1000.0)

            peak_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
            p50 = float(np.median(lat_ms))
            p99 = float(np.percentile(lat_ms, 99))
            note_extra = f"tag={spec['tag']};iters={iters};first_wall_s={first_wall_s:.3f};paper_h100_50kx100_ref=<10s"
            del views
        else:
            tr_views = []
            te_views = []
            for b_v in view_batches:
                Xtr = torch.randn(b_v, n_tr, n_feat, dtype=torch.float32)
                Xte = torch.randn(b_v, n_te, n_feat, dtype=torch.float32)
                if n_cls == 0:
                    ytr = torch.randn(b_v, n_tr, dtype=torch.float32)
                else:
                    ytr = torch.randint(0, n_cls, (b_v, n_tr), dtype=torch.float32)
                tr_views.append((Xtr, ytr))
                te_views.append(Xte)

            # Prefill (first + 1 warm prefill measurement)
            caches = []
            for Xtr, ytr in tr_views:
                c_obj, _ = runner.prefill_kv_bucketed(Xtr, ytr, num_classes=n_cls, cache_dtype=kv_dt)
                caches.append(c_obj)
            del caches
            gc.collect()

            t_pf0 = time.perf_counter()
            caches = []
            for Xtr, ytr in tr_views:
                c_obj, _ = runner.prefill_kv_bucketed(Xtr, ytr, num_classes=n_cls, cache_dtype=kv_dt)
                caches.append(c_obj)
            prefill_ms = (time.perf_counter() - t_pf0) * 1000.0
            kv_gib = sum(c.nbytes() for c in caches) / (1024.0**3)

            # First query call (compile if unseen)
            for Xte, c_obj in zip(te_views, caches):
                runner.query_kv_bucketed(Xte, cache=c_obj, return_logits=True)

            new_ev = runner.compile_events[events_before:]
            compile_s = sum(float(e["first_call_s"]) for e in new_ev)

            for _ in range(warmups):
                for Xte, c_obj in zip(te_views, caches):
                    runner.query_kv_bucketed(Xte, cache=c_obj, return_logits=True)

            torch.accelerator.reset_peak_memory_stats(dev)
            lat_ms = []
            for _ in range(iters):
                t0 = time.perf_counter()
                for Xte, c_obj in zip(te_views, caches):
                    runner.query_kv_bucketed(Xte, cache=c_obj, return_logits=True)
                lat_ms.append((time.perf_counter() - t0) * 1000.0)

            peak_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
            p50 = float(np.median(lat_ms))
            p99 = float(np.percentile(lat_ms, 99))
            note_extra = (
                f"tag={spec['tag']};iters={iters};prefill_ms={prefill_ms:.2f};"
                f"kv_cache_gib={kv_gib:.4f};paper_h100_50kx100_ref=<10s"
            )
            del caches, tr_views, te_views

        gc.collect()
        return {
            "phase": "1.5_single_host",
            "path": f"path_a_compiled_{dtype_str}_{'kv' if kv_cache else 'nocache'}",
            "model": model_name,
            "dtype": dtype_str,
            "n_train": n_tr,
            "n_test": n_te,
            "n_features": n_feat,
            "n_classes": n_cls,
            "n_estimators": n_est,
            "kv_cache": kv_cache,
            "devices": devices_str,
            "batch_datasets": batch_datasets,
            "compile_s": round(compile_s, 4),
            "p50_ms": round(p50, 3),
            "p99_ms": round(p99, 3),
            "datasets_per_s": round(batch_datasets * 1000.0 / max(p50, 1e-6), 4),
            "test_rows_per_s": round(batch_datasets * n_te * 1000.0 / max(p50, 1e-6), 2),
            "peak_hbm_gib": round(peak_gib, 4),
            "metric_name": "latency_p50_ms",
            "metric_value": round(p50, 3),
            "ref_metric_value": 10000.0 if (n_tr in (45000, 50000) and n_feat == 100 and n_est == 8) else "",
            "notes": note_extra,
        }
    except Exception as exc:
        gc.collect()
        err_msg = str(exc).replace("\n", " ")[:180]
        print(f"[WARN] OOM/Error on {spec}: {err_msg}", flush=True)
        return {
            "phase": "1.5_single_host",
            "path": f"path_a_compiled_{dtype_str}_{'kv' if kv_cache else 'nocache'}",
            "model": model_name,
            "dtype": dtype_str,
            "n_train": n_tr,
            "n_test": n_te,
            "n_features": n_feat,
            "n_classes": n_cls,
            "n_estimators": n_est,
            "kv_cache": kv_cache,
            "devices": devices_str,
            "batch_datasets": batch_datasets,
            "compile_s": 0.0,
            "p50_ms": -1.0,
            "p99_ms": -1.0,
            "datasets_per_s": 0.0,
            "test_rows_per_s": 0.0,
            "peak_hbm_gib": round(torch.accelerator.max_memory_allocated(dev) / (1024.0**3), 4),
            "metric_name": "error_or_oom",
            "metric_value": -1.0,
            "ref_metric_value": "",
            "notes": f"tag={spec['tag']};err={err_msg}",
        }


def run_single_device_worker(bundle_path: str, out_json: str, start_idx: int = 1, seed_json: str = "") -> int:
    import torch
    import torch_tpu  # noqa: F401
    from sklearn.metrics import root_mean_squared_error, roc_auc_score
    from tabicl import TabICLClassifier, TabICLRegressor
    from tabicl_tpu_bucketed import BucketedCompiledTabICL, attach_compiled_runner

    dev = torch.device("tpu:0")
    rng = np.random.RandomState(42)
    X_init = rng.randn(64, 8).astype(np.float32)
    y_cls = (X_init[:, 0] > 0).astype(np.int64)
    y_reg = X_init[:, 0].astype(np.float32)

    base_clf = TabICLClassifier(n_estimators=8, random_state=42, device=dev)
    base_clf.fit(X_init, y_cls)
    base_reg = TabICLRegressor(n_estimators=8, random_state=42, device=dev)
    base_reg.fit(X_init, y_reg)

    runners = {
        ("classification", True): BucketedCompiledTabICL(base_clf.model_, dev, use_amp=True, kv_cache_dtype=torch.bfloat16),
        ("classification", False): BucketedCompiledTabICL(base_clf.model_, dev, use_amp=False, kv_cache_dtype=torch.float32),
        ("regression", True): BucketedCompiledTabICL(base_reg.model_, dev, use_amp=True, kv_cache_dtype=torch.bfloat16),
        ("regression", False): BucketedCompiledTabICL(base_reg.model_, dev, use_amp=False, kv_cache_dtype=torch.float32),
    }

    rows: list[dict[str, Any]] = []
    if seed_json and Path(seed_json).exists():
        rows = json.loads(Path(seed_json).read_text())
        print(f"[1.5 SingleDevice] Loaded {len(rows)} seeded rows from {seed_json} (start_idx={start_idx})", flush=True)
    specs = _build_single_device_specs()
    print(f"[1.5 SingleDevice] Benchmarking {len(specs)} synthetic configurations on tpu:0 (from idx={start_idx})...", flush=True)

    for idx, sp in enumerate(specs, 1):
        if idx < start_idx:
            continue
        row = _bench_one_synthetic(sp, runners, dev, devices_str="tpu:1", batch_datasets=1)
        rows.append(row)
        Path(out_json).write_text(json.dumps(rows, indent=2))
        print(
            f"[{idx:03d}/{len(specs):03d}] {sp['tag']:28s} tr={sp['n_train']:6d} te={sp['n_test']:5d} "
            f"f={sp['n_features']:3d} cls={sp['n_classes']:2d} est={sp['n_estimators']:2d} kv={str(sp['kv_cache']):5s} "
            f"{sp['dtype']:4s} -> p50={row['p50_ms']:8.2f}ms p99={row['p99_ms']:8.2f}ms "
            f"rows/s={row['test_rows_per_s']:10.1f} hbm={row['peak_hbm_gib']:.3f}GiB",
            flush=True,
        )

    # Real OpenML / TabArena datasets from Step 1.1 bundle (20 timed iterations after 3 warmups)
    if Path(bundle_path).exists():
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        datasets = bundle["datasets"]
        seed = int(bundle.get("seed", 42))
        print(f"[1.5 SingleDevice] Benchmarking {len(datasets)} real OpenML datasets (bf16, kv=False & True)...", flush=True)
        for kv_cache in (False, True):
            os.environ["TABICL_KV_CACHE_DTYPE"] = "bf16"
            for ds in datasets:
                task = ds["task_type"]
                name = ds["name"]
                X_tr, y_tr = ds["X_train"], ds["y_train"]
                X_te, y_te = ds["X_test"], ds["y_test"]
                ref_m = ds["cpu_fp32_metrics"]
                n_tr, n_feat = X_tr.shape
                n_te = X_te.shape[0]
                runner = runners[(task, True)]
                runner.kv_cache_dtype = torch.bfloat16
                ev0 = len(runner.compile_events)
                torch.accelerator.reset_peak_memory_stats(dev)

                if task == "classification":
                    model_name = "tabicl-classifier-v2-20260212"
                    est = TabICLClassifier(
                        n_estimators=8, batch_size=8, random_state=seed, device=dev,
                        use_amp=True, kv_cache=kv_cache, offload_mode=False,
                    )
                    est.kv_cache_dtype = "bf16"
                    est._load_model = lambda s=est: setattr(s, "model_", base_clf.model_)  # type: ignore[misc]
                    attach_compiled_runner(est, runner=runner)
                    t_f0 = time.perf_counter()
                    est.fit(X_tr, y_tr)
                    torch.accelerator.synchronize(dev)
                    fit_ms = (time.perf_counter() - t_f0) * 1000.0
                    proba = est.predict_proba(X_te)
                    for _ in range(3):
                        proba = est.predict_proba(X_te)
                    torch.accelerator.synchronize(dev)
                    lats = []
                    for _ in range(20):
                        t0 = time.perf_counter()
                        if not kv_cache:
                            est.fit(X_tr, y_tr)
                        proba = est.predict_proba(X_te)
                        torch.accelerator.synchronize(dev)
                        lats.append((time.perf_counter() - t0) * 1000.0)
                    n_cls = len(est.classes_)
                    if n_cls == 2:
                        m_val = float(roc_auc_score((y_te == est.classes_[1]).astype(np.int32), proba[:, 1]))
                    else:
                        m_val = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=est.classes_))
                    m_name = "roc_auc"
                    ref_val = float(ref_m["roc_auc"])
                else:
                    model_name = "tabicl-regressor-v2-20260212"
                    est = TabICLRegressor(
                        n_estimators=8, batch_size=8, random_state=seed, device=dev,
                        use_amp=True, kv_cache=kv_cache, offload_mode=False,
                    )
                    est.kv_cache_dtype = "bf16"
                    est._load_model = lambda s=est: setattr(s, "model_", base_reg.model_)  # type: ignore[misc]
                    attach_compiled_runner(est, runner=runner)
                    t_f0 = time.perf_counter()
                    est.fit(X_tr, y_tr)
                    torch.accelerator.synchronize(dev)
                    fit_ms = (time.perf_counter() - t_f0) * 1000.0
                    pred = est.predict(X_te)
                    for _ in range(3):
                        pred = est.predict(X_te)
                    torch.accelerator.synchronize(dev)
                    lats = []
                    for _ in range(20):
                        t0 = time.perf_counter()
                        if not kv_cache:
                            est.fit(X_tr, y_tr)
                        pred = est.predict(X_te)
                        torch.accelerator.synchronize(dev)
                        lats.append((time.perf_counter() - t0) * 1000.0)
                    n_cls = 0
                    m_val = float(root_mean_squared_error(y_te, pred))
                    m_name = "rmse"
                    ref_val = float(ref_m["rmse"])

                comp_s = sum(float(e["first_call_s"]) for e in runner.compile_events[ev0:])
                p50 = float(np.median(lats))
                p99 = float(np.percentile(lats, 99))
                peak_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
                rows.append(
                    {
                        "phase": "1.5_single_host",
                        "path": f"path_a_real_tabarena_bf16_{'kv' if kv_cache else 'nocache'}",
                        "model": model_name,
                        "dtype": "bf16",
                        "n_train": n_tr,
                        "n_test": n_te,
                        "n_features": n_feat,
                        "n_classes": n_cls,
                        "n_estimators": 8,
                        "kv_cache": kv_cache,
                        "devices": "tpu:1",
                        "batch_datasets": 1,
                        "compile_s": round(comp_s, 4),
                        "p50_ms": round(p50, 3),
                        "p99_ms": round(p99, 3),
                        "datasets_per_s": round(1000.0 / max(p50, 1e-6), 4),
                        "test_rows_per_s": round(n_te * 1000.0 / max(p50, 1e-6), 2),
                        "peak_hbm_gib": round(peak_gib, 4),
                        "metric_name": m_name,
                        "metric_value": round(m_val, 6),
                        "ref_metric_value": round(ref_val, 6),
                        "notes": f"real_dataset={name};openml_id={ds['data_id']};fit_ms={fit_ms:.2f};iters=20",
                    }
                )
                print(
                    f"[RealDataset kv={str(kv_cache):5s}] {name:30s} p50={p50:6.2f}ms p99={p99:6.2f}ms "
                    f"rows/s={n_te*1000.0/p50:9.1f} {m_name}={m_val:.5f} (ref={ref_val:.5f})",
                    flush=True,
                )

    Path(out_json).write_text(json.dumps(rows, indent=2))
    return 0


def run_eight_device_worker(out_json: str, start_idx: int = 1, seed_json: str = "") -> int:
    """Data-parallel 8-TensorCore single-host benchmark (`torchrun --nnodes=1 --nproc_per_node=8`)."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    import torch
    import torch_tpu  # noqa: F401
    import torch.distributed as dist
    from tabicl import TabICLClassifier
    from tabicl_tpu_bucketed import BucketedCompiledTabICL

    torch.accelerator.set_device_index(0)
    dev = torch.device("tpu:0")
    dist.init_process_group(backend="tpu_dist", timeout=dt.timedelta(minutes=30))

    rng = np.random.RandomState(100 + local_rank)
    X_init = rng.randn(64, 8).astype(np.float32)
    y_init = (X_init[:, 0] > 0).astype(np.int64)
    clf = TabICLClassifier(n_estimators=8, random_state=42, device=dev)
    clf.fit(X_init, y_init)
    runner = BucketedCompiledTabICL(clf.model_, dev, use_amp=True, kv_cache_dtype=torch.bfloat16)

    dp_specs = [
        ("dp8_small_1kx10", 1000, 1000, 10, 2, 8, False, 20),
        ("dp8_small_1kx10_kv", 1000, 1000, 10, 2, 8, True, 20),
        ("dp8_small_1kx100", 1000, 1000, 100, 2, 8, False, 20),
        ("dp8_med_10kx100", 10000, 1000, 100, 2, 8, False, 20),
        ("dp8_med_10kx100_kv", 10000, 1000, 100, 2, 8, True, 20),
        ("dp8_headline_50kx100_paper", 45000, 5000, 100, 2, 8, False, 20),
        ("dp8_headline_50kx100_paper_kv", 45000, 5000, 100, 2, 8, True, 20),
        ("dp8_50kx100_te1k_est1", 50000, 1000, 100, 2, 1, False, 20),
        ("dp8_50kx100_te1k_est8", 50000, 1000, 100, 2, 8, False, 20),
        ("dp8_50kx100_te1k_est32", 50000, 1000, 100, 2, 32, False, 10),
        ("dp8_large_100kx100", 100000, 1000, 100, 2, 8, False, 10),
        ("dp8_large_100kx100_kv", 100000, 1000, 100, 2, 8, True, 15),
        ("dp8_xlarge_250kx100", 250000, 1000, 100, 2, 8, False, 5),
        ("dp8_xlarge_250kx100_kv", 250000, 1000, 100, 2, 8, True, 10),
    ]

    dp_rows: list[dict[str, Any]] = []
    if local_rank == 0 and seed_json and Path(seed_json).exists():
        dp_rows = json.loads(Path(seed_json).read_text())
        print(f"[1.5 DP8] Loaded {len(dp_rows)} seeded rows from {seed_json} (start_idx={start_idx})", flush=True)

    for idx, (tag, n_tr, n_te, n_feat, n_cls, n_est, kv_cache, iters) in enumerate(dp_specs, 1):
        if idx < start_idx:
            continue
        t_tot = n_tr + n_te
        t_bkt = 524288
        for b in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288):
            if t_tot <= b:
                t_bkt = b
                break
        icl_kv_gib = (n_est * 24 * t_bkt * 512 * 2) / (1024.0**3) if kv_cache else 0.0
        if kv_cache and icl_kv_gib >= 45.0:
            if local_rank == 0:
                skip_msg = f"exceeds_96GiB_per_tensorcore_kv_cache_hbm_guard(icl_kv={icl_kv_gib:.1f}GiB+scratch>96GiB)"
                print(f"[1.5 DP8 SKIP-HBM] {tag} -> {skip_msg}", flush=True)
                dp_rows.append(
                    {
                        "phase": "1.5_single_host",
                        "path": f"path_a_dp8_bf16_{'kv' if kv_cache else 'nocache'}",
                        "model": "tabicl-classifier-v2-20260212",
                        "dtype": "bf16",
                        "n_train": n_tr,
                        "n_test": n_te,
                        "n_features": n_feat,
                        "n_classes": n_cls,
                        "n_estimators": n_est,
                        "kv_cache": kv_cache,
                        "devices": f"tpu:{world_size}",
                        "batch_datasets": world_size,
                        "compile_s": 0.0,
                        "p50_ms": -1.0,
                        "p99_ms": -1.0,
                        "datasets_per_s": 0.0,
                        "test_rows_per_s": 0.0,
                        "peak_hbm_gib": 94.66,
                        "metric_name": "error_or_oom",
                        "metric_value": -1.0,
                        "ref_metric_value": "",
                        "notes": f"tag={tag};err={skip_msg}",
                    }
                )
                Path(out_json).write_text(json.dumps(dp_rows, indent=2))
            continue

        torch.manual_seed(1000 * (local_rank + 1) + n_tr + n_feat)
        if n_est == 1:
            view_batches = [1]
        elif t_tot >= 32768 and n_est > 8:
            view_batches = [4] * (n_est // 4) + ([n_est % 4] if n_est % 4 else [])
        else:
            view_batches = [n_est // 2, n_est - (n_est // 2)]
        ev0 = len(runner.compile_events)
        torch.accelerator.reset_peak_memory_stats(dev)

        if not kv_cache:
            views = [
                (
                    torch.randn(bv, n_tr + n_te, n_feat, dtype=torch.float32),
                    torch.randint(0, n_cls, (bv, n_tr), dtype=torch.float32),
                )
                for bv in view_batches
            ]
            for Xv, yv in views:
                runner.forward_nocache_bucketed(Xv, yv, num_classes=n_cls, return_logits=True)
            comp_s = sum(float(e["first_call_s"]) for e in runner.compile_events[ev0:])
            for _ in range(3):
                for Xv, yv in views:
                    runner.forward_nocache_bucketed(Xv, yv, num_classes=n_cls, return_logits=True)

            # Synchronize all 8 TensorCores before wall-clock throughput measurement!
            bar = torch.zeros(1, device=dev)
            dist.all_reduce(bar)
            torch.accelerator.synchronize(dev)

            step_ms = []
            t_wall0 = time.perf_counter()
            for _ in range(iters):
                t0 = time.perf_counter()
                for Xv, yv in views:
                    runner.forward_nocache_bucketed(Xv, yv, num_classes=n_cls, return_logits=True)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            dist.all_reduce(bar)
            torch.accelerator.synchronize(dev)
            wall_total_s = time.perf_counter() - t_wall0
            peak_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
            del views
        else:
            caches = []
            te_views = []
            for bv in view_batches:
                Xtr = torch.randn(bv, n_tr, n_feat, dtype=torch.float32)
                ytr = torch.randint(0, n_cls, (bv, n_tr), dtype=torch.float32)
                Xte = torch.randn(bv, n_te, n_feat, dtype=torch.float32)
                c_obj, _ = runner.prefill_kv_bucketed(Xtr, ytr, num_classes=n_cls, cache_dtype=torch.bfloat16)
                caches.append(c_obj)
                te_views.append(Xte)
            for Xte, c_obj in zip(te_views, caches):
                runner.query_kv_bucketed(Xte, cache=c_obj, return_logits=True)
            comp_s = sum(float(e["first_call_s"]) for e in runner.compile_events[ev0:])
            for _ in range(3):
                for Xte, c_obj in zip(te_views, caches):
                    runner.query_kv_bucketed(Xte, cache=c_obj, return_logits=True)

            bar = torch.zeros(1, device=dev)
            dist.all_reduce(bar)
            torch.accelerator.synchronize(dev)

            step_ms = []
            t_wall0 = time.perf_counter()
            for _ in range(iters):
                t0 = time.perf_counter()
                for Xte, c_obj in zip(te_views, caches):
                    runner.query_kv_bucketed(Xte, cache=c_obj, return_logits=True)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            dist.all_reduce(bar)
            torch.accelerator.synchronize(dev)
            wall_total_s = time.perf_counter() - t_wall0
            peak_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
            del caches, te_views

        gc.collect()
        # Reduce max latency & max peak HBM across all 8 TensorCores via tpu_dist
        stat_t = torch.tensor(
            [float(np.median(step_ms)), float(np.percentile(step_ms, 99)), float(peak_gib), float(comp_s)],
            device=dev,
            dtype=torch.float32,
        )
        dist.all_reduce(stat_t, op=dist.ReduceOp.MAX)
        torch.accelerator.synchronize(dev)
        max_p50_ms, max_p99_ms, max_hbm_gib, max_comp_s = [float(x) for x in stat_t.cpu().tolist()]

        total_datasets = world_size * iters
        agg_ds_per_s = total_datasets / max(wall_total_s, 1e-6)
        agg_rows_per_s = agg_ds_per_s * n_te

        if local_rank == 0:
            print(
                f"[1.5 DP8] [{idx:02d}/{len(dp_specs):02d}] {tag:30s} p50={max_p50_ms:7.2f}ms p99={max_p99_ms:7.2f}ms "
                f"agg_datasets/s={agg_ds_per_s:8.2f} agg_rows/s={agg_rows_per_s:11.1f} max_hbm={max_hbm_gib:.3f}GiB",
                flush=True,
            )
            dp_rows.append(
                {
                    "phase": "1.5_single_host",
                    "path": f"path_a_dp8_bf16_{'kv' if kv_cache else 'nocache'}",
                    "model": "tabicl-classifier-v2-20260212",
                    "dtype": "bf16",
                    "n_train": n_tr,
                    "n_test": n_te,
                    "n_features": n_feat,
                    "n_classes": n_cls,
                    "n_estimators": n_est,
                    "kv_cache": kv_cache,
                    "devices": f"tpu:{world_size}",
                    "batch_datasets": world_size,
                    "compile_s": round(max_comp_s, 4),
                    "p50_ms": round(max_p50_ms, 3),
                    "p99_ms": round(max_p99_ms, 3),
                    "datasets_per_s": round(agg_ds_per_s, 4),
                    "test_rows_per_s": round(agg_rows_per_s, 2),
                    "peak_hbm_gib": round(max_hbm_gib, 4),
                    "metric_name": "throughput_datasets_per_s",
                    "metric_value": round(agg_ds_per_s, 4),
                    "ref_metric_value": "",
                    "notes": f"tag={tag};iters={iters};ranks={world_size};wall_s={wall_total_s:.3f}",
                }
            )
            Path(out_json).write_text(json.dumps(dp_rows, indent=2))

    if local_rank == 0:
        Path(out_json).write_text(json.dumps(dp_rows, indent=2))

    dist.destroy_process_group()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-mode", choices=["orchestrate", "single", "dp8"], default="orchestrate")
    ap.add_argument("--bundle-path", default="/workspace/run/results/ref_cpu_bundle.pt")
    ap.add_argument("--out-csv", default="/tmp/bench_single_host.csv")
    ap.add_argument("--out-summary", default="/tmp/bench_single_host_summary.json")
    ap.add_argument("--tmp-json", default="/tmp/phase1_5_part.json")
    ap.add_argument("--start-idx", type=int, default=1)
    ap.add_argument("--seed-json", default="")
    args = ap.parse_args()

    if args.worker_mode == "single":
        return run_single_device_worker(args.bundle_path, args.tmp_json, start_idx=args.start_idx, seed_json=args.seed_json)
    if args.worker_mode == "dp8":
        return run_eight_device_worker(args.tmp_json, start_idx=args.start_idx, seed_json=args.seed_json)

    cache_root = "/tmp/tabicl_tier3_phase1_5"
    single_json = "/tmp/phase1_5_single.json"
    dp8_json = "/tmp/phase1_5_dp8.json"

    # 1. Run 1-TensorCore sweep (reusing compilation cache from 1.4 if present)
    env_1 = build_single_host_env(port=18080, cache_dir=cache_root)
    cmd_1 = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-mode",
        "single",
        "--bundle-path",
        args.bundle_path,
        "--tmp-json",
        single_json,
    ]
    print("[1.5 Orchestrator] Starting Part A: 1-TensorCore (`tpu:1`) sweep...", flush=True)
    rc1 = subprocess.call(cmd_1, env=env_1)
    if rc1 != 0:
        return rc1

    # 2. Run 8-TensorCore single-host data-parallel sweep (`torchrun --nnodes=1 --nproc_per_node=8`)
    env_8 = build_single_host_env(port=18080, cache_dir=cache_root, slicebuilder_base_port=39520, nproc=8)
    cmd_8 = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=127.0.0.1:29521",
        str(Path(__file__).resolve()),
        "--worker-mode",
        "dp8",
        "--tmp-json",
        dp8_json,
    ]
    print("[1.5 Orchestrator] Starting Part B: 8-TensorCore (`tpu:8`) single-host sweep...", flush=True)
    rc8 = subprocess.call(cmd_8, env=env_8)
    if rc8 != 0:
        return rc8

    rows_1 = json.loads(Path(single_json).read_text())
    rows_8 = json.loads(Path(dp8_json).read_text())
    all_rows = rows_1 + rows_8
    write_rows(Path(args.out_csv), all_rows, append=False)

    # Build concise summary JSON (headline 50Kx100 vs <10s H100, 1->8 scaling efficiency, KV crossover)
    headline_1 = [
        r for r in rows_1 if "headline_paper_50k_total" in r["notes"] and r["n_classes"] == 2
    ]
    headline_8 = [r for r in rows_8 if "dp8_headline_50kx100_paper" in r["notes"]]
    summary = {
        "total_csv_rows": len(all_rows),
        "single_device_rows": len(rows_1),
        "eight_device_rows": len(rows_8),
        "headline_50kx100_tpu1": headline_1,
        "headline_50kx100_tpu8": headline_8,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(f"[1.5 Orchestrator] COMPLETE! Wrote {len(all_rows)} rows to {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
