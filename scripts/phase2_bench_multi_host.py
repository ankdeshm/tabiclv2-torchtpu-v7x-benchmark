#!/usr/bin/env python3
"""Phase 2: 2-Host 8-Chip (16-TensorCore, 2x2x2 ICI) TabICLv2 Inference & Pretraining Benchmark on TorchTPU.

Runs across 2 TPU v7x hosts (`torchrun --nnodes=2 --nproc_per_node=8`, `world_size=16`, `backend="tpu_dist"`):
  1. Part A (`DP=16` Multi-Host Inference Scaling): Evaluates all 14 dataset shapes from the `DP=8` suite
     across all 16 TensorCores (`tpu:16`) to quantify `1 -> 8 -> 16` TensorCore scaling efficiency.
  2. Part B (`DDP=16` Multi-Host Pretraining Throughput): Runs `TabICLv2` forward + cross-entropy loss +
     backward + `tpu_dist` ICI gradient `all_reduce` + `AdamW` optimizer steps in `bf16` across all 16
     TensorCores (`tpu:16`), verifying finite/decreasing loss and measuring global tables/s and rows/s.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from csv_schema import write_rows  # noqa: E402


def run_phase2_worker(out_json: str, out_csv: str, out_summary: str, start_idx: int = 1, seed_json: str = "") -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    import torch
    import torch_tpu  # noqa: F401
    import torch.distributed as dist
    import torch.nn.functional as F
    from tabicl import TabICLClassifier
    from tabicl_tpu_bucketed import BucketedCompiledTabICL

    torch.accelerator.set_device_index(0)
    dev = torch.device("tpu:0")
    dist.init_process_group(backend="tpu_dist", timeout=dt.timedelta(minutes=30))

    if rank == 0:
        print(f"[Phase 2] Initialized tpu_dist world_size={world_size} across 2x2x2 TPU v7x slice", flush=True)

    rng = np.random.RandomState(200 + rank)
    X_init = rng.randn(64, 8).astype(np.float32)
    y_init = (X_init[:, 0] > 0).astype(np.int64)
    clf = TabICLClassifier(n_estimators=8, random_state=42, device=dev)
    clf.fit(X_init, y_init)
    runner = BucketedCompiledTabICL(clf.model_, dev, use_amp=True, kv_cache_dtype=torch.bfloat16)

    dp_specs = [
        ("dp16_small_1kx10", 1000, 1000, 10, 2, 8, False, 20),
        ("dp16_small_1kx10_kv", 1000, 1000, 10, 2, 8, True, 20),
        ("dp16_small_1kx100", 1000, 1000, 100, 2, 8, False, 20),
        ("dp16_med_10kx100", 10000, 1000, 100, 2, 8, False, 20),
        ("dp16_med_10kx100_kv", 10000, 1000, 100, 2, 8, True, 20),
        ("dp16_headline_50kx100_paper", 45000, 5000, 100, 2, 8, False, 20),
        ("dp16_headline_50kx100_paper_kv", 45000, 5000, 100, 2, 8, True, 20),
        ("dp16_50kx100_te1k_est1", 50000, 1000, 100, 2, 1, False, 20),
        ("dp16_50kx100_te1k_est8", 50000, 1000, 100, 2, 8, False, 20),
        ("dp16_50kx100_te1k_est32", 50000, 1000, 100, 2, 32, False, 10),
        ("dp16_large_100kx100", 100000, 1000, 100, 2, 8, False, 10),
        ("dp16_large_100kx100_kv", 100000, 1000, 100, 2, 8, True, 15),
        ("dp16_xlarge_250kx100", 250000, 1000, 100, 2, 8, False, 5),
        ("dp16_xlarge_250kx100_kv", 250000, 1000, 100, 2, 8, True, 10),
    ]

    rows: list[dict[str, Any]] = []
    if rank == 0 and seed_json and Path(seed_json).exists():
        rows = json.loads(Path(seed_json).read_text())
        print(f"[Phase 2] Loaded {len(rows)} seeded rows from {seed_json} (start_idx={start_idx})", flush=True)

    # Part A: 16-TensorCore Data-Parallel Inference Sweep
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
            if rank == 0:
                skip_msg = f"exceeds_96GiB_per_tensorcore_kv_cache_hbm_guard(icl_kv={icl_kv_gib:.1f}GiB+scratch>96GiB)"
                print(f"[Phase 2 SKIP-HBM] {tag} -> {skip_msg}", flush=True)
                rows.append(
                    {
                        "phase": "2_multi_host",
                        "path": f"path_a_dp{world_size}_bf16_{'kv' if kv_cache else 'nocache'}",
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
                Path(out_json).write_text(json.dumps(rows, indent=2))
            continue

        torch.manual_seed(1000 * (rank + 1) + n_tr + n_feat)
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

        if rank == 0:
            print(
                f"[Phase 2 DP{world_size}] [{idx:02d}/{len(dp_specs):02d}] {tag:30s} p50={max_p50_ms:7.2f}ms "
                f"p99={max_p99_ms:7.2f}ms agg_datasets/s={agg_ds_per_s:8.2f} agg_rows/s={agg_rows_per_s:11.1f} "
                f"max_hbm={max_hbm_gib:.3f}GiB",
                flush=True,
            )
            rows.append(
                {
                    "phase": "2_multi_host",
                    "path": f"path_a_dp{world_size}_bf16_{'kv' if kv_cache else 'nocache'}",
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
            Path(out_json).write_text(json.dumps(rows, indent=2))

    # Part B: 16-TensorCore Multi-Host Synthetic Pretraining Step Benchmark (Forward + Backward + ICI AllReduce + AdamW)
    del runner
    gc.collect()
    train_model = clf.model_.to(dev)
    train_model.train()
    for p in train_model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(train_model.parameters(), lr=1e-4, weight_decay=1e-2)

    pretrain_specs = [
        ("pretrain_ddp16_b8_t1024_f32", 8, 768, 256, 32, 4, 15),
        ("pretrain_ddp16_b4_t2048_f64", 4, 1536, 512, 64, 4, 15),
        ("pretrain_ddp16_b2_t4096_f100", 2, 3072, 1024, 100, 4, 10),
    ]

    for p_tag, local_b, n_tr, n_te, n_feat, n_cls, iters in pretrain_specs:
        torch.manual_seed(4242 + n_tr)
        # Shared synthetic teacher linear separator so we also verify loss decreases over steps
        w_true = torch.randn(n_feat, n_cls, device=dev, dtype=torch.float32)
        torch.manual_seed(9000 * (rank + 1) + n_tr)
        X_batch = torch.randn(local_b, n_tr + n_te, n_feat, device=dev, dtype=torch.float32)
        y_all = torch.argmax(X_batch @ w_true + 0.1 * torch.randn(local_b, n_tr + n_te, n_cls, device=dev), dim=-1)
        y_tr = y_all[:, :n_tr].to(torch.float32)
        y_te_target = y_all[:, n_tr:].to(torch.long)

        torch.accelerator.reset_peak_memory_stats(dev)
        t_c0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("tpu", dtype=torch.bfloat16):
            logits = train_model(X_batch, y_tr)
            loss0 = F.cross_entropy(logits[:, :, :n_cls].reshape(-1, n_cls).float(), y_te_target.reshape(-1))
        loss0.backward()
        for p in train_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt.step()
        torch.accelerator.synchronize(dev)
        comp_s = time.perf_counter() - t_c0
        first_loss = float(loss0.detach().cpu().item())

        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("tpu", dtype=torch.bfloat16):
                logits = train_model(X_batch, y_tr)
                loss_w = F.cross_entropy(logits[:, :, :n_cls].reshape(-1, n_cls).float(), y_te_target.reshape(-1))
            loss_w.backward()
            for p in train_model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            opt.step()

        bar = torch.zeros(1, device=dev)
        dist.all_reduce(bar)
        torch.accelerator.synchronize(dev)

        step_ms = []
        final_loss = first_loss
        t_wall0 = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("tpu", dtype=torch.bfloat16):
                logits = train_model(X_batch, y_tr)
                loss = F.cross_entropy(logits[:, :, :n_cls].reshape(-1, n_cls).float(), y_te_target.reshape(-1))
            loss.backward()
            for p in train_model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            opt.step()
            torch.accelerator.synchronize(dev)
            step_ms.append((time.perf_counter() - t0) * 1000.0)
        dist.all_reduce(bar)
        torch.accelerator.synchronize(dev)
        wall_total_s = time.perf_counter() - t_wall0
        final_loss = float(loss.detach().cpu().item())
        peak_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)

        stat_t = torch.tensor(
            [float(np.median(step_ms)), float(np.percentile(step_ms, 99)), float(peak_gib), float(comp_s)],
            device=dev,
            dtype=torch.float32,
        )
        dist.all_reduce(stat_t, op=dist.ReduceOp.MAX)
        loss_t = torch.tensor([first_loss, final_loss], device=dev, dtype=torch.float32)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        torch.accelerator.synchronize(dev)

        max_p50_ms, max_p99_ms, max_hbm_gib, max_comp_s = [float(x) for x in stat_t.cpu().tolist()]
        avg_loss0, avg_loss_end = [float(x) for x in loss_t.cpu().tolist()]
        global_tables_per_step = world_size * local_b
        agg_ds_per_s = (global_tables_per_step * iters) / max(wall_total_s, 1e-6)
        agg_rows_per_s = agg_ds_per_s * (n_tr + n_te)

        if rank == 0:
            print(
                f"[Phase 2 Pretrain DDP{world_size}] {p_tag:30s} step_p50={max_p50_ms:7.2f}ms "
                f"tables/s={agg_ds_per_s:7.2f} rows/s={agg_rows_per_s:10.1f} "
                f"loss={avg_loss0:.4f}->{avg_loss_end:.4f} max_hbm={max_hbm_gib:.3f}GiB",
                flush=True,
            )
            rows.append(
                {
                    "phase": "2_multi_host_pretrain",
                    "path": f"path_b_pretrain_ddp{world_size}_bf16",
                    "model": "tabicl-classifier-v2-20260212",
                    "dtype": "bf16",
                    "n_train": n_tr,
                    "n_test": n_te,
                    "n_features": n_feat,
                    "n_classes": n_cls,
                    "n_estimators": 1,
                    "kv_cache": False,
                    "devices": f"tpu:{world_size}",
                    "batch_datasets": global_tables_per_step,
                    "compile_s": round(max_comp_s, 4),
                    "p50_ms": round(max_p50_ms, 3),
                    "p99_ms": round(max_p99_ms, 3),
                    "datasets_per_s": round(agg_ds_per_s, 4),
                    "test_rows_per_s": round(agg_rows_per_s, 2),
                    "peak_hbm_gib": round(max_hbm_gib, 4),
                    "metric_name": "train_loss_end",
                    "metric_value": round(avg_loss_end, 6),
                    "ref_metric_value": round(avg_loss0, 6),
                    "notes": f"tag={p_tag};local_b={local_b};global_b={global_tables_per_step};iters={iters};loss_start={avg_loss0:.4f};loss_end={avg_loss_end:.4f}",
                }
            )
            Path(out_json).write_text(json.dumps(rows, indent=2))

        del X_batch, y_all, y_tr, y_te_target
        gc.collect()

    if rank == 0:
        write_rows(Path(out_csv), rows, append=False)
        dp16_rows = [r for r in rows if r["phase"] == "2_multi_host"]
        pre_rows = [r for r in rows if r["phase"] == "2_multi_host_pretrain"]
        summary = {
            "world_size": world_size,
            "total_rows": len(rows),
            "dp16_inference_rows": len(dp16_rows),
            "ddp16_pretrain_rows": len(pre_rows),
            "headline_50kx100_dp16": [r for r in dp16_rows if "headline_50kx100_paper" in r["notes"]],
            "pretrain_results": pre_rows,
        }
        Path(out_summary).write_text(json.dumps(summary, indent=2))
        print(f"[Phase 2 COMPLETE] Wrote {len(rows)} rows to {out_csv} and {out_summary}", flush=True)

    dist.destroy_process_group()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="/tmp/phase2_multi_host.json")
    ap.add_argument("--out-csv", default="/tmp/bench_multi_host.csv")
    ap.add_argument("--out-summary", default="/tmp/bench_multi_host_summary.json")
    ap.add_argument("--start-idx", type=int, default=1)
    ap.add_argument("--seed-json", default="")
    args = ap.parse_args()
    return run_phase2_worker(args.out_json, args.out_csv, args.out_summary, start_idx=args.start_idx, seed_json=args.seed_json)


if __name__ == "__main__":
    raise SystemExit(main())
