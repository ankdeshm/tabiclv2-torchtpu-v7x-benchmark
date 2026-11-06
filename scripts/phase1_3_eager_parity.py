#!/usr/bin/env python3
"""Phase 1.3: TabICLv2 Eager Bring-Up & Numerical Parity on TorchTPU (`torch.device("tpu:0")`).

Evaluates all 15 datasets (10 classification + 5 regression) from `results/ref_cpu_bundle.pt`
in eager mode on TPU v7x (`torch_tpu`) across 5 configurations:
  1. `fp32` (`use_amp=False`), `kv_cache=False`
  2. `fp32` (`use_amp=False`), `kv_cache=True` (`kv_cache_dtype=fp32`)
  3. `bf16` (`use_amp=True`, `torch.bfloat16`), `kv_cache=False`
  4. `bf16` (`use_amp=True`, `torch.bfloat16`), `kv_cache=True` (`kv_cache_dtype=bf16`)
  5. `bf16` (`use_amp=True`, `torch.bfloat16`), `kv_cache=True` (`kv_cache_dtype=fp32`)

Verifies strict parity gates against the CPU fp32 reference:
  - `fp32`: `|AUC_tpu - AUC_cpu| <= 0.002` and `|RMSE_tpu / RMSE_cpu - 1| <= 0.01`
  - `bf16`: `|AUC_tpu - AUC_cpu| <= 0.005` and `|RMSE_tpu / RMSE_cpu - 1| <= 0.02` (targeting <= 0.01)
  - Never uses `fp16`.
  - Quantifies `bf16` vs `fp32` KV-cache accuracy and HBM byte footprint trade-offs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


def _kv_cache_nbytes(estimator: Any) -> tuple[int, str]:
    """Sum exact device byte footprint and dtype of stored TabICLCache tensors."""
    cache_map = getattr(estimator, "model_kv_cache_", None)
    if not cache_map:
        return 0, "none"
    total_bytes = 0
    dtypes: set[str] = set()
    for cache in cache_map.values():
        if getattr(cache, "col_cache", None) is not None:
            for entry in cache.col_cache.kv.values():
                for t in (entry.key, entry.value):
                    if t is not None:
                        total_bytes += t.numel() * t.element_size()
                        dtypes.add(str(t.dtype).replace("torch.", ""))
        if getattr(cache, "icl_cache", None) is not None:
            for entry in cache.icl_cache.kv.values():
                for t in (entry.key, entry.value):
                    if t is not None:
                        total_bytes += t.numel() * t.element_size()
                        dtypes.add(str(t.dtype).replace("torch.", ""))
        row_repr = getattr(cache, "row_repr", None)
        if row_repr is not None:
            total_bytes += row_repr.numel() * row_repr.element_size()
            dtypes.add(str(row_repr.dtype).replace("torch.", ""))
    return total_bytes, "+".join(sorted(dtypes)) if dtypes else "empty"


def _reset_peak_hbm(torch_mod: Any, device: Any) -> None:
    try:
        torch_mod.accelerator.reset_peak_memory_stats(device)
    except Exception:
        pass


def _read_peak_hbm_gib(torch_mod: Any, device: Any) -> float:
    for fn_name in ("max_memory_allocated", "memory_allocated"):
        fn = getattr(torch_mod.accelerator, fn_name, None)
        if callable(fn):
            try:
                val = int(fn(device))
                if val > 0:
                    return round(val / float(1024**3), 4)
            except Exception:
                pass
    return 0.0


def _sync_tpu(torch_mod: Any, device: Any) -> None:
    try:
        torch_mod.accelerator.synchronize(device)
    except Exception:
        try:
            torch_mod.accelerator.synchronize()
        except Exception:
            pass


def _upload_to_gcs(bucket_name: str, prefix: str, local_path: Path, rel_dest: str) -> None:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix.strip('/')}/{rel_dest.lstrip('/')}")
    blob.upload_from_filename(str(local_path))
    print(f"[GCS] Uploaded {local_path} -> gs://{bucket_name}/{blob.name}", flush=True)


def run_inner(args: argparse.Namespace) -> int:
    import torch
    import torch_tpu  # noqa: F401
    from sklearn.metrics import accuracy_score, log_loss, r2_score, roc_auc_score, root_mean_squared_error

    from tabicl import TabICLClassifier, TabICLRegressor
    from csv_schema import write_rows

    device = torch.device("tpu:0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check device placement & verify fp16 is never active
    probe = torch.ones((4, 4), device=device, dtype=torch.float32)
    _sync_tpu(torch, device)
    print(
        f"[1.3] TorchTPU initialized cleanly: probe.device={probe.device}, "
        f"accelerator={torch.accelerator.current_accelerator()}, "
        f"device_count={torch.accelerator.device_count()}",
        flush=True,
    )

    bundle = torch.load(args.bundle_path, map_location="cpu", weights_only=False)
    datasets: list[dict[str, Any]] = bundle["datasets"]
    seed: int = int(bundle.get("seed", 42))
    n_estimators: int = int(args.n_estimators or bundle.get("n_estimators", 8))

    # Warmup eager ops on TPU once before timed/logged parity runs
    rng = np.random.RandomState(0)
    Xw = rng.randn(64, 6).astype(np.float32)
    yw_c = (Xw[:, 0] > 0).astype(int)
    yw_r = (Xw[:, 0] * 1.5 - 0.3).astype(np.float32)
    for use_amp in (False, True):
        clf_w = TabICLClassifier(
            n_estimators=2, batch_size=2, kv_cache=True, device=device, use_amp=use_amp, offload_mode=False, random_state=seed
        )
        clf_w.fit(Xw[:48], yw_c[:48])
        _ = clf_w.predict_proba(Xw[48:])
        reg_w = TabICLRegressor(
            n_estimators=2, batch_size=2, kv_cache=True, device=device, use_amp=use_amp, offload_mode=False, random_state=seed
        )
        reg_w.fit(Xw[:48], yw_r[:48])
        _ = reg_w.predict(Xw[48:])
    _sync_tpu(torch, device)
    print("[1.3] Eager warmup complete on tpu:0.", flush=True)

    configs = [
        {"tag": "fp32_nocache", "dtype": "fp32", "use_amp": False, "kv_cache": False, "kv_cache_dtype": None, "auc_tol": 0.002, "rmse_rel_tol": 0.01},
        {"tag": "fp32_kv_fp32", "dtype": "fp32", "use_amp": False, "kv_cache": True, "kv_cache_dtype": "fp32", "auc_tol": 0.002, "rmse_rel_tol": 0.01},
        {"tag": "bf16_nocache", "dtype": "bf16", "use_amp": True, "kv_cache": False, "kv_cache_dtype": None, "auc_tol": 0.005, "rmse_rel_tol": 0.01},
        {"tag": "bf16_kv_bf16", "dtype": "bf16", "use_amp": True, "kv_cache": True, "kv_cache_dtype": "bf16", "auc_tol": 0.005, "rmse_rel_tol": 0.01},
        {"tag": "bf16_kv_fp32", "dtype": "bf16", "use_amp": True, "kv_cache": True, "kv_cache_dtype": "fp32", "auc_tol": 0.005, "rmse_rel_tol": 0.01},
    ]

    csv_rows: list[dict[str, Any]] = []
    parity_report: dict[str, Any] = {"configs": {}, "all_gates_passed": True, "failures": []}

    for cfg in configs:
        tag = cfg["tag"]
        dtype_label = cfg["dtype"]
        use_amp = cfg["use_amp"]
        kv_cache = cfg["kv_cache"]
        kv_dtype = cfg["kv_cache_dtype"]

        if dtype_label == "fp32":
            torch.set_float32_matmul_precision("highest")
        else:
            torch.set_float32_matmul_precision("high")

        if kv_dtype:
            os.environ["TABICL_KV_CACHE_DTYPE"] = kv_dtype
        else:
            os.environ.pop("TABICL_KV_CACHE_DTYPE", None)

        print(f"\n=== [1.3] Running config={tag} (use_amp={use_amp}, kv_cache={kv_cache}, kv_cache_dtype={kv_dtype}) ===", flush=True)
        cfg_records: list[dict[str, Any]] = []
        max_auc_abs_diff = 0.0
        max_rmse_rel_diff = 0.0
        total_kv_bytes = 0

        for ds in datasets:
            task_type = ds["task_type"]
            name = ds["name"]
            data_id = ds["data_id"]
            X_train = ds["X_train"]
            X_test = ds["X_test"]
            y_train = ds["y_train"]
            y_test = ds["y_test"]
            ref_metrics = ds["cpu_fp32_metrics"]

            _reset_peak_hbm(torch, device)
            _sync_tpu(torch, device)

            if task_type == "classification":
                est = TabICLClassifier(
                    n_estimators=n_estimators,
                    batch_size=n_estimators,
                    kv_cache=kv_cache,
                    device=device,
                    use_amp=use_amp,
                    offload_mode=False,
                    random_state=seed,
                )
                if kv_dtype:
                    est.kv_cache_dtype = kv_dtype
                t0 = time.perf_counter()
                est.fit(X_train, y_train)
                _sync_tpu(torch, device)
                t_fit = time.perf_counter() - t0

                t1 = time.perf_counter()
                proba = est.predict_proba(X_test)
                _sync_tpu(torch, device)
                t_pred = time.perf_counter() - t1

                total_ms = (t_fit + t_pred) * 1000.0
                pred_ms = t_pred * 1000.0
                latency_ms = pred_ms if kv_cache else total_ms

                y_pred = est.classes_[np.argmax(proba, axis=1)]
                acc = float(accuracy_score(y_test, y_pred))
                ll = float(log_loss(y_test, proba, labels=est.classes_))
                if len(est.classes_) == 2:
                    y_bin = (y_test == est.classes_[1]).astype(np.int32)
                    auc = float(roc_auc_score(y_bin, proba[:, 1]))
                else:
                    auc = float(roc_auc_score(y_test, proba, labels=est.classes_, multi_class="ovr", average="macro"))

                metrics = {"accuracy": acc, "roc_auc": auc, "log_loss": ll}
                auc_diff = abs(auc - float(ref_metrics["roc_auc"]))
                max_proba_diff = float(np.max(np.abs(proba - ds["cpu_fp32_proba"])))
                max_auc_abs_diff = max(max_auc_abs_diff, auc_diff)
                gate_ok = auc_diff <= cfg["auc_tol"]
                if not gate_ok:
                    parity_report["all_gates_passed"] = False
                    parity_report["failures"].append(
                        f"{tag}:{name}: |AUC_tpu - AUC_cpu|={auc_diff:.6f} > {cfg['auc_tol']}"
                    )

                kv_bytes, actual_kv_dtype = _kv_cache_nbytes(est)
                total_kv_bytes += kv_bytes
                peak_gib = _read_peak_hbm_gib(torch, device)
                print(
                    f"  [CLS] {name:<22} AUC={auc:.5f} (ref={ref_metrics['roc_auc']:.5f}, |d|={auc_diff:.5f}) "
                    f"max|dp|={max_proba_diff:.5f} fit={t_fit*1e3:.1f}ms pred={pred_ms:.1f}ms "
                    f"kv={kv_bytes/1024:.1f}KiB({actual_kv_dtype}) hbm={peak_gib:.3f}GiB [{'PASS' if gate_ok else 'FAIL'}]",
                    flush=True,
                )
                model_name = "tabicl-classifier-v2-20260212"
                extra_note = f"max_proba_diff={max_proba_diff:.6f};fit_ms={t_fit*1e3:.2f};pred_ms={pred_ms:.2f};kv_kib={kv_bytes/1024:.1f};kv_dtype={actual_kv_dtype}"
                rec = {
                    "data_id": data_id,
                    "name": name,
                    "task_type": "classification",
                    "metrics": metrics,
                    "ref_metrics": ref_metrics,
                    "auc_abs_diff": auc_diff,
                    "max_proba_abs_diff": max_proba_diff,
                    "fit_ms": round(t_fit * 1000.0, 3),
                    "pred_ms": round(pred_ms, 3),
                    "kv_cache_bytes": kv_bytes,
                    "kv_cache_actual_dtype": actual_kv_dtype,
                    "peak_hbm_gib": peak_gib,
                    "gate_ok": gate_ok,
                }
            else:
                est = TabICLRegressor(
                    n_estimators=n_estimators,
                    batch_size=n_estimators,
                    kv_cache=kv_cache,
                    device=device,
                    use_amp=use_amp,
                    offload_mode=False,
                    random_state=seed,
                )
                if kv_dtype:
                    est.kv_cache_dtype = kv_dtype
                t0 = time.perf_counter()
                est.fit(X_train, y_train)
                _sync_tpu(torch, device)
                t_fit = time.perf_counter() - t0

                t1 = time.perf_counter()
                y_pred = est.predict(X_test)
                _sync_tpu(torch, device)
                t_pred = time.perf_counter() - t1

                total_ms = (t_fit + t_pred) * 1000.0
                pred_ms = t_pred * 1000.0
                latency_ms = pred_ms if kv_cache else total_ms

                rmse = float(root_mean_squared_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                metrics = {"rmse": rmse, "r2": r2}
                rmse_rel = abs(rmse / float(ref_metrics["rmse"]) - 1.0)
                max_rmse_rel_diff = max(max_rmse_rel_diff, rmse_rel)
                gate_ok = rmse_rel <= cfg["rmse_rel_tol"]
                if not gate_ok:
                    parity_report["all_gates_passed"] = False
                    parity_report["failures"].append(
                        f"{tag}:{name}: |RMSE_tpu/RMSE_cpu - 1|={rmse_rel:.6f} > {cfg['rmse_rel_tol']}"
                    )

                kv_bytes, actual_kv_dtype = _kv_cache_nbytes(est)
                total_kv_bytes += kv_bytes
                peak_gib = _read_peak_hbm_gib(torch, device)
                print(
                    f"  [REG] {name:<22} RMSE={rmse:.5f} (ref={ref_metrics['rmse']:.5f}, rel={rmse_rel*100:.3f}%) "
                    f"R2={r2:.5f} fit={t_fit*1e3:.1f}ms pred={pred_ms:.1f}ms "
                    f"kv={kv_bytes/1024:.1f}KiB({actual_kv_dtype}) hbm={peak_gib:.3f}GiB [{'PASS' if gate_ok else 'FAIL'}]",
                    flush=True,
                )
                model_name = "tabicl-regressor-v2-20260212"
                extra_note = f"rmse_rel_diff={rmse_rel:.6f};fit_ms={t_fit*1e3:.2f};pred_ms={pred_ms:.2f};kv_kib={kv_bytes/1024:.1f};kv_dtype={actual_kv_dtype}"
                rec = {
                    "data_id": data_id,
                    "name": name,
                    "task_type": "regression",
                    "metrics": metrics,
                    "ref_metrics": ref_metrics,
                    "rmse_rel_diff": rmse_rel,
                    "fit_ms": round(t_fit * 1000.0, 3),
                    "pred_ms": round(pred_ms, 3),
                    "kv_cache_bytes": kv_bytes,
                    "kv_cache_actual_dtype": actual_kv_dtype,
                    "peak_hbm_gib": peak_gib,
                    "gate_ok": gate_ok,
                }

            cfg_records.append(rec)
            kv_label: Any = False if not kv_cache else f"kv_{kv_dtype}"
            for m_name, m_val in metrics.items():
                csv_rows.append(
                    {
                        "phase": "1.3_tpu_eager",
                        "path": f"path_a_eager_{tag}",
                        "model": model_name,
                        "dtype": dtype_label,
                        "n_train": ds["n_train"],
                        "n_test": ds["n_test"],
                        "n_features": ds["n_features"],
                        "n_classes": ds["n_classes"],
                        "n_estimators": n_estimators,
                        "kv_cache": kv_label,
                        "devices": "tpu_v7x:1",
                        "batch_datasets": 1,
                        "compile_s": 0.0,
                        "p50_ms": round(latency_ms, 3),
                        "p99_ms": round(latency_ms, 3),
                        "datasets_per_s": round(1000.0 / max(latency_ms, 1e-6), 4),
                        "test_rows_per_s": round(ds["n_test"] * 1000.0 / max(latency_ms, 1e-6), 2),
                        "peak_hbm_gib": peak_gib,
                        "metric_name": m_name,
                        "metric_value": round(m_val, 6),
                        "ref_metric_value": round(float(ref_metrics[m_name]), 6),
                        "notes": f"openml_id={data_id};name={name};gate={'PASS' if gate_ok else 'FAIL'};{extra_note}",
                    }
                )

        parity_report["configs"][tag] = {
            "max_auc_abs_diff": round(max_auc_abs_diff, 6),
            "max_rmse_rel_diff": round(max_rmse_rel_diff, 6),
            "total_kv_cache_bytes_across_15_datasets": total_kv_bytes,
            "records": cfg_records,
        }
        print(
            f"=== [1.3] Summary {tag}: max|dAUC|={max_auc_abs_diff:.6f} (tol={cfg['auc_tol']}), "
            f"max|dRMSE/RMSE|={max_rmse_rel_diff*100:.4f}% (tol={cfg['rmse_rel_tol']*100:.1f}%), "
            f"total_kv_MiB={total_kv_bytes/(1024**2):.2f} ===",
            flush=True,
        )

    csv_path = out_dir / "eager_parity_tpu.csv"
    write_rows(csv_path, csv_rows, append=False)

    json_path = out_dir / "eager_parity_tpu.json"
    json_path.write_text(json.dumps(parity_report, indent=2), encoding="utf-8")
    print(f"\n[1.3] Wrote {csv_path} and {json_path} (all_gates_passed={parity_report['all_gates_passed']})", flush=True)

    if args.bucket and args.run_prefix:
        _upload_to_gcs(args.bucket, args.run_prefix, csv_path, "results/eager_parity_tpu.csv")
        _upload_to_gcs(args.bucket, args.run_prefix, json_path, "results/eager_parity_tpu.json")

    if not parity_report["all_gates_passed"]:
        print(f"[1.3][FAIL] Gate failures: {parity_report['failures']}", flush=True)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true", help="Internal flag when relaunched inside single-host TPU env")
    ap.add_argument("--bundle-path", default="/workspace/run/results/ref_cpu_bundle.pt")
    ap.add_argument("--out-dir", default="/workspace/run/results")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--run-prefix", default="")
    ap.add_argument("--n-estimators", type=int, default=8)
    args = ap.parse_args()

    if not args.inner:
        from tpu_single_host_setup import build_single_host_env

        child_env = build_single_host_env(port=18080, use_metadata_proxy=True, cache_dir="/workspace/torchtpu_cache")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--inner", *sys.argv[1:]]
        proc = subprocess.run(cmd, env=child_env)
        return proc.returncode

    return run_inner(args)


if __name__ == "__main__":
    raise SystemExit(main())
