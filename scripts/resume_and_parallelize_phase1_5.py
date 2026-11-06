#!/usr/bin/env python3
"""Salvage completed tpu:1 specs from log and launch resumed Part A (host1) or Part B DP8 (host0)."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase1_5_bench_single_host as m  # noqa: E402
from tpu_single_host_setup import build_single_host_env  # noqa: E402


def salvage(log_path: pathlib.Path, out_seed: pathlib.Path) -> int:
    specs = m._build_single_device_specs()
    pat = re.compile(
        r"^\[(\d+)/\d+\].*?->\s*p50=\s*([-\d.]+)ms\s+p99=\s*([-\d.]+)ms\s+rows/s=\s*([-\d.]+)\s+hbm=([-\d.]+)GiB"
    )
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        mm = pat.match(line.strip())
        if not mm:
            continue
        idx = int(mm.group(1))
        p50, p99, rps, hbm = map(float, mm.groups()[1:])
        sp = specs[idx - 1]
        n_tr = sp["n_train"]
        n_te = sp["n_test"]
        n_feat = sp["n_features"]
        n_cls = sp["n_classes"]
        n_est = sp["n_estimators"]
        kv = sp["kv_cache"]
        dt_s = sp["dtype"]
        kv_tag = "kv" if kv else "nocache"
        ok = p50 > 0
        rows.append(
            {
                "phase": "1.5_single_host",
                "path": f"path_a_compiled_{dt_s}_{kv_tag}",
                "model": "tabicl-regressor-v2-20260212" if n_cls == 0 else "tabicl-classifier-v2-20260212",
                "dtype": dt_s,
                "n_train": n_tr,
                "n_test": n_te,
                "n_features": n_feat,
                "n_classes": n_cls,
                "n_estimators": n_est,
                "kv_cache": kv,
                "devices": "tpu:1",
                "batch_datasets": 1,
                "compile_s": 0.0,
                "p50_ms": round(p50, 3),
                "p99_ms": round(p99, 3),
                "datasets_per_s": round(1000.0 / p50, 4) if ok else 0.0,
                "test_rows_per_s": round(rps, 2) if ok else 0.0,
                "peak_hbm_gib": round(hbm, 4),
                "metric_name": "latency_p50_ms" if ok else "error_or_oom",
                "metric_value": round(p50, 3) if ok else -1.0,
                "ref_metric_value": 10000.0 if (n_tr in (45000, 50000) and n_feat == 100 and n_est == 8 and ok) else "",
                "notes": f"tag={sp['tag']};salvaged_from_run1",
            }
        )
    out_seed.write_text(json.dumps(rows, indent=2))
    print(f"Salvaged {len(rows)} rows -> {out_seed}", flush=True)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["resume_single", "run_dp8"])
    args = ap.parse_args()

    if args.role == "resume_single":
        seed_fp = pathlib.Path("/tmp/phase1_5_seed_1_34.json")
        n = salvage(pathlib.Path("/tmp/phase1_5_run.log"), seed_fp)
        env = build_single_host_env(port=18080, cache_dir="/tmp/tabicl_tier3_phase1_5")
        return subprocess.call(
            [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "phase1_5_bench_single_host.py"),
                "--worker-mode",
                "single",
                "--bundle-path",
                "/workspace/run/results/ref_cpu_bundle.pt",
                "--start-idx",
                str(n + 1),
                "--seed-json",
                str(seed_fp),
                "--tmp-json",
                "/tmp/phase1_5_single.json",
            ],
            env=env,
        )

    seed_dp8 = pathlib.Path("/tmp/phase1_5_dp8_seed.json")
    existing_dp8 = pathlib.Path("/tmp/phase1_5_dp8.json")
    start_dp8 = 1
    if existing_dp8.exists():
        rows_existing = json.loads(existing_dp8.read_text())
        if rows_existing:
            seed_dp8.write_text(json.dumps(rows_existing, indent=2))
            start_dp8 = len(rows_existing) + 1
            print(f"[DP8 Resume] Seeded {len(rows_existing)} rows -> starting at idx={start_dp8}", flush=True)

    shutil_mod = __import__("shutil")
    shutil_mod.rmtree("/dev/shm/torch_tpu_cache", ignore_errors=True)
    env = build_single_host_env(port=18080, cache_dir=None, slicebuilder_base_port=39520, nproc=8)
    # Must be set in parent env BEFORE torchrun spawns Python children because torch_tpu .pth
    # autoloads at interpreter init and otherwise enables shared /dev/shm/torch_tpu_cache/default when WORLD_SIZE>1!
    env["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"
    env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"
    env.pop("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", None)
    env.pop("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT", None)
    env.pop("TORCH_TPU_CACHE_DIR", None)
    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=127.0.0.1:29521",
        str(SCRIPT_DIR / "phase1_5_bench_single_host.py"),
        "--worker-mode",
        "dp8",
        "--start-idx",
        str(start_dp8),
        "--tmp-json",
        "/tmp/phase1_5_dp8.json",
    ]
    if start_dp8 > 1:
        cmd.extend(["--seed-json", str(seed_dp8)])
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
