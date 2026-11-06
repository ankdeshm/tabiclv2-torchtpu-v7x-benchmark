#!/usr/bin/env python3
"""Phase 1.7 — Optional `torchao` FP8 (`W+A` on `ICLPredictor` Linear Layers) Evaluation on TorchTPU.

Strict Gate (per runbook Step 1.7):
  - Evaluate `torchao` dynamic FP8 (`Float8DynamicActivationFloat8WeightConfig`) on `icl_predictor` linear layers
    under `torch.compile(backend="tpu", dynamic=False)`.
  - Measure BOTH:
      1. Latency speedup vs `bf16` on `50K x 100` (`n_estimators=8`) and `16K x 100` (`n_estimators=8`)
      2. Numerical accuracy parity (`max |AUC_fp8 - AUC_bf16|`) across all 10 OpenML classification datasets
         in `results/ref_cpu_bundle.pt`
  - Keep ONLY if `speedup >= 15%` AND `max |AUC_fp8 - AUC_bf16| <= 0.005`; otherwise revert/reject cleanly
    and log exact findings in `results/fp8_eval_tpu.json` + `results/fp8_eval_tpu.csv`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from csv_schema import write_rows  # noqa: E402
from tpu_single_host_setup import build_single_host_env  # noqa: E402


def run_inner(bundle_path: str, out_csv: str, out_json: str) -> int:
    import torch
    import torch_tpu  # noqa: F401
    from sklearn.metrics import roc_auc_score
    from tabicl import TabICLClassifier
    from tabicl_tpu_bucketed import BucketedCompiledTabICL, attach_compiled_runner

    dev = torch.device("tpu:0")
    report: dict[str, Any] = {"torchao_available": False, "decision": "REJECT", "reason": ""}
    csv_rows: list[dict[str, Any]] = []

    try:
        import torchao
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        report["torchao_available"] = True
        report["torchao_version"] = getattr(torchao, "__version__", "unknown")
    except Exception as exc:
        report["reason"] = f"torchao FP8 unavailable in container: {exc}"
        Path(out_json).write_text(json.dumps(report, indent=2))
        print(f"[1.7 FP8] {report['reason']} -> REJECT (keep bf16)", flush=True)
        return 0

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    cls_datasets = [d for d in bundle["datasets"] if d["task_type"] == "classification"]
    seed = int(bundle.get("seed", 42))

    rng = np.random.RandomState(seed)
    X_init = rng.randn(64, 8).astype(np.float32)
    y_init = (X_init[:, 0] > 0).astype(np.int64)

    clf_bf16 = TabICLClassifier(n_estimators=8, random_state=seed, device=dev)
    clf_bf16.fit(X_init, y_init)
    model_bf16 = clf_bf16.model_
    model_fp8 = copy.deepcopy(model_bf16).to(dev).eval()

    try:
        # Quantize Linear layers inside `icl_predictor` (`tf_icl` QKV/O + MLP linears, excluding `y_encoder`/`decoder`)
        quantized_layers = []

        def _icl_linear_filter(mod: torch.nn.Module, fqn: str) -> bool:
            if isinstance(mod, torch.nn.Linear) and fqn.startswith("tf_icl"):
                if mod.in_features % 16 == 0 and mod.out_features % 16 == 0:
                    quantized_layers.append(fqn)
                    return True
            return False

        quantize_(model_fp8.icl_predictor, Float8DynamicActivationFloat8WeightConfig(), filter_fn=_icl_linear_filter)
        report["quantized_icl_linear_layers"] = len(quantized_layers)
        print(f"[1.7 FP8] Quantized {len(quantized_layers)} ICL Linear layers with torchao FP8 W+A.", flush=True)

        runner_bf16 = BucketedCompiledTabICL(model_bf16, dev, use_amp=True, kv_cache_dtype=torch.bfloat16)
        # Note: torchao Float8Tensor subclasses may have graph breaks if fullgraph=True; test fullgraph=False fallback if needed
        try:
            runner_fp8 = BucketedCompiledTabICL(model_fp8, dev, use_amp=True, kv_cache_dtype=torch.bfloat16)
            X_probe = torch.randn(4, 1024, 32, dtype=torch.float32)
            y_probe = torch.randint(0, 2, (4, 800), dtype=torch.float32)
            runner_fp8.forward_nocache_bucketed(X_probe, y_probe, num_classes=2, return_logits=True)
            report["fp8_fullgraph"] = True
        except Exception as fg_exc:
            report["fp8_fullgraph"] = False
            report["fp8_fullgraph_note"] = str(fg_exc)[:240]
            print(f"[1.7 FP8] fullgraph=True raised ({str(fg_exc)[:120]}), testing dynamic=False fullgraph=False...", flush=True)
            runner_fp8 = BucketedCompiledTabICL(model_fp8, dev, use_amp=True, kv_cache_dtype=torch.bfloat16)
            runner_fp8._nocache_compiled = torch.compile(runner_fp8._nocache_fn, backend="tpu", dynamic=False, fullgraph=False)
            runner_fp8._icl_only_compiled = torch.compile(runner_fp8._icl_only_fn, backend="tpu", dynamic=False, fullgraph=False)
            X_probe = torch.randn(4, 1024, 32, dtype=torch.float32)
            y_probe = torch.randint(0, 2, (4, 800), dtype=torch.float32)
            runner_fp8.forward_nocache_bucketed(X_probe, y_probe, num_classes=2, return_logits=True)

        # 1. Benchmark latency on 16Kx100 and 50Kx100 (n_estimators=8 -> 2 views of B=4)
        speed_results: dict[str, Any] = {}
        for label, n_tr, n_te, H in [("16kx100", 14384, 2000, 100), ("50kx100", 45000, 5000, 100)]:
            Xv = torch.randn(4, n_tr + n_te, H, dtype=torch.float32)
            yv = torch.randint(0, 2, (4, n_tr), dtype=torch.float32)
            for tag, r in [("bf16", runner_bf16), ("fp8", runner_fp8)]:
                _, comp_s = r.forward_nocache_bucketed(Xv, yv, num_classes=2, return_logits=True)
                for _ in range(2):
                    r.forward_nocache_bucketed(Xv, yv, num_classes=2, return_logits=True)
                    r.forward_nocache_bucketed(Xv, yv, num_classes=2, return_logits=True)
                torch.accelerator.reset_peak_memory_stats(dev)
                lats = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    r.forward_nocache_bucketed(Xv, yv, num_classes=2, return_logits=True)
                    r.forward_nocache_bucketed(Xv, yv, num_classes=2, return_logits=True)
                    lats.append((time.perf_counter() - t0) * 1000.0)
                p50 = float(np.median(lats))
                p99 = float(np.percentile(lats, 99))
                hbm = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
                speed_results[f"{label}_{tag}"] = {"p50_ms": round(p50, 3), "p99_ms": round(p99, 3), "compile_s": round(comp_s, 3), "peak_hbm_gib": round(hbm, 4)}
                csv_rows.append(
                    {
                        "phase": "1.7_fp8_eval",
                        "path": f"path_a_compiled_{tag}",
                        "model": "tabicl-classifier-v2-20260212",
                        "dtype": tag,
                        "n_train": n_tr,
                        "n_test": n_te,
                        "n_features": H,
                        "n_classes": 2,
                        "n_estimators": 8,
                        "kv_cache": False,
                        "devices": "tpu:1",
                        "batch_datasets": 1,
                        "compile_s": round(comp_s, 4),
                        "p50_ms": round(p50, 3),
                        "p99_ms": round(p99, 3),
                        "datasets_per_s": round(1000.0 / p50, 4),
                        "test_rows_per_s": round(n_te * 1000.0 / p50, 2),
                        "peak_hbm_gib": round(hbm, 4),
                        "metric_name": "latency_p50_ms",
                        "metric_value": round(p50, 3),
                        "ref_metric_value": round(speed_results[f"{label}_bf16"]["p50_ms"], 3),
                        "notes": f"label={label};fullgraph={report['fp8_fullgraph']}",
                    }
                )
                print(f"[1.7 FP8] {label} {tag}: p50={p50:.2f}ms p99={p99:.2f}ms hbm={hbm:.3f}GiB", flush=True)

        # 2. Parity check across all 10 OpenML classification datasets (`|AUC_fp8 - AUC_bf16| <= 0.005`)
        max_auc_diff_vs_bf16 = 0.0
        parity_list = []
        for ds in cls_datasets:
            X_tr, y_tr = ds["X_train"], ds["y_train"]
            X_te, y_te = ds["X_test"], ds["y_test"]
            aucs = {}
            for tag, m_obj, r_obj in [("bf16", model_bf16, runner_bf16), ("fp8", model_fp8, runner_fp8)]:
                est = TabICLClassifier(n_estimators=8, batch_size=8, random_state=seed, device=dev, use_amp=True, kv_cache=False, offload_mode=False)
                est._load_model = lambda s=est, m=m_obj: setattr(s, "model_", m)  # type: ignore[misc]
                attach_compiled_runner(est, runner=r_obj)
                est.fit(X_tr, y_tr)
                proba = est.predict_proba(X_te)
                if len(est.classes_) == 2:
                    aucs[tag] = float(roc_auc_score((y_te == est.classes_[1]).astype(np.int32), proba[:, 1]))
                else:
                    aucs[tag] = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=est.classes_))
            diff = abs(aucs["fp8"] - aucs["bf16"])
            max_auc_diff_vs_bf16 = max(max_auc_diff_vs_bf16, diff)
            parity_list.append({"name": ds["name"], "auc_bf16": round(aucs["bf16"], 6), "auc_fp8": round(aucs["fp8"], 6), "abs_diff": round(diff, 6)})
            print(f"[1.7 FP8 Parity] {ds['name']:30s} bf16={aucs['bf16']:.5f} fp8={aucs['fp8']:.5f} |d|={diff:.5f}", flush=True)

        speedup_50k = speed_results["50kx100_bf16"]["p50_ms"] / max(speed_results["50kx100_fp8"]["p50_ms"], 1e-6)
        speedup_16k = speed_results["16kx100_bf16"]["p50_ms"] / max(speed_results["16kx100_fp8"]["p50_ms"], 1e-6)
        keep = (speedup_50k >= 1.15) and (max_auc_diff_vs_bf16 <= 0.005)
        report.update(
            {
                "speed_results": speed_results,
                "speedup_16kx100": round(speedup_16k, 4),
                "speedup_50kx100": round(speedup_50k, 4),
                "max_auc_diff_vs_bf16": round(max_auc_diff_vs_bf16, 6),
                "parity_datasets": parity_list,
                "decision": "KEEP" if keep else "REJECT",
                "reason": (
                    f"speedup_50kx100={speedup_50k:.3f}x (gate >= 1.15x), "
                    f"max_auc_diff_vs_bf16={max_auc_diff_vs_bf16:.6f} (gate <= 0.005)"
                ),
            }
        )
        print(f"[1.7 FP8 Gate] Decision={report['decision']} ({report['reason']})", flush=True)
    except Exception as exc:
        report["decision"] = "REJECT"
        report["reason"] = f"TorchTPU/XLA FP8 lowering error: {str(exc)[:300]}"
        print(f"[1.7 FP8] {report['reason']} -> REJECT (keep bf16)", flush=True)

    if csv_rows:
        write_rows(Path(out_csv), csv_rows, append=False)
    Path(out_json).write_text(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--bundle-path", default="/workspace/run/results/ref_cpu_bundle.pt")
    ap.add_argument("--out-csv", default="/tmp/fp8_eval_tpu.csv")
    ap.add_argument("--out-json", default="/tmp/fp8_eval_tpu.json")
    args = ap.parse_args()

    if not args.inner:
        env = build_single_host_env(port=18080, cache_dir="/tmp/tabicl_tier3_fp8")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--inner", *sys.argv[1:]]
        return subprocess.call(cmd, env=env)
    return run_inner(args.bundle_path, args.out_csv, args.out_json)


if __name__ == "__main__":
    raise SystemExit(main())
