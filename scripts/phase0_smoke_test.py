#!/usr/bin/env python3
"""Phase 0.4 Smoke Test for TorchTPU 0.2.0 on GKE TPU7x (tpu-v7x-8-nodepool, 2x2x2).

Runs in two clean stages inside each JobSet pod (hostNetwork=True, 4 TPU chips/host):
  Stage A (Isolated Single-Host Process Inspection & BF16 Matmul):
    - Spawns a pure single-host Python process (no cross-host coordination) to run the exact prompt snippet:
        import torch, torch_tpu
        d = torch.device("tpu"); x = torch.randn(4096,4096,device=d,dtype=torch.bfloat16)
        print((x@x).float().sum().item())
    - Records physical chip vs addressable TensorCore device count on TPU v7x (`4` physical dual-core chips
      = `8` addressable TensorCore devices per host, `1` TensorCore device bound per OS process).
    - Captures `pip freeze`, `torch` / `torch_tpu` / `libtpu` / `tabicl` versions, and HBM metrics.
  Stage B (Multi-Host `tpu_dist` all_reduce across BOTH hosts):
    - Orders the 2 hosts by physical TPU slice coordinate (`TPU_WORKER_ID` 0 vs 1).
    - Launches `torchrun --nnodes=2 --nproc_per_node=8` (`16` TensorCore ranks across `8` physical chips on `2` hosts)
      with `TORCH_TPU_TOPOLOGY=2,2,2,2` and `dist.init_process_group(backend="tpu_dist")`.
    - Runs both the `4096x4096` bf16 matmul and `torch.distributed.all_reduce` (`backend="tpu_dist"`)
      across all 16 ranks spanning both hosts, asserting exact sum parity (`136.0`).
    - Uploads `env/` and `logs/` artifacts to `gs://<BUCKET>/tabicl-tpu/<RUN_ID>/`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


SINGLE_HOST_SNIPPET = r"""
import glob
import importlib.metadata as meta
import json, os, platform, socket
import torch, torch_tpu

d = torch.device("tpu")
x = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
matmul_sum = (x @ x).float().sum().item()
print(f"SINGLE_HOST_MATMUL_SUM={matmul_sum}", flush=True)

acc_cnt = torch.accelerator.device_count() if hasattr(torch, "accelerator") else None
cur_idx = torch.accelerator.current_device_index() if hasattr(torch, "accelerator") else None
hbm_alloc = None
if hasattr(torch, "accelerator"):
    try:
        hbm_alloc = int(torch.accelerator.max_memory_allocated(d))
    except Exception:
        pass

devices_per_chip = None
try:
    from torch_tpu._internal.utils.hardware import get_devices_per_chip
    devices_per_chip = get_devices_per_chip()
except Exception:
    pass

vfio_devs = sorted(d for d in glob.glob("/dev/vfio/[0-9]*"))

info = {
    "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "ip": socket.gethostbyname(socket.gethostname()),
    "python": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_tpu_version": meta.version("torch-tpu"),
    "libtpu_version": meta.version("libtpu"),
    "tabicl_version": meta.version("tabicl"),
    "tabicl_git_sha": open("/workspace/TABICL_COMMIT_SHA.txt").read().strip() if os.path.exists("/workspace/TABICL_COMMIT_SHA.txt") else "unknown",
    "physical_vfio_chips_on_host": len(vfio_devs),
    "vfio_devices": vfio_devs,
    "devices_per_physical_chip": devices_per_chip,
    "torch_accelerator_device_count_in_single_proc": acc_cnt,
    "torch_accelerator_current_device_index": cur_idx,
    "max_memory_allocated_bytes_after_4096_matmul": hbm_alloc,
    "single_host_4096_bf16_matmul_sum": matmul_sum,
}
with open(os.environ["STAGE_A_JSON"], "w") as f:
    json.dump(info, f, indent=2)
print("STAGE_A_OK", json.dumps(info), flush=True)
"""


MULTI_HOST_WORKER = r"""
import datetime as dt
import json, os, socket
import torch, torch_tpu
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.accelerator.set_device_index(0)
d = torch.device("tpu:0")

dist.init_process_group(backend="tpu_dist", timeout=dt.timedelta(minutes=10))

torch.manual_seed(1000 + rank)
x = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
m_sum = (x @ x).float().sum().item()

t = torch.tensor([float(rank + 1)], device=d, dtype=torch.float32)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
torch.accelerator.synchronize()
got = float(t.item())
expected = float(world_size * (world_size + 1) // 2)
assert abs(got - expected) < 1e-3, f"all_reduce mismatch on rank {rank}: got {got} vs expected {expected}"

print(
    f"[RANK {rank:02d}/{world_size:02d} local={local_rank} host={socket.gethostname()}] "
    f"matmul_sum={m_sum:.4f} all_reduce_sum={got:.1f} (expected={expected:.1f}) PASS",
    flush=True,
)

if local_rank == 0:
    out = {
        "node_rank": int(os.environ.get("GROUP_RANK", "0")),
        "rank0_global": rank,
        "world_size": world_size,
        "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
        "backend": "tpu_dist",
        "hostname": socket.gethostname(),
        "matmul_sum_rank": m_sum,
        "all_reduce_got": got,
        "all_reduce_expected": expected,
        "status": "PASS",
    }
    with open(os.environ["STAGE_B_JSON"], "w") as f:
        json.dump(out, f, indent=2)

dist.destroy_process_group()
"""


def upload_to_gcs(bucket_name: str, prefix: str, local_path: Path, rel_dest: str) -> None:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix.strip('/')}/{rel_dest.lstrip('/')}")
    blob.upload_from_filename(str(local_path))
    print(f"[GCS] Uploaded {local_path} -> gs://{bucket_name}/{blob.name}", flush=True)


def resolve_host(addr: str, timeout_s: int = 180) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(addr)
        except socket.gaierror:
            time.sleep(2.0)
    raise RuntimeError(f"Timed out resolving {addr}")


def exchange_peer_info(job_idx: int, peer_addr: str, port: int, my_payload: dict, timeout_s: int = 300) -> dict:
    """2-node TCP barrier exchanging JSON `{tpu_worker_id, ip, hostname, addressable_devices}`."""
    deadline = time.time() + timeout_s
    raw_self = (json.dumps(my_payload) + "\n").encode("utf-8")
    if job_idx == 0:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        srv.settimeout(max(5.0, deadline - time.time()))
        conn, _ = srv.accept()
        peer_raw = conn.recv(4096).decode("utf-8").strip()
        conn.sendall(raw_self)
        conn.close()
        srv.close()
        return json.loads(peer_raw)
    else:
        while time.time() < deadline:
            try:
                cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                cli.settimeout(5.0)
                cli.connect((peer_addr, port))
                cli.sendall(raw_self)
                peer_raw = cli.recv(4096).decode("utf-8").strip()
                cli.close()
                return json.loads(peer_raw)
            except OSError:
                time.sleep(2.0)
        raise RuntimeError(f"Timed out connecting to pod-0 barrier {peer_addr}:{port}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("--master-dns", required=True)
    ap.add_argument("--rdzv-port", type=int, default=29540)
    ap.add_argument("--barrier-port", type=int, default=29539)
    ap.add_argument("--slicebuilder-port", type=int, default=8070)
    args = ap.parse_args()

    job_idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    out_dir = Path(f"/tmp/phase0_host{job_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}][job_idx={job_idx}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    orig_tpu_env = {k: v for k, v in os.environ.items() if "TPU" in k or "MEGASCALE" in k or "PJRT" in k}
    log(f"Initial GKE TPU env keys: {json.dumps(orig_tpu_env, sort_keys=True)}")

    # 1. Capture pip freeze
    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    pip_path = out_dir / f"host_{job_idx}_pip_freeze.txt"
    pip_path.write_text(pip_freeze, encoding="utf-8")

    # 2. Stage A: Pure Single-Host isolated process (no cross-host env vars!)
    stage_a_json = out_dir / f"host_{job_idx}_single_host_env.json"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tpu_single_host_setup import build_single_host_env

    env_a = build_single_host_env(port=18080, use_metadata_proxy=True, cache_dir="/tmp/phase0_cache_a")
    env_a["STAGE_A_JSON"] = str(stage_a_json)

    log("Running Stage A (pure single-host device inspection & 4096x4096 bf16 matmul)...")
    proc_a = subprocess.run([sys.executable, "-c", SINGLE_HOST_SNIPPET], env=env_a, text=True, capture_output=True)
    log(f"Stage A rc={proc_a.returncode}\nSTDOUT:\n{proc_a.stdout}\nSTDERR:\n{proc_a.stderr}")
    if proc_a.returncode != 0:
        raise RuntimeError(f"Stage A failed with rc={proc_a.returncode}")

    info_a = json.loads(stage_a_json.read_text(encoding="utf-8"))
    # Parse PJRT's addressable device count from stderr (`Only using 1 device out of all the 8 addressable devices.`)
    m_addr = re.search(r"out of all the (\d+) addressable devices", proc_a.stderr)
    pjrt_addressable = int(m_addr.group(1)) if m_addr else (
        int(info_a["physical_vfio_chips_on_host"]) * int(info_a.get("devices_per_physical_chip") or 2)
    )
    info_a["pjrt_addressable_tensorcore_devices_on_host"] = pjrt_addressable
    stage_a_json.write_text(json.dumps(info_a, indent=2), encoding="utf-8")
    log(
        f"Stage A complete: physical_chips={info_a['physical_vfio_chips_on_host']}, "
        f"devices_per_chip={info_a['devices_per_physical_chip']}, "
        f"pjrt_addressable_devices_per_host={pjrt_addressable}"
    )

    # Wait briefly so Stage A child releases /dev/vfio/* before Stage B multi-host launch
    time.sleep(3.0)
    my_ip = socket.gethostbyname(socket.gethostname())
    tpu_worker_id = int(orig_tpu_env.get("TPU_WORKER_ID", str(job_idx)))
    master_pod_ip = resolve_host(args.master_dns) if job_idx != 0 else my_ip
    my_info = {
        "job_idx": job_idx,
        "tpu_worker_id": tpu_worker_id,
        "ip": my_ip,
        "hostname": socket.gethostname(),
        "addressable_devices": pjrt_addressable,
    }
    log(f"Entering pre-Stage-B TCP barrier against {master_pod_ip}:{args.barrier_port} with {my_info}...")
    peer_info = exchange_peer_info(job_idx, master_pod_ip, args.barrier_port, my_info)
    both = sorted([my_info, peer_info], key=lambda d: int(d["tpu_worker_id"]))
    hosts_ordered = [d["ip"] for d in both]
    node_rank = int(tpu_worker_id)
    devices_per_host = int(pjrt_addressable)
    sb_addrs = ",".join(
        f"{h}:{args.slicebuilder_port + i}" for h in hosts_ordered for i in range(devices_per_host)
    )
    log(f"Barrier complete: ordered_hosts={hosts_ordered}, my_node_rank={node_rank}, slicebuilder_count={len(sb_addrs.split(','))}")

    # 3. Stage B: Multi-host `tpu_dist` all_reduce across BOTH hosts (16 TensorCore ranks across 8 chips)
    stage_b_script = out_dir / "stage_b_worker.py"
    stage_b_script.write_text(MULTI_HOST_WORKER, encoding="utf-8")
    stage_b_json = out_dir / f"host_{node_rank}_multihost_allreduce.json"

    env_b = os.environ.copy()
    for strip_k in ("LD_PRELOAD", "MOCK_METADATA_PORT", "TPU_PROCESS_ADDRESSES", "TPU_PROCESS_PORT"):
        env_b.pop(strip_k, None)
    env_b.update(
        {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": str(node_rank),
            "TPU_WORKER_HOSTNAMES": ",".join(hosts_ordered),
            "TPU_TOPOLOGY": "2x2x2",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "TPU_HOST_BOUNDS": "1,1,2",
            "TORCH_TPU_TOPOLOGY": "2,2,2,2",
            "TORCH_TPU_DEVICES_PER_HOST": str(devices_per_host),
            "TORCH_TPU_BASE_PORT": str(args.slicebuilder_port),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addrs,
            "STAGE_B_JSON": str(stage_b_json),
        }
    )

    cmd_b = [
        "torchrun",
        "--nnodes=2",
        f"--nproc_per_node={devices_per_host}",
        "--rdzv_backend=static",
        f"--rdzv_endpoint={hosts_ordered[0]}:{args.rdzv_port}",
        f"--node_rank={node_rank}",
        str(stage_b_script),
    ]
    log(f"Running Stage B multi-host tpu_dist all_reduce: {' '.join(cmd_b)}")
    proc_b = subprocess.run(cmd_b, env=env_b, text=True, capture_output=True)
    log(f"Stage B rc={proc_b.returncode}\nSTDOUT:\n{proc_b.stdout}\nSTDERR:\n{proc_b.stderr}")
    if proc_b.returncode != 0:
        raise RuntimeError(f"Stage B failed with rc={proc_b.returncode}")

    info_b = json.loads(stage_b_json.read_text(encoding="utf-8"))
    combined = {
        "job_completion_index": job_idx,
        "tpu_worker_id": node_rank,
        "gke_injected_tpu_env": orig_tpu_env,
        "single_host": info_a,
        "multi_host_tpu_dist": info_b,
    }
    combined_path = out_dir / f"host_{node_rank}_env_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    smoke_log_path = out_dir / f"phase0_smoke_host_{node_rank}.log"
    smoke_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    upload_to_gcs(args.bucket, args.run_prefix, pip_path, f"env/host_{node_rank}_pip_freeze.txt")
    upload_to_gcs(args.bucket, args.run_prefix, combined_path, f"env/host_{node_rank}_env_summary.json")
    upload_to_gcs(args.bucket, args.run_prefix, smoke_log_path, f"logs/phase0_smoke_host_{node_rank}.log")
    log("PHASE 0.4 SMOKE TEST SUCCEEDED ON THIS HOST!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
