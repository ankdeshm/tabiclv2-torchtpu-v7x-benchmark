# TabICLv2 Inference on Google Cloud TPU v7x (Ironwood) with TorchTPU

- **Upstream TabICLv2 Commit:** `d83091b845a8c5f7c087bd54964f755a9e560f45`
- **Hardware:** Google Cloud TPU `v7x` (`2x2x2` slice = `8` physical chips / `16` TensorCores, `192 GB` HBM3e/chip)
- **Stack:** PyTorch + `torch_tpu` (`PJRT_DEVICE=TPU`) with shape-bucketed `torch.compile` (`scripts/tabicl_tpu_bucketed.py`) & KV-cache support

## Milestone Status

- **Phase 0 (TPU Smoke Test) & Phase 1.1–1.4 (CPU Reference, Pytest Suite, Eager Numerical Parity, XLA Compile & Profile):** Completed
- **Phase 1.5 (Single-Host `tpu:1` 94-Point Grid + `tpu:8` Scaling Sweep):** Actively running (`34/94` `tpu:1` configurations completed so far; see `results/phase1_5_tpu1_live_progress.txt`)
- **Phase 2 (2-Host `8`-Chip / `16`-TensorCore Multi-Host Scaling):** Queued immediately after Phase 1.5

## Latest Live TPU v7x (`1` TensorCore) Progress Snapshot

```text
[015/094] grid_ntrain_nfeat            tr= 10000 te= 1000 f=500 cls= 2 est= 8 kv=False bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=38.990GiB
[016/094] grid_ntrain_nfeat            tr= 10000 te= 1000 f=500 cls= 2 est= 8 kv=True  bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=39.689GiB
[017/094] grid_ntrain_nfeat            tr= 50000 te= 1000 f= 10 cls= 2 est= 8 kv=False bf16 -> p50= 9137.35ms p99= 9138.00ms rows/s=     109.4 hbm=31.815GiB
[018/094] grid_ntrain_nfeat            tr= 50000 te= 1000 f= 10 cls= 2 est= 8 kv=True  bf16 -> p50=  189.49ms p99=  190.02ms rows/s=    5277.3 hbm=55.677GiB
[019/094] grid_ntrain_nfeat            tr= 50000 te= 1000 f=100 cls= 2 est= 8 kv=False bf16 -> p50=21085.70ms p99=21088.57ms rows/s=      47.4 hbm=50.709GiB
[020/094] grid_ntrain_nfeat            tr= 50000 te= 1000 f=100 cls= 2 est= 8 kv=True  bf16 -> p50=  347.03ms p99=  349.51ms rows/s=    2881.6 hbm=65.004GiB
[021/094] grid_ntrain_nfeat            tr= 50000 te= 1000 f=500 cls= 2 est= 8 kv=False bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.317GiB
[022/094] grid_ntrain_nfeat            tr= 50000 te= 1000 f=500 cls= 2 est= 8 kv=True  bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.552GiB
[023/094] grid_ntrain_nfeat            tr=100000 te= 1000 f= 10 cls= 2 est= 8 kv=False bf16 -> p50=32470.13ms p99=32471.01ms rows/s=      30.8 hbm=43.288GiB
[024/094] grid_ntrain_nfeat            tr=100000 te= 1000 f= 10 cls= 2 est= 8 kv=True  bf16 -> p50=  336.16ms p99=  336.62ms rows/s=    2974.8 hbm=61.759GiB
[025/094] grid_ntrain_nfeat            tr=100000 te= 1000 f=100 cls= 2 est= 8 kv=False bf16 -> p50=53488.44ms p99=53490.51ms rows/s=      18.7 hbm=56.664GiB
[026/094] grid_ntrain_nfeat            tr=100000 te= 1000 f=100 cls= 2 est= 8 kv=True  bf16 -> p50=  488.98ms p99=  489.87ms rows/s=    2045.1 hbm=64.601GiB
[027/094] grid_ntrain_nfeat            tr=100000 te= 1000 f=500 cls= 2 est= 8 kv=False bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.281GiB
[028/094] grid_ntrain_nfeat            tr=100000 te= 1000 f=500 cls= 2 est= 8 kv=True  bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.643GiB
[029/094] grid_ntrain_nfeat            tr=250000 te= 1000 f= 10 cls= 2 est= 8 kv=False bf16 -> p50=116079.10ms p99=116080.88ms rows/s=       8.6 hbm=49.003GiB
[030/094] grid_ntrain_nfeat            tr=250000 te= 1000 f= 10 cls= 2 est= 8 kv=True  bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.369GiB
[031/094] grid_ntrain_nfeat            tr=250000 te= 1000 f=100 cls= 2 est= 8 kv=False bf16 -> p50=158169.97ms p99=158176.56ms rows/s=       6.3 hbm=61.514GiB
[032/094] grid_ntrain_nfeat            tr=250000 te= 1000 f=100 cls= 2 est= 8 kv=True  bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.659GiB
[033/094] grid_ntrain_nfeat            tr=250000 te= 1000 f=500 cls= 2 est= 8 kv=False bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=94.293GiB
[034/094] grid_ntrain_nfeat            tr=250000 te= 1000 f=500 cls= 2 est= 8 kv=True  bf16 -> p50=   -1.00ms p99=   -1.00ms rows/s=       0.0 hbm=92.639GiB
```
