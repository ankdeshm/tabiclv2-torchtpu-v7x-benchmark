#!/usr/bin/env python3
"""Bucketed `torch.compile(backend="tpu", dynamic=False, fullgraph=True)` Engine for TabICLv2 on TorchTPU.

Provides:
  - Fixed power-of-two bucket grids along `(n_rows, n_features, batch_tables)` so TPU XLA/Pallas
    compiles each bucket once and serves arbitrary dataset shapes with zero recompilations and zero
    Dynamo graph breaks (`fullgraph=True`).
  - Mathematically exact masked padding:
      * Circular feature grouping modulo `d_real` (`real_features`)
      * Masked `RowInteraction` feature attention over `d_real + num_cls` columns
      * Masked `SetTransformer` inducing-point attention and `QASSMaxMLP` `log(train_size_real)`
      * Masked `ICLPredictor` context attention via TorchTPU Pallas rectangular Splash Attention
        (`tabicl::tpu_splash_sdpa_seg`) for `T_kv >= 16,384` and fused/tiled XLA SDPA for `< 16,384`
  - Both `kv_cache=False` (end-to-end single-graph fused OR automatic HBM-safe stage-tiled execution
    up to `500K+` rows x `500` features) and `kv_cache=True` (`prefill` + `query` pure-tensor
    `fullgraph=True` compiled graphs supporting `bf16` and `fp32` KV caches).
  - Seamless drop-in attachment to `TabICLClassifier` and `TabICLRegressor` via `attach_compiled_runner`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from tabicl._model.kv_cache import KVCacheEntry

# Mandatory Step 1.4 bucket grids (+ Step 1.5 ultra-long 128K / 256K / 512K buckets)
ROW_BUCKETS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
)
TEST_ROW_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384)
FEATURE_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
ESTIMATOR_BUCKETS: tuple[int, ...] = (1, 4, 8, 16, 32)
BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def pick_bucket(val: int, buckets: tuple[int, ...]) -> int:
    """Return smallest bucket >= val (or round up to multiple of 1024 if above max bucket)."""
    if val <= 0:
        return buckets[0]
    for b in buckets:
        if val <= b:
            return b
    return int(math.ceil(val / 1024.0) * 1024)


@dataclass
class BucketedKVCache:
    """Pure-tensor bucket-aligned KV cache on TPU HBM (`bf16` or `fp32`)."""

    col_keys: tuple[Tensor, ...]
    col_values: tuple[Tensor, ...]
    icl_keys: tuple[Tensor, ...]
    icl_values: tuple[Tensor, ...]
    real_batch: int
    padded_batch: int
    real_train_size: int
    padded_train_size: int
    real_features: int
    padded_features: int
    num_classes: int
    cache_dtype: torch.dtype

    def nbytes(self) -> int:
        total = 0
        for seq in (self.col_keys, self.col_values, self.icl_keys, self.icl_values):
            for t in seq:
                total += t.numel() * t.element_size()
        return total


class BucketedCompiledTabICL:
    """Manages `torch.compile(backend='tpu', dynamic=False, fullgraph=True)` kernels for a `TabICL` model."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool = True,
        kv_cache_dtype: torch.dtype = torch.bfloat16,
        max_fused_elements: int = 4_500_000,
        row_tile_size: int = 4096,
        col_group_tile: int = 32,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.use_amp = bool(use_amp)
        self.kv_cache_dtype = kv_cache_dtype if use_amp else torch.float32
        if self.kv_cache_dtype == torch.float16:
            raise ValueError("fp16 is strictly forbidden on TorchTPU; use torch.bfloat16 or torch.float32.")
        self.max_fused_elements = int(max_fused_elements)
        self.row_tile_size = int(row_tile_size)
        self.col_group_tile = int(col_group_tile)

        torch._dynamo.config.recompile_limit = max(
            getattr(torch._dynamo.config, "recompile_limit", 8), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                getattr(torch._dynamo.config, "accumulated_recompile_limit", 64), 1024
            )

        self.compile_events: list[dict[str, Any]] = []
        self._seen_keys: set[tuple[Any, ...]] = set()

        self._nocache_fn = self._make_nocache_fn()
        self._nocache_compiled = torch.compile(
            self._nocache_fn, backend="tpu", dynamic=False, fullgraph=True
        )

        self._prefill_bfloat16_fn = self._make_prefill_fn(torch.bfloat16, icl_amp=True)
        self._prefill_float32_fn = self._make_prefill_fn(torch.float32, icl_amp=False)
        self._prefill_bfloat16_compiled = torch.compile(
            self._prefill_bfloat16_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._prefill_float32_compiled = torch.compile(
            self._prefill_float32_fn, backend="tpu", dynamic=False, fullgraph=True
        )

        self._query_fn = self._make_query_fn()
        self._query_compiled = torch.compile(
            self._query_fn, backend="tpu", dynamic=False, fullgraph=True
        )

        # Stage-tiled fullgraph=True kernels for ultra-large tables (100K-500K rows x 100-500 features)
        self._col_chunk_fn = self._make_col_chunk_fn()
        self._col_chunk_compiled = torch.compile(
            self._col_chunk_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._col_chunk_prefill_bfloat16_fn = self._make_col_chunk_prefill_fn(torch.bfloat16)
        self._col_chunk_prefill_float32_fn = self._make_col_chunk_prefill_fn(torch.float32)
        self._col_chunk_prefill_bfloat16_compiled = torch.compile(
            self._col_chunk_prefill_bfloat16_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._col_chunk_prefill_float32_compiled = torch.compile(
            self._col_chunk_prefill_float32_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._row_tile_fn = self._make_row_tile_fn()
        self._row_tile_compiled = torch.compile(
            self._row_tile_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._icl_only_fn = self._make_icl_only_fn()
        self._icl_only_compiled = torch.compile(
            self._icl_only_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._icl_prefill_bfloat16_fn = self._make_icl_prefill_fn(torch.bfloat16, icl_amp=True)
        self._icl_prefill_float32_fn = self._make_icl_prefill_fn(torch.float32, icl_amp=False)
        self._icl_prefill_bfloat16_compiled = torch.compile(
            self._icl_prefill_bfloat16_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._icl_prefill_float32_compiled = torch.compile(
            self._icl_prefill_float32_fn, backend="tpu", dynamic=False, fullgraph=True
        )

    def _sync(self) -> None:
        try:
            torch.accelerator.synchronize(self.device)
        except Exception:
            torch.accelerator.synchronize()

    def _make_nocache_fn(self) -> Callable[..., Tensor]:
        model = self.model
        use_amp = self.use_amp
        fixed_nc = 0 if model.max_classes == 0 else model.max_classes

        def _fn(X: Tensor, y_train: Tensor, d_vec: Tensor, train_sz: Tensor) -> Tensor:
            col_emb = model.col_embedder._train_forward(
                X,
                y_train=y_train,
                d=d_vec,
                embed_with_test=False,
                num_classes=fixed_nc,
                train_size_override=train_sz,
            )
            row_d = model._resolve_row_d(d_vec)
            reprs = model.row_interactor._train_forward(col_emb, d=row_d)
            if use_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    out = model.icl_predictor._icl_predictions(
                        reprs, y_train, train_size_override=train_sz
                    )
            else:
                out = model.icl_predictor._icl_predictions(
                    reprs, y_train, train_size_override=train_sz
                )
            return out.float()

        return _fn

    def _make_col_chunk_fn(self) -> Callable[..., Tensor]:
        col_emb = self.model.col_embedder

        def _fn(feat_chunk: Tensor, y_chunk: Tensor, train_sz: Tensor) -> Tensor:
            src = col_emb.in_linear(feat_chunk)
            if col_emb.max_classes > 0:
                y_e = col_emb.y_encoder(y_chunk.float())
            else:
                y_e = col_emb.y_encoder(y_chunk.unsqueeze(-1))
            src = col_emb._add_train_target_emb(src, y_e, train_sz)
            src = col_emb.tf_col(src, train_size=train_sz)
            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                return feat_chunk * weights + biases
            return src

        return _fn

    def _make_col_chunk_prefill_fn(self, cache_dtype: torch.dtype) -> Callable[..., tuple[Tensor, ...]]:
        col_emb = self.model.col_embedder

        def _fn(feat_chunk: Tensor, y_chunk: Tensor, train_sz: Tensor) -> tuple[Tensor, ...]:
            src = col_emb.in_linear(feat_chunk)
            if col_emb.max_classes > 0:
                y_e = col_emb.y_encoder(y_chunk.float())
            else:
                y_e = col_emb.y_encoder(y_chunk.unsqueeze(-1))
            src = col_emb._add_train_target_emb(src, y_e, train_sz)
            ks: list[Tensor] = []
            vs: list[Tensor] = []
            for block in col_emb.tf_col.blocks:
                skip_mask = (src == block.skip_value).all(dim=(-2, -1), keepdim=True)
                ind_vec = block.ind_vectors.expand(*src.shape[:-2], block.num_inds, src.shape[-1])
                hidden = block.multihead_attn1(
                    ind_vec, src, src, src_len_override=train_sz, valid_kv_len=train_sz
                )
                out_b, k_b, v_b = block.multihead_attn2(src, hidden, hidden, need_kv=True)
                src = torch.where(skip_mask, torch.full_like(out_b, block.skip_value), out_b)
                ks.append(k_b.to(cache_dtype))
                vs.append(v_b.to(cache_dtype))
            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                src = feat_chunk * weights + biases
            return (src, *ks, *vs)

        return _fn

    def _make_row_tile_fn(self) -> Callable[..., Tensor]:
        model = self.model

        def _fn(col_emb_tile: Tensor, d_vec: Tensor) -> Tensor:
            row_d = model._resolve_row_d(d_vec)
            return model.row_interactor._train_forward(col_emb_tile, d=row_d)

        return _fn

    def _make_icl_only_fn(self) -> Callable[..., Tensor]:
        model = self.model
        use_amp = self.use_amp

        def _fn(reprs: Tensor, y_train: Tensor, train_sz: Tensor) -> Tensor:
            if use_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    out = model.icl_predictor._icl_predictions(
                        reprs, y_train, train_size_override=train_sz
                    )
            else:
                out = model.icl_predictor._icl_predictions(
                    reprs, y_train, train_size_override=train_sz
                )
            return out.float()

        return _fn

    def _make_icl_prefill_fn(self, cache_dtype: torch.dtype, icl_amp: bool) -> Callable[..., tuple[Tensor, ...]]:
        icl = self.model.icl_predictor

        def _fn(reprs: Tensor, y_train: Tensor, train_sz: Tensor) -> tuple[Tensor, ...]:
            icl_ks: list[Tensor] = []
            icl_vs: list[Tensor] = []
            if icl_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    if icl.max_classes > 0:
                        ry = icl.y_encoder(y_train.float())
                    else:
                        ry = icl.y_encoder(y_train.unsqueeze(-1))
                    r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                    for block in icl.tf_icl.blocks:
                        r_in, k_l, v_l = block(
                            q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                        )
                        icl_ks.append(k_l.to(cache_dtype))
                        icl_vs.append(v_l.to(cache_dtype))
            else:
                if icl.max_classes > 0:
                    ry = icl.y_encoder(y_train.float())
                else:
                    ry = icl.y_encoder(y_train.unsqueeze(-1))
                r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                for block in icl.tf_icl.blocks:
                    r_in, k_l, v_l = block(
                        q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                    )
                    icl_ks.append(k_l.to(cache_dtype))
                    icl_vs.append(v_l.to(cache_dtype))
            return (*icl_ks, *icl_vs)

        return _fn

    def _make_prefill_fn(self, cache_dtype: torch.dtype, icl_amp: bool) -> Callable[..., tuple[Tensor, ...]]:
        model = self.model
        col_emb = model.col_embedder
        icl = model.icl_predictor

        def _fn(X_tr: Tensor, y_tr: Tensor, d_vec: Tensor, train_sz: Tensor) -> tuple[Tensor, ...]:
            x_grp = col_emb.feature_grouping(X_tr, real_features=d_vec)
            if col_emb.reserve_cls_tokens > 0:
                x_grp = F.pad(x_grp, (0, 0, col_emb.reserve_cls_tokens, 0), value=-100.0)
            features = x_grp.transpose(1, 2)
            y_exp = y_tr.unsqueeze(1).expand(-1, features.shape[1], -1)

            src = col_emb.in_linear(features)
            if col_emb.max_classes > 0:
                y_e = col_emb.y_encoder(y_exp.float())
            else:
                y_e = col_emb.y_encoder(y_exp.unsqueeze(-1))
            src = col_emb._add_train_target_emb(src, y_e, train_sz)

            col_ks: list[Tensor] = []
            col_vs: list[Tensor] = []
            for block in col_emb.tf_col.blocks:
                skip_mask = (src == block.skip_value).all(dim=(-2, -1), keepdim=True)
                ind_vec = block.ind_vectors.expand(*src.shape[:-2], block.num_inds, src.shape[-1])
                hidden = block.multihead_attn1(
                    ind_vec, src, src, src_len_override=train_sz, valid_kv_len=train_sz
                )
                out_b, k_b, v_b = block.multihead_attn2(src, hidden, hidden, need_kv=True)
                src = torch.where(skip_mask, torch.full_like(out_b, block.skip_value), out_b)
                col_ks.append(k_b.to(cache_dtype))
                col_vs.append(v_b.to(cache_dtype))

            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                src = features * weights + biases

            reprs = model.row_interactor._train_forward(
                src.transpose(1, 2), d=model._resolve_row_d(d_vec)
            )

            icl_ks: list[Tensor] = []
            icl_vs: list[Tensor] = []
            if icl_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    if icl.max_classes > 0:
                        ry = icl.y_encoder(y_tr.float())
                    else:
                        ry = icl.y_encoder(y_tr.unsqueeze(-1))
                    r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                    for block in icl.tf_icl.blocks:
                        r_in, k_l, v_l = block(
                            q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                        )
                        icl_ks.append(k_l.to(cache_dtype))
                        icl_vs.append(v_l.to(cache_dtype))
            else:
                if icl.max_classes > 0:
                    ry = icl.y_encoder(y_tr.float())
                else:
                    ry = icl.y_encoder(y_tr.unsqueeze(-1))
                r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                for block in icl.tf_icl.blocks:
                    r_in, k_l, v_l = block(
                        q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                    )
                    icl_ks.append(k_l.to(cache_dtype))
                    icl_vs.append(v_l.to(cache_dtype))

            return (*col_ks, *col_vs, *icl_ks, *icl_vs)

        return _fn

    def _make_query_fn(self) -> Callable[..., Tensor]:
        model = self.model
        col_emb = model.col_embedder
        icl = model.icl_predictor
        use_amp = self.use_amp
        n_col = len(col_emb.tf_col.blocks)
        n_icl = len(icl.tf_icl.blocks)

        def _fn(X_te: Tensor, d_vec: Tensor, train_sz: Tensor, *flat_kv: Tensor) -> Tensor:
            col_ks = flat_kv[:n_col]
            col_vs = flat_kv[n_col : 2 * n_col]
            icl_ks = flat_kv[2 * n_col : 2 * n_col + n_icl]
            icl_vs = flat_kv[2 * n_col + n_icl : 2 * n_col + 2 * n_icl]

            x_grp = col_emb.feature_grouping(X_te, real_features=d_vec)
            if col_emb.reserve_cls_tokens > 0:
                x_grp = F.pad(x_grp, (0, 0, col_emb.reserve_cls_tokens, 0), value=-100.0)
            features = x_grp.transpose(1, 2)

            src = col_emb.in_linear(features)
            for idx, block in enumerate(col_emb.tf_col.blocks):
                skip_mask = (src == block.skip_value).all(dim=(-2, -1), keepdim=True)
                entry = KVCacheEntry(key=col_ks[idx], value=col_vs[idx])
                out_b = block.multihead_attn2(src, cached_kv=entry)
                src = torch.where(skip_mask, torch.full_like(out_b, block.skip_value), out_b)

            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                src = features * weights + biases

            reprs = model.row_interactor._train_forward(
                src.transpose(1, 2), d=model._resolve_row_d(d_vec)
            )

            if use_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    r_q = reprs
                    for idx, block in enumerate(icl.tf_icl.blocks):
                        entry = KVCacheEntry(key=icl_ks[idx], value=icl_vs[idx])
                        r_q = block(
                            q=r_q,
                            rope=icl.tf_icl.rope,
                            cached_kv=entry,
                            src_len_override=train_sz,
                            valid_kv_len=train_sz,
                        )
                    if icl.norm_first:
                        r_q = icl.ln(r_q)
                    out = icl.decoder(r_q)
            else:
                r_q = reprs
                for idx, block in enumerate(icl.tf_icl.blocks):
                    entry = KVCacheEntry(key=icl_ks[idx], value=icl_vs[idx])
                    r_q = block(
                        q=r_q,
                        rope=icl.tf_icl.rope,
                        cached_kv=entry,
                        src_len_override=train_sz,
                        valid_kv_len=train_sz,
                    )
                if icl.norm_first:
                    r_q = icl.ln(r_q)
                out = icl.decoder(r_q)
            return out.float()

        return _fn

    def explain_all(self, B: int = 4, T: int = 1024, H: int = 32, T_test: int = 256) -> dict[str, Any]:
        """Run `torch._dynamo.explain` on nocache, prefill, and query paths and return graph/break counts."""
        results: dict[str, Any] = {}
        with torch.no_grad():
            X = torch.randn(B, T, H, device=self.device, dtype=torch.float32)
            y = torch.zeros(B, T, device=self.device, dtype=torch.float32)
            d_vec = torch.full((B,), max(1, H - 4), device=self.device, dtype=torch.long)
            tsz = torch.tensor(int(T * 0.75), device=self.device, dtype=torch.long)

            exp_nc = torch._dynamo.explain(self._nocache_fn)(X, y, d_vec, tsz)
            results["nocache"] = {
                "shape": (B, T, H),
                "graph_count": int(exp_nc.graph_count),
                "graph_break_count": int(exp_nc.graph_break_count),
                "break_reasons": [str(r) for r in exp_nc.break_reasons],
            }

            pf_fn = (
                self._prefill_bfloat16_fn
                if (self.use_amp and self.kv_cache_dtype == torch.bfloat16)
                else self._prefill_float32_fn
            )
            exp_pf = torch._dynamo.explain(pf_fn)(X, y, d_vec, tsz)
            results["kv_prefill"] = {
                "shape": (B, T, H),
                "graph_count": int(exp_pf.graph_count),
                "graph_break_count": int(exp_pf.graph_break_count),
                "break_reasons": [str(r) for r in exp_pf.break_reasons],
            }

            flat_kv = pf_fn(X, y, d_vec, tsz)
            X_te = torch.randn(B, T_test, H, device=self.device, dtype=torch.float32)
            exp_q = torch._dynamo.explain(self._query_fn)(X_te, d_vec, tsz, *flat_kv)
            results["kv_query"] = {
                "shape": (B, T_test, H),
                "graph_count": int(exp_q.graph_count),
                "graph_break_count": int(exp_q.graph_break_count),
                "break_reasons": [str(r) for r in exp_q.break_reasons],
            }
        return results

    def _run_stage_tiled_col_row(
        self,
        X_bkt: Tensor,
        y_bkt: Tensor,
        d_vec: Tensor,
        train_sz: Tensor,
        store_col_cache: bool = False,
        cache_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Compute `(B, T, 512)` row representations for massive `(B, T, H)` using fixed compiled stage tiles."""
        col_emb = self.model.col_embedder
        B, T, H = X_bkt.shape
        C = col_emb.reserve_cls_tokens

        # Micro-batch B=1 per step when T >= 32768 to cap XLA static HBM well below 16 GiB even up to 500K x 500
        b_step = 1 if T >= 32768 else min(B, 2)
        reprs_list: list[Tensor] = []
        col_k_batches: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]
        col_v_batches: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]

        prev_prec = torch.get_float32_matmul_precision()
        try:
            for b0 in range(0, B, b_step):
                xb = X_bkt[b0 : b0 + b_step]
                yb = y_bkt[b0 : b0 + b_step]
                db = d_vec[b0 : b0 + b_step]

                x_grp = col_emb.feature_grouping(xb, real_features=db)  # (bs, T, H, 3)
                if C > 0:
                    x_grp = F.pad(x_grp, (0, 0, C, 0), value=-100.0)
                features = x_grp.transpose(1, 2).contiguous()  # (bs, H+C, T, 3)
                del x_grp

                HC = features.shape[1]
                max_g = 8 if T >= 262144 else (16 if T >= 131072 else self.col_group_tile)
                g_tile = min(max_g, HC)
                HC_pad = int(math.ceil(HC / g_tile) * g_tile)
                if HC_pad > HC:
                    features = F.pad(features, (0, 0, 0, 0, 0, HC_pad - HC), value=-100.0)

                torch.set_float32_matmul_precision("highest")
                col_chunks: list[Tensor] = []
                k_chunks: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]
                v_chunks: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]

                ultra_cells = T * HC
                use_bf16_buf = self.use_amp and (ultra_cells > 25_000_000)
                stage_cpu = ultra_cells > 100_000_000

                for g0 in range(0, HC_pad, g_tile):
                    f_sub = features[:, g0 : g0 + g_tile].contiguous()
                    y_sub = yb.unsqueeze(1).expand(-1, g_tile, -1).contiguous()
                    if store_col_cache:
                        fn_pf = (
                            self._col_chunk_prefill_bfloat16_compiled
                            if cache_dtype == torch.bfloat16
                            else self._col_chunk_prefill_float32_compiled
                        )
                        res = fn_pf(f_sub, y_sub, train_sz)
                        ch_out = res[0].to(torch.bfloat16) if use_bf16_buf else res[0]
                        col_chunks.append(ch_out.cpu() if stage_cpu else ch_out)
                        n_blk = len(col_emb.tf_col.blocks)
                        for bi in range(n_blk):
                            k_chunks[bi].append(res[1 + bi])
                            v_chunks[bi].append(res[1 + n_blk + bi])
                    else:
                        ch_raw = self._col_chunk_compiled(f_sub, y_sub, train_sz)
                        ch_out = ch_raw.to(torch.bfloat16) if use_bf16_buf else ch_raw
                        col_chunks.append(ch_out.cpu() if stage_cpu else ch_out)
                del features

                if store_col_cache:
                    for bi in range(len(col_emb.tf_col.blocks)):
                        col_k_batches[bi].append(torch.cat(k_chunks[bi], dim=1)[:, :HC].contiguous())
                        col_v_batches[bi].append(torch.cat(v_chunks[bi], dim=1)[:, :HC].contiguous())

                torch.set_float32_matmul_precision("high" if self.use_amp else "highest")
                r_tile = min(self.row_tile_size, T)
                row_outs: list[Tensor] = []
                for t0 in range(0, T, r_tile):
                    tile_slices = [
                        (ch[:, :, t0 : t0 + r_tile, :].to(self.device, dtype=torch.float32) if stage_cpu else ch[:, :, t0 : t0 + r_tile, :].float())
                        for ch in col_chunks
                    ]
                    c_sub = torch.cat(tile_slices, dim=1)[:, :HC].transpose(1, 2).contiguous()
                    del tile_slices
                    row_outs.append(self._row_tile_compiled(c_sub, db))
                    del c_sub
                del col_chunks
                reprs_list.append(torch.cat(row_outs, dim=1))
        finally:
            torch.set_float32_matmul_precision(prev_prec)

        reprs = torch.cat(reprs_list, dim=0)
        if store_col_cache:
            col_ks = tuple(torch.cat(col_k_batches[bi], dim=0) for bi in range(len(col_emb.tf_col.blocks)))
            col_vs = tuple(torch.cat(col_v_batches[bi], dim=0) for bi in range(len(col_emb.tf_col.blocks)))
            return reprs, col_ks, col_vs
        return reprs, (), ()

    def forward_nocache_bucketed(
        self,
        X: Tensor,
        y_train: Tensor,
        num_classes: int = 0,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
    ) -> tuple[Tensor, float]:
        """Run bucket-padded compiled forward (`kv_cache=False`) and return `(out_real, compile_s)`."""
        if (self.model.max_classes > 0 and num_classes > self.model.max_classes) or (
            not self.model.col_embedder.feature_group
        ):
            with torch.no_grad():
                out_fb = self.model(
                    X.to(self.device, dtype=torch.float32),
                    y_train.to(self.device, dtype=torch.float32),
                    return_logits=return_logits,
                    softmax_temperature=softmax_temperature,
                    num_classes=num_classes if num_classes > 0 else None,
                )
                self._sync()
            return out_fb.float(), 0.0

        B_real, T_real, H_real = X.shape
        train_real = y_train.shape[1]
        test_real = T_real - train_real

        B_bkt = pick_bucket(B_real, BATCH_BUCKETS)
        T_bkt = pick_bucket(T_real, ROW_BUCKETS)
        H_bkt = pick_bucket(H_real, FEATURE_BUCKETS)

        X_bkt = torch.zeros((B_bkt, T_bkt, H_bkt), device=self.device, dtype=torch.float32)
        X_bkt[:B_real, :T_real, :H_real] = X.to(self.device, dtype=torch.float32)

        y_bkt = torch.zeros((B_bkt, T_bkt), device=self.device, dtype=torch.float32)
        y_bkt[:B_real, :train_real] = y_train.to(self.device, dtype=torch.float32)
        if B_bkt > B_real:
            X_bkt[B_real:] = X_bkt[0:1]
            y_bkt[B_real:] = y_bkt[0:1]

        d_vec = torch.full((B_bkt,), H_real, device=self.device, dtype=torch.long)
        train_sz = torch.tensor(train_real, device=self.device, dtype=torch.long)

        use_tiled = (B_bkt * T_bkt * (H_bkt + 4)) > self.max_fused_elements
        key = ("nocache_tiled" if use_tiled else "nocache_fused", B_bkt, T_bkt, H_bkt, self.use_amp)
        is_first = key not in self._seen_keys

        with torch.no_grad():
            t0 = time.perf_counter()
            if not use_tiled:
                prev_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision("highest")
                try:
                    raw_full = self._nocache_compiled(X_bkt, y_bkt, d_vec, train_sz)
                finally:
                    torch.set_float32_matmul_precision(prev_prec)
            else:
                reprs, _, _ = self._run_stage_tiled_col_row(X_bkt, y_bkt, d_vec, train_sz, store_col_cache=False)
                icl_b_step = 1 if T_bkt >= 65536 else (min(B_bkt, 2) if T_bkt >= 32768 else B_bkt)
                if icl_b_step == B_bkt:
                    raw_full = self._icl_only_compiled(reprs, y_bkt, train_sz)
                else:
                    parts = [
                        self._icl_only_compiled(reprs[i : i + icl_b_step], y_bkt[i : i + icl_b_step], train_sz)
                        for i in range(0, B_bkt, icl_b_step)
                    ]
                    raw_full = torch.cat(parts, dim=0)

            out = raw_full[:B_real, train_real : train_real + test_real]
            if self.model.max_classes > 0:
                out = out[..., :num_classes]
                if not return_logits:
                    out = torch.softmax(out / softmax_temperature, dim=-1)
            self._sync()
            elapsed = time.perf_counter() - t0

        compile_s = 0.0
        if is_first:
            self._seen_keys.add(key)
            compile_s = elapsed
            self.compile_events.append({"key": key, "first_call_s": round(compile_s, 4)})

        return out, compile_s

    def prefill_kv_bucketed(
        self,
        X_train: Tensor,
        y_train: Tensor,
        num_classes: int = 0,
        cache_dtype: Optional[torch.dtype] = None,
    ) -> tuple[BucketedKVCache, float]:
        """Compile & execute bucket-padded KV-cache prefill (`store_cache=True`)."""
        B_real, T_real, H_real = X_train.shape
        B_bkt = pick_bucket(B_real, BATCH_BUCKETS)
        T_bkt = pick_bucket(T_real, ROW_BUCKETS)
        H_bkt = pick_bucket(H_real, FEATURE_BUCKETS)

        eff_dtype = cache_dtype if cache_dtype is not None else self.kv_cache_dtype
        if eff_dtype == torch.float16:
            raise ValueError("fp16 KV cache is forbidden on TorchTPU.")

        X_bkt = torch.zeros((B_bkt, T_bkt, H_bkt), device=self.device, dtype=torch.float32)
        X_bkt[:B_real, :T_real, :H_real] = X_train.to(self.device, dtype=torch.float32)

        y_bkt = torch.zeros((B_bkt, T_bkt), device=self.device, dtype=torch.float32)
        y_bkt[:B_real, :T_real] = y_train.to(self.device, dtype=torch.float32)
        if B_bkt > B_real:
            X_bkt[B_real:] = X_bkt[0:1]
            y_bkt[B_real:] = y_bkt[0:1]

        d_vec = torch.full((B_bkt,), H_real, device=self.device, dtype=torch.long)
        train_sz = torch.tensor(T_real, device=self.device, dtype=torch.long)

        use_tiled = (B_bkt * T_bkt * (H_bkt + 4)) > self.max_fused_elements
        key = ("prefill_tiled" if use_tiled else "prefill_fused", B_bkt, T_bkt, H_bkt, str(eff_dtype), self.use_amp)
        is_first = key not in self._seen_keys

        n_col = len(self.model.col_embedder.tf_col.blocks)
        n_icl = len(self.model.icl_predictor.tf_icl.blocks)

        with torch.no_grad():
            t0 = time.perf_counter()
            if not use_tiled:
                prev_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision("highest")
                try:
                    pf_fn = (
                        self._prefill_bfloat16_compiled
                        if (self.use_amp and eff_dtype == torch.bfloat16)
                        else self._prefill_float32_compiled
                    )
                    flat_kv = pf_fn(X_bkt, y_bkt, d_vec, train_sz)
                finally:
                    torch.set_float32_matmul_precision(prev_prec)
                col_ks = tuple(flat_kv[:n_col])
                col_vs = tuple(flat_kv[n_col : 2 * n_col])
                icl_ks = tuple(flat_kv[2 * n_col : 2 * n_col + n_icl])
                icl_vs = tuple(flat_kv[2 * n_col + n_icl : 2 * n_col + 2 * n_icl])
            else:
                reprs, col_ks, col_vs = self._run_stage_tiled_col_row(
                    X_bkt, y_bkt, d_vec, train_sz, store_col_cache=True, cache_dtype=eff_dtype
                )
                icl_pf_fn = (
                    self._icl_prefill_bfloat16_compiled
                    if (self.use_amp and eff_dtype == torch.bfloat16)
                    else self._icl_prefill_float32_compiled
                )
                icl_b_step = 1 if T_bkt >= 65536 else (min(B_bkt, 2) if T_bkt >= 32768 else B_bkt)
                if icl_b_step == B_bkt:
                    icl_flat = icl_pf_fn(reprs, y_bkt, train_sz)
                    icl_ks = tuple(icl_flat[:n_icl])
                    icl_vs = tuple(icl_flat[n_icl : 2 * n_icl])
                else:
                    ks_acc: list[list[Tensor]] = [[] for _ in range(n_icl)]
                    vs_acc: list[list[Tensor]] = [[] for _ in range(n_icl)]
                    for i in range(0, B_bkt, icl_b_step):
                        sub_flat = icl_pf_fn(reprs[i : i + icl_b_step], y_bkt[i : i + icl_b_step], train_sz)
                        for li in range(n_icl):
                            ks_acc[li].append(sub_flat[li])
                            vs_acc[li].append(sub_flat[n_icl + li])
                    icl_ks = tuple(torch.cat(ks_acc[li], dim=0) for li in range(n_icl))
                    icl_vs = tuple(torch.cat(vs_acc[li], dim=0) for li in range(n_icl))

            self._sync()
            elapsed = time.perf_counter() - t0

        compile_s = 0.0
        if is_first:
            self._seen_keys.add(key)
            compile_s = elapsed
            self.compile_events.append({"key": key, "first_call_s": round(compile_s, 4)})

        cache = BucketedKVCache(
            col_keys=col_ks,
            col_values=col_vs,
            icl_keys=icl_ks,
            icl_values=icl_vs,
            real_batch=B_real,
            padded_batch=B_bkt,
            real_train_size=T_real,
            padded_train_size=T_bkt,
            real_features=H_real,
            padded_features=H_bkt,
            num_classes=num_classes,
            cache_dtype=eff_dtype,
        )
        return cache, compile_s

    def query_kv_bucketed(
        self,
        X_test: Tensor,
        cache: BucketedKVCache,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
    ) -> tuple[Tensor, float]:
        """Compile & execute bucket-padded KV-cache query (`use_cache=True`)."""
        B_real, T_te_real, H_real = X_test.shape
        B_bkt = cache.padded_batch
        H_bkt = cache.padded_features
        T_te_bkt = pick_bucket(T_te_real, TEST_ROW_BUCKETS)

        X_te_bkt = torch.zeros((B_bkt, T_te_bkt, H_bkt), device=self.device, dtype=torch.float32)
        X_te_bkt[:B_real, :T_te_real, :H_real] = X_test.to(self.device, dtype=torch.float32)
        if B_bkt > B_real:
            X_te_bkt[B_real:] = X_te_bkt[0:1]

        d_vec = torch.full((B_bkt,), cache.real_features, device=self.device, dtype=torch.long)
        train_sz = torch.tensor(cache.real_train_size, device=self.device, dtype=torch.long)

        key = (
            "query_fused",
            B_bkt,
            T_te_bkt,
            cache.padded_train_size,
            H_bkt,
            str(cache.cache_dtype),
            self.use_amp,
        )
        is_first = key not in self._seen_keys

        with torch.no_grad():
            t0 = time.perf_counter()
            prev_prec = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            try:
                icl_b_step = (
                    1
                    if (cache.padded_train_size >= 65536 or B_bkt * T_te_bkt * (H_bkt + 4) > self.max_fused_elements)
                    else B_bkt
                )
                if icl_b_step == B_bkt:
                    raw_out = self._query_compiled(
                        X_te_bkt,
                        d_vec,
                        train_sz,
                        *cache.col_keys,
                        *cache.col_values,
                        *cache.icl_keys,
                        *cache.icl_values,
                    )
                else:
                    parts = []
                    for i in range(0, B_bkt, icl_b_step):
                        ck = [t[i : i + icl_b_step] for t in cache.col_keys]
                        cv = [t[i : i + icl_b_step] for t in cache.col_values]
                        ik = [t[i : i + icl_b_step] for t in cache.icl_keys]
                        iv = [t[i : i + icl_b_step] for t in cache.icl_values]
                        parts.append(
                            self._query_compiled(
                                X_te_bkt[i : i + icl_b_step],
                                d_vec[i : i + icl_b_step],
                                train_sz,
                                *ck,
                                *cv,
                                *ik,
                                *iv,
                            )
                        )
                    raw_out = torch.cat(parts, dim=0)
            finally:
                torch.set_float32_matmul_precision(prev_prec)

            out = raw_out[:B_real, :T_te_real]
            if self.model.max_classes > 0:
                out = out[..., : cache.num_classes]
                if not return_logits:
                    out = torch.softmax(out / softmax_temperature, dim=-1)
            self._sync()
            elapsed = time.perf_counter() - t0

        compile_s = 0.0
        if is_first:
            self._seen_keys.add(key)
            compile_s = elapsed
            self.compile_events.append({"key": key, "first_call_s": round(compile_s, 4)})

        return out, compile_s


def attach_compiled_runner(
    estimator: Any,
    runner: Optional[BucketedCompiledTabICL] = None,
) -> BucketedCompiledTabICL:
    """Attach `BucketedCompiledTabICL` to a fitted or about-to-be-fitted `TabICLClassifier` / `TabICLRegressor`."""
    import types
    from tabicl import TabICLClassifier, TabICLRegressor

    is_cls = isinstance(estimator, TabICLClassifier)
    is_reg = isinstance(estimator, TabICLRegressor)
    if not (is_cls or is_reg):
        raise TypeError(f"Expected TabICLClassifier or TabICLRegressor, got {type(estimator)}")

    orig_fit = estimator.fit

    def _ensure_runner(est: Any) -> BucketedCompiledTabICL:
        nonlocal runner
        use_amp, _ = est._resolve_amp_fa3()
        kv_dt = est._resolve_kv_cache_dtype() or (torch.bfloat16 if use_amp else torch.float32)
        if runner is None or runner.model is not est.model_ or runner.use_amp != use_amp:
            runner = BucketedCompiledTabICL(
                model=est.model_,
                device=est.device_,
                use_amp=use_amp,
                kv_cache_dtype=kv_dt,
            )
        else:
            runner.kv_cache_dtype = kv_dt
        est._compiled_runner_ = runner
        return runner

    if is_cls:

        def _cls_build_kv_cache(self: Any) -> None:
            r = _ensure_runner(self)
            train_data = self.ensemble_generator_.transform(X=None, mode="train")
            self.model_kv_cache_ = OrderedDict()
            cache_dt = self._resolve_kv_cache_dtype() or r.kv_cache_dtype
            for norm_method, (Xs, ys) in train_data.items():
                X_t = torch.from_numpy(Xs).float()
                y_t = torch.from_numpy(ys).float()
                cache_obj, _ = r.prefill_kv_bucketed(
                    X_t, y_t, num_classes=self.n_classes_, cache_dtype=cache_dt
                )
                self.model_kv_cache_[norm_method] = cache_obj

        def _cls_batch_forward(
            self: Any, Xs: np.ndarray, ys: np.ndarray, feature_shuffles: Optional[np.ndarray] = None
        ) -> np.ndarray:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            y_t = torch.from_numpy(ys).float()
            out, _ = r.forward_nocache_bucketed(
                X_t,
                y_t,
                num_classes=self.n_classes_,
                return_logits=self.average_logits,
                softmax_temperature=self.softmax_temperature,
            )
            return out.float().cpu().numpy()

        def _cls_batch_forward_with_cache(
            self: Any, Xs: np.ndarray, kv_cache: BucketedKVCache
        ) -> np.ndarray:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            out, _ = r.query_kv_bucketed(
                X_t,
                cache=kv_cache,
                return_logits=self.average_logits,
                softmax_temperature=self.softmax_temperature,
            )
            return out.float().cpu().numpy()

        estimator._build_kv_cache = types.MethodType(_cls_build_kv_cache, estimator)
        estimator._batch_forward = types.MethodType(_cls_batch_forward, estimator)
        estimator._batch_forward_with_cache = types.MethodType(_cls_batch_forward_with_cache, estimator)

    else:

        def _reg_quantiles_to_stats(
            model_obj: Any, raw_q: Tensor, output_type: str | list[str], alphas: Optional[list[float]]
        ) -> dict[str, np.ndarray] | np.ndarray:
            dist = model_obj.quantile_dist(raw_q.float())
            mono_q = dist.quantiles
            out_types = [output_type] if isinstance(output_type, str) else list(output_type)
            res: dict[str, np.ndarray] = {}
            if "mean" in out_types:
                res["mean"] = mono_q.mean(dim=-1).cpu().numpy()
            if "variance" in out_types:
                res["variance"] = mono_q.var(dim=-1).cpu().numpy()
            if "median" in out_types:
                med = dist.icdf(alpha=torch.tensor(0.5, device=mono_q.device, dtype=mono_q.dtype))
                res["median"] = med.cpu().numpy()
            if "quantiles" in out_types:
                lvls = alphas if alphas is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                qs = dist.icdf(alpha=torch.tensor(lvls, device=mono_q.device, dtype=mono_q.dtype))
                res["quantiles"] = qs.cpu().numpy()
            if "raw_quantiles" in out_types:
                res["raw_quantiles"] = mono_q.cpu().numpy()
            if len(out_types) == 1:
                return res[out_types[0]]
            return res

        def _reg_build_kv_cache(self: Any) -> None:
            r = _ensure_runner(self)
            train_data = self.ensemble_generator_.transform(X=None, mode="train")
            self.model_kv_cache_ = OrderedDict()
            cache_dt = self._resolve_kv_cache_dtype() or r.kv_cache_dtype
            for norm_method, (Xs, ys) in train_data.items():
                X_t = torch.from_numpy(Xs).float()
                y_t = torch.from_numpy(ys).float()
                cache_obj, _ = r.prefill_kv_bucketed(
                    X_t, y_t, num_classes=0, cache_dtype=cache_dt
                )
                self.model_kv_cache_[norm_method] = cache_obj

        def _reg_batch_forward(
            self: Any,
            Xs: np.ndarray,
            ys: np.ndarray,
            output_type: str | list[str] = "mean",
            alphas: Optional[list[float]] = None,
        ) -> np.ndarray | dict[str, np.ndarray]:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            y_t = torch.from_numpy(ys).float()
            raw_q, _ = r.forward_nocache_bucketed(X_t, y_t, num_classes=0)
            return _reg_quantiles_to_stats(self.model_, raw_q, output_type, alphas)

        def _reg_batch_forward_with_cache(
            self: Any,
            Xs: np.ndarray,
            kv_cache: BucketedKVCache,
            output_type: str | list[str] = "mean",
            alphas: Optional[list[float]] = None,
        ) -> np.ndarray | dict[str, np.ndarray]:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            raw_q, _ = r.query_kv_bucketed(X_t, cache=kv_cache)
            return _reg_quantiles_to_stats(self.model_, raw_q, output_type, alphas)

        estimator._build_kv_cache = types.MethodType(_reg_build_kv_cache, estimator)
        estimator._batch_forward = types.MethodType(_reg_batch_forward, estimator)
        estimator._batch_forward_with_cache = types.MethodType(_reg_batch_forward_with_cache, estimator)

    if getattr(estimator, "model_", None) is not None and getattr(estimator, "device_", None) is not None:
        return _ensure_runner(estimator)

    def _wrapped_fit(self: Any, X: Any, y: Any) -> Any:
        out = orig_fit(X, y)
        _ensure_runner(self)
        return out

    estimator.fit = types.MethodType(_wrapped_fit, estimator)
    return runner  # type: ignore[return-value]
