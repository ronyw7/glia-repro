"""Synthesize an A10 compute profile for Meta-Llama-3-8B for Vidur.

Vidur ships no A10 profile, and Llama-3-8B is only profiled on A100/H100.
We transfer the measured A100 Llama-3-8B profile to A10 in two steps:

  1. A100 -> A40: multiply each operator by the *measured* A40/A100 time ratio
     from Vidur's Llama-2-7B profiles (both devices profiled on the same grid).
     The ratio is looked up as a function of the operator's "work" (tokens,
     KV bytes read, attention FLOPs), so memory-bound regimes get ~2.9x
     (HBM2e 2.0 TB/s vs GDDR6 0.7 TB/s) and compute-bound regimes ~2.0x.
  2. A40 -> A10: multiply by a constant hardware factor (default 1.2):
     A10 has 600 GB/s vs 696 GB/s (1.16x) and 125 vs 150 dense FP16 TFLOPS (1.2x).

Writes <out_dir>/mlp.csv and <out_dir>/attention.csv in Vidur's format.
"""
import argparse
import os

import numpy as np
import pandas as pd

MLP_OPS = [
    "attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj", "mlp_act",
    "input_layernorm", "post_attention_layernorm", "attn_rope", "add", "emb",
]
STATS = ["min", "max", "mean", "median"]


def ratio_curve(work_src, t_a100, t_a40, nbins=40):
    """Median A40/A100 ratio as a function of log2(work), returned as (x, y) for np.interp."""
    df = pd.DataFrame({"w": np.log2(np.maximum(work_src, 1)), "r": t_a40 / t_a100})
    df = df[np.isfinite(df.r) & (df.r > 0)]
    bins = np.linspace(df.w.min(), df.w.max() + 1e-9, nbins + 1)
    df["b"] = np.digitize(df.w, bins)
    g = df.groupby("b").agg(w=("w", "median"), r=("r", "median")).dropna()
    # smooth a little with a rolling median to remove kernel-selection noise
    g["r"] = g.r.rolling(3, center=True, min_periods=1).median()
    return g.w.values, g.r.values


def apply_ratio(df, cols, work_dst, curve, hw_factor):
    x, y = curve
    r = np.interp(np.log2(np.maximum(work_dst, 1)), x, y) * hw_factor
    for c in cols:
        if c in df.columns:
            df[c] = df[c] * r
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidur", required=True, help="path to vidur checkout")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--a40_to_a10", type=float, default=1.2)
    args = ap.parse_args()

    prof = os.path.join(args.vidur, "data/profiling/compute")
    src_a100 = os.path.join(prof, "a100/meta-llama/Llama-2-7b-hf")
    src_a40 = os.path.join(prof, "a40/meta-llama/Llama-2-7b-hf")
    tgt_a100 = os.path.join(prof, "a100/meta-llama/Meta-Llama-3-8B")

    # ---------------- MLP / linear / elementwise ops ----------------
    a = pd.read_csv(os.path.join(src_a100, "mlp.csv"))
    b = pd.read_csv(os.path.join(src_a40, "mlp.csv"))
    key = ["num_tokens", "num_tensor_parallel_workers"]
    m = a.merge(b, on=key, suffixes=("_a100", "_a40"))
    m = m[m.num_tensor_parallel_workers == 1]
    t = t_mlp = pd.read_csv(os.path.join(tgt_a100, "mlp.csv"))
    out_mlp = t.copy()
    for op in MLP_OPS:
        med = f"time_stats.{op}.median"
        if med + "_a100" not in m.columns or med not in t.columns:
            continue
        curve = ratio_curve(m.num_tokens.values * m.num_tensor_parallel_workers.values,
                            m[med + "_a100"].values, m[med + "_a40"].values)
        if op == "add":
            add_curve = curve
        cols = [f"time_stats.{op}.{s}" for s in STATS]
        # per-GPU work ~ tokens / tp for the linear layers
        out_mlp = apply_ratio(out_mlp, cols, out_mlp.num_tokens.values, curve, args.a40_to_a10)
        out_mlp[f"time_stats.{op}.std"] = t[f"time_stats.{op}.std"] * (
            out_mlp[med] / t[med].replace(0, np.nan)).fillna(1)

    # ---------------- attention ----------------
    a = pd.read_csv(os.path.join(src_a100, "attention.csv"))
    b = pd.read_csv(os.path.join(src_a40, "attention.csv"))
    key = ["batch_size", "prefill_chunk_size", "kv_cache_size", "num_tensor_parallel_workers", "block_size"]
    m = a.merge(b, on=key, suffixes=("_a100", "_a40"))
    m = m[(m.num_tensor_parallel_workers == 1) & (m.block_size == 16)]
    L2_KV, L2_Q = 32, 32  # Llama-2-7B heads
    t = pd.read_csv(os.path.join(tgt_a100, "attention.csv"))
    out_att = t.copy()
    kv_t = t.n_kv_head / t.num_tensor_parallel_workers
    q_t = t.n_q_head / t.num_tensor_parallel_workers

    dec_s = m.prefill_chunk_size == 0
    dec_t = t.prefill_chunk_size == 0

    # decode attention: memory bound, work = KV bytes read ~ bs * kv_len * n_kv_heads
    md = m[dec_s]
    curve = ratio_curve(md.batch_size * md.kv_cache_size * L2_KV,
                        md["time_stats.attn_decode.median_a100"], md["time_stats.attn_decode.median_a40"])
    cols = [f"time_stats.attn_decode.{s}" for s in STATS + ["std"]]
    sub = out_att[dec_t].copy()
    sub = apply_ratio(sub, cols, (t.batch_size * t.kv_cache_size * kv_t)[dec_t].values, curve, args.a40_to_a10)
    out_att.loc[dec_t, cols] = sub[cols].values

    # prefill attention: work = attention FLOPs ~ chunk * (kv + chunk/2) * n_q_heads
    mp = m[~dec_s]
    curve = ratio_curve(mp.prefill_chunk_size * (mp.kv_cache_size + mp.prefill_chunk_size / 2) * L2_Q,
                        mp["time_stats.attn_prefill.median_a100"], mp["time_stats.attn_prefill.median_a40"])
    cols = [f"time_stats.attn_prefill.{s}" for s in STATS + ["std"]]
    sub = out_att[~dec_t].copy()
    sub = apply_ratio(sub, cols, (t.prefill_chunk_size * (t.kv_cache_size + t.prefill_chunk_size / 2) * q_t)[~dec_t].values,
                      curve, args.a40_to_a10)
    out_att.loc[~dec_t, cols] = sub[cols].values

    # KV-cache save + reshapes: memory-bound copies, work ~ tokens.
    # Vidur's Llama-2 profiles (FlashAttention backend) record these as 0, so there is no
    # measured A40/A100 ratio for them; use the ratio of the elementwise `add` kernel
    # (also a pure memory op) from the MLP profiles as a function of token count.
    for op in ["attn_kv_cache_save", "attn_input_reshape", "attn_output_reshape"]:
        med = f"time_stats.{op}.median"
        if med not in t.columns or (t[med] == 0).all():
            continue
        ntok_t = np.maximum(t.prefill_chunk_size, t.batch_size)
        cols = [f"time_stats.{op}.{s}" for s in STATS + ["std"]]
        out_att = apply_ratio(out_att, cols, ntok_t.values, add_curve, args.a40_to_a10)

    os.makedirs(args.out_dir, exist_ok=True)
    out_mlp.to_csv(os.path.join(args.out_dir, "mlp.csv"), index=False)
    out_att.to_csv(os.path.join(args.out_dir, "attention.csv"), index=False)

    # report effective A10/A100 slowdowns at a few operating points
    tp1 = out_mlp.num_tensor_parallel_workers == 1
    for n in [1, 32, 128, 512, 2048, 8192]:
        r = out_mlp[tp1 & (out_mlp.num_tokens == n)]
        r0 = t_mlp[(t_mlp.num_tensor_parallel_workers == 1) & (t_mlp.num_tokens == n)]
        if len(r):
            tot = sum(r[f"time_stats.{o}.median"].iloc[0] for o in ["attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj"])
            tot0 = sum(r0[f"time_stats.{o}.median"].iloc[0] for o in ["attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj"])
            print(f"linear layers @ {n:5d} tokens: A100 {tot0:7.3f} ms -> A10 {tot:7.3f} ms  (x{tot / tot0:.2f})")


if __name__ == "__main__":
    main()
