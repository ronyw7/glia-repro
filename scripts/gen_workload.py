"""Generate Glia-paper style traces (arrived_at, num_prefill_tokens, num_decode_tokens).

Paper (Sec. 5): ShareGPT on 4x A10 / Llama-3-8B-Instruct; independently inflate 5% of
decode lengths and 5% of prompt lengths by 10x; 7.5 QPS with log-normal (sigma=2)
inter-arrival times; 1000-second benchmark; ten random seeds.
"""
import argparse
import os

import numpy as np
import pandas as pd


def make_trace(lengths: pd.DataFrame, qps: float, duration: float, sigma: float,
               inflate_frac: float, inflate_factor: float, max_tokens: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean_gap = 1.0 / qps
    mu = np.log(mean_gap) - sigma**2 / 2  # so that E[gap] = 1/qps
    gaps = []
    t = 0.0
    while True:
        chunk = rng.lognormal(mu, sigma, size=4096)
        for g in chunk:
            t += g
            if t > duration:
                break
            gaps.append(t)
        if t > duration:
            break
    arrivals = np.array(gaps) - gaps[0]  # first request at t=0
    n = len(arrivals)

    idx = rng.integers(0, len(lengths), size=n)
    p = lengths.num_prefill_tokens.values[idx].astype(np.int64)
    d = lengths.num_decode_tokens.values[idx].astype(np.int64)
    p_inf = rng.random(n) < inflate_frac
    d_inf = rng.random(n) < inflate_frac
    p = np.where(p_inf, p * inflate_factor, p).astype(np.int64)
    d = np.where(d_inf, d * inflate_factor, d).astype(np.int64)

    # respect the model context: cap decode first (leave room for a prompt), then trim prompt
    d = np.minimum(d, max_tokens - 16)
    p = np.clip(p, 1, max_tokens - d)

    return pd.DataFrame({
        "arrived_at": arrivals,
        "num_prefill_tokens": p,
        "num_decode_tokens": d,
        "prefill_inflated": p_inf.astype(int),
        "decode_inflated": d_inf.astype(int),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default="sharegpt")
    ap.add_argument("--qps", type=float, default=7.5)
    ap.add_argument("--duration", type=float, default=1000.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--inflate_frac", type=float, default=0.05)
    ap.add_argument("--inflate_factor", type=float, default=10.0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()

    lengths = pd.read_csv(args.lengths)
    os.makedirs(args.out_dir, exist_ok=True)
    for s in args.seeds:
        df = make_trace(lengths, args.qps, args.duration, args.sigma, args.inflate_frac,
                        args.inflate_factor, args.max_tokens, s)
        path = os.path.join(args.out_dir, f"{args.name}_{args.qps:g}_seed{s}.csv")
        df.to_csv(path, index=False)
        print(f"seed {s}: {len(df)} reqs, mean prefill {df.num_prefill_tokens.mean():.0f}, "
              f"mean decode {df.num_decode_tokens.mean():.0f}, last arrival {df.arrived_at.max():.0f}s -> {path}")


if __name__ == "__main__":
    main()
