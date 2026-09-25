"""Benchmark routers over the 10-seed Glia workload.

Examples
  python -m glia_repro.run --routers llq rr lor routers/glia_hra.py            # paper-calibrated setting
  python -m glia_repro.run --routers routers/my_router.py --seeds 0 1 2       # quick check of a new router
  python -m glia_repro.run --routers llq routers/glia_hra.py --num_blocks 4096 # Vidur-default memory
"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

from glia_repro.sim import REPO


def _router_name(r: str) -> str:
    return os.path.splitext(os.path.basename(r))[0] if r.endswith(".py") else r


def _job(args):
    router, trace, out_dir, env, kw = args
    from glia_repro.sim import run_one
    if os.path.exists(os.path.join(out_dir, "summary.json")) and not kw.pop("force", False):
        return json.load(open(os.path.join(out_dir, "summary.json")))
    kw.pop("force", None)
    try:
        return run_one(router, trace, out_dir, env=env, **kw)
    except Exception as e:  # keep the sweep going
        import traceback
        traceback.print_exc()
        return {"router": router, "trace": trace, "error": repr(e)}


def bootstrap_ci(x, level=0.90, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", required=True, help="rr|llq|lor|random or path/to/router.py")
    ap.add_argument("--workload", default="first_turn", help="subdir of data/traces")
    ap.add_argument("--qps", default="7.5")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--tag", default=None, help="results/<tag>/ (default: <workload>_kv<num_blocks>)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env vars for routers (e.g. HRA_R=0.4)")
    ap.add_argument("--num_blocks", type=int, default=3328,
                    help="KV-cache blocks per replica. 3328 = calibrated so LLQ matches the paper's baseline; "
                         "4096 = Vidur's memory planner for A10/24GB; 2048 ~ real vLLM on a 24GB A10")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"{args.workload}_kv{args.num_blocks}"
    env = dict(kv.split("=", 1) for kv in args.env)
    env_sfx = ("__" + "_".join(f"{k}{v}" for k, v in sorted(env.items()))) if env else ""
    jobs = []
    for router, seed in itertools.product(args.routers, args.seeds):
        trace = os.path.join(REPO, "data", "traces", args.workload,
                             f"sharegpt_{args.workload}_{args.qps}_seed{seed}.csv")
        assert os.path.exists(trace), trace
        out_dir = os.path.join(REPO, "results", tag, _router_name(router) + env_sfx, f"seed{seed}")
        kw = {"force": args.force, "num_blocks": args.num_blocks}
        jobs.append((router, trace, out_dir, env, kw))

    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
        results = []
        for r in pool.imap_unordered(_job, jobs):
            results.append(r)
            if "error" in r:
                print(f"[ERR] {r['router']} {os.path.basename(r['trace'])}: {r['error']}", flush=True)
            else:
                print(f"{_router_name(r['router']):>24s} {os.path.basename(r['trace']):>40s}  "
                      f"mean RT {r['mean_e2e']:7.2f}s  p99 {r['p99_e2e']:7.1f}s  restarted {r['frac_restarted']:.2f}  "
                      f"done {r['num_completed']}/{r['num_requests']}  ({r['wall_seconds']:.0f}s)", flush=True)

    ok = [r for r in results if "error" not in r]
    df = pd.DataFrame(ok)
    if len(df):
        df["router"] = df.router.map(_router_name) + env_sfx
        rows = []
        for name, g in df.groupby("router"):
            lo, hi = bootstrap_ci(g.mean_e2e)
            rows.append({"router": name, "seeds": len(g), "mean_RT": g.mean_e2e.mean(), "ci90_lo": lo, "ci90_hi": hi,
                         "p99_RT": g.p99_e2e.mean(), "mean_TTFT": g.mean_ttft.mean(),
                         "global_q_delay": g.mean_global_queue_delay.mean(), "frac_restarted": g.frac_restarted.mean(),
                         "incomplete": int((g.num_requests - g.num_completed).sum())})
        print()
        print(pd.DataFrame(rows).sort_values("mean_RT").to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
