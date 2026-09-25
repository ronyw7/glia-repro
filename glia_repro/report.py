"""Aggregate results/<tag>/<router>/seed*/summary.json into a table and a bar chart."""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from glia_repro.run import bootstrap_ci
from glia_repro.sim import REPO

# Paper values (10-seed means, Glia Fig. 3b / Fig. 2 reference lines, Fig. 10 for HRA @ r=0.6)
PAPER_RT = {"rr": 63.4, "lor": 59.6, "llq": 54.6, "funsearch_fig13": 32.5, "glia_hra": 24.1}

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
             axis="#c3c2b7", ours="#2a78d6", paper="#eb6834")


def load(tag):
    rows = []
    for f in glob.glob(os.path.join(REPO, "results", tag, "*", "seed*", "summary.json")):
        s = json.load(open(f))
        s["router_name"] = f.split(os.sep)[-3]
        s["seed"] = int(f.split(os.sep)[-2][4:])
        rows.append(s)
    return pd.DataFrame(rows)


def table(df):
    out = []
    for name, g in df.groupby("router_name"):
        lo, hi = bootstrap_ci(g.mean_e2e)
        out.append({
            "router": name, "seeds": len(g), "mean_RT_s": g.mean_e2e.mean(), "ci90": f"[{lo:.1f}, {hi:.1f}]",
            "p99_RT_s": g.p99_e2e.mean(), "mean_TTFT_s": g.mean_ttft.mean(),
            "global_queue_s": g.mean_global_queue_delay.mean(), "restarted_frac": g.frac_restarted.mean(),
            "incomplete": int((g.num_requests - g.num_completed).sum()),
            "paper_RT_s": PAPER_RT.get(name, np.nan),
        })
    t = pd.DataFrame(out).sort_values("mean_RT_s", ascending=False)
    llq = t.set_index("router").mean_RT_s.get("llq", np.nan)
    t["speedup_vs_LLQ"] = llq / t.mean_RT_s
    return t


def paired(df, baseline):
    """Per-seed paired comparison against `baseline` (same traces => paired samples)."""
    b = df[df.router_name == baseline].set_index("seed").mean_e2e
    rows = []
    for name, g in df.groupby("router_name"):
        if name == baseline:
            continue
        a = g.set_index("seed").mean_e2e
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        ratio = (a[common] / b[common]).values  # < 1 means faster than baseline
        lo, hi = np.exp(bootstrap_ci(np.log(ratio)))
        rows.append({"router": name, f"RT / {baseline} (geo-mean)": float(np.exp(np.log(ratio).mean())),
                     "ci90": f"[{lo:.3f}, {hi:.3f}]",
                     f"seeds faster than {baseline}": f"{int((ratio < 1).sum())}/{len(ratio)}"})
    return pd.DataFrame(rows).sort_values(f"RT / {baseline} (geo-mean)") if rows else pd.DataFrame()


def plot_sweep(df, path, title):
    """Fig.-10-style panels: HRA mean RT vs decode-to-prefill ratio r and vs safety margin m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = LIGHT
    panels = []
    for key, label, default in [("HRA_R", "decode-to-prefill ratio r", 0.6), ("HRA_M", "safety margin m (fraction of blocks)", 0.03)]:
        pts = []
        for name, g in df.groupby("router_name"):
            if name == "glia_hra":
                v = default
            elif name.startswith(f"glia_hra__{key}"):
                v = float(name[len(f"glia_hra__{key}"):])
            else:
                continue
            lo, hi = bootstrap_ci(g.mean_e2e)
            pts.append((v, g.mean_e2e.mean(), lo, hi, g.frac_restarted.mean()))
        if len(pts) > 1:
            panels.append((label, default, sorted(pts)))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(c["surface"])
    for ax, (label, default, pts) in zip(axes, panels):
        x, m, lo, hi, rs = map(np.array, zip(*pts))
        ax.set_facecolor(c["surface"])
        ax.fill_between(x, lo, hi, color=c["ours"], alpha=0.10, linewidth=0)
        ax.plot(x, m, color=c["ours"], linewidth=2, marker="o", markersize=6, markeredgecolor=c["surface"],
                markeredgewidth=2)
        for xi, mi, ri in zip(x, m, rs):
            ax.annotate(f"{ri:.0%} restarted", (xi, mi), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=7, color=c["muted"])
        ax.axvline(default, color=c["axis"], linewidth=1)
        ax.text(default, 0.02, " paper default", transform=ax.get_xaxis_transform(), fontsize=7, color=c["muted"])
        ax.set_xlabel(label, color=c["ink2"], fontsize=9)
        ax.set_ylim(0, max(hi) * 1.25)
        ax.grid(axis="y", color=c["grid"], linewidth=1)
        ax.set_axisbelow(True)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(c["axis"])
        ax.tick_params(colors=c["muted"], length=0, labelsize=8)
    axes[0].set_ylabel("HRA mean RT (s), 10 seeds, 90% CI", color=c["ink2"], fontsize=9)
    fig.suptitle(title, color=c["ink"], fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=c["surface"])
    plt.close(fig)


def plot(df, t, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = LIGHT
    order = list(t.router)
    means, errs = [], [[], []]
    for name in order:
        g = df[df.router_name == name]
        lo, hi = bootstrap_ci(g.mean_e2e)
        m = g.mean_e2e.mean()
        means.append(m)
        errs[0].append(m - lo)
        errs[1].append(hi - m)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(order) + 1.4), dpi=150)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ax.barh(y, means, height=0.45, color=c["ours"], label="This reproduction (10-seed mean, 90% CI)",
            xerr=errs, error_kw=dict(ecolor=c["ink2"], elinewidth=1, capsize=3))
    paper = [PAPER_RT.get(n, np.nan) for n in order]
    ax.scatter(paper, y, marker="D", s=40, color=c["paper"], edgecolor=c["surface"], linewidth=2, zorder=3,
               label="Glia paper")
    for yi, m in zip(y, means):  # value inside the bar, at its base (clear of CI whiskers / markers)
        ax.text(max(means) * 0.015, yi, f"{m:.1f}s", va="center", ha="left", fontsize=8.5,
                color="white", fontweight="bold")
    ax.set_yticks(y, order, color=c["ink"])
    ax.set_xlabel("Mean request completion time (s) - lower is better", color=c["ink2"])
    ax.set_title(title, color=c["ink"], fontsize=11, loc="left", pad=28)
    ax.grid(axis="x", color=c["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(colors=c["muted"], length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8,
              labelcolor=c["ink2"], borderaxespad=0.2, handletextpad=0.4, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=c["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--paired", nargs="*", default=["llq", "glia_hra"], help="baselines for paired comparison")
    args = ap.parse_args()
    df = load(args.tag)
    t = table(df)
    out = os.path.join(REPO, "results", args.tag)
    t.to_csv(os.path.join(out, "summary.csv"), index=False)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(t.to_markdown(index=False, floatfmt=".2f"))
    main_df = df[~df.router_name.str.contains("__")]  # parameter-sweep variants go in their own chart
    plot(main_df, t[t.router.isin(set(main_df.router_name))], os.path.join(out, "mean_rt.png"),
         args.title or f"Mean RT, workload: {args.tag}")
    plot_sweep(df, os.path.join(out, "hra_sensitivity.png"), "HRA parameter sensitivity (cf. Glia Fig. 10)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    with open(os.path.join(out, "summary.md"), "a") as f:
        for base in args.paired:
            if base in set(df.router_name):
                p = paired(df, base)
                print(f"\nPaired vs {base}:\n" + p.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
                f.write(f"\n\nPaired per-seed comparison vs `{base}`:\n\n" + p.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
