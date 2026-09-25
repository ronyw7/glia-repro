# Reproducing Glia's LLM request-routing results in Vidur

This repo reproduces the request-routing experiment from
**Glia: A Human-Inspired AI for Automated Systems Design and Optimization**
(Hamadanian et al., CAIS '26, [arXiv 2510.27176](https://arxiv.org/abs/2510.27176)).
The authors did not release code, but the paper gives the full simulation setup
and the full source of the routers it discusses, so both can be rebuilt on top of
[microsoft/vidur](https://github.com/microsoft/vidur).

RESULTS_PLACEHOLDER

## What was rebuilt, and how faithfully

| Paper (Sec. 5, Figs. 12-15) | This repo | Confidence |
|---|---|---|
| Vidur simulator | `microsoft/vidur` @ `abae7f6` + `patches/vidur-glia.patch` | exact code base |
| Router called on every arrival **and every completion** (Fig. 14/15 prompt) | patch: `BatchEndEvent` emits a `GlobalScheduleEvent` when a request finishes | exact semantics |
| Replicas expose `pending_queue`, `active_queue`, `num_active_requests`, `block_size`, `num_blocks` (used by Figs. 12/13) | patch: `BaseReplicaScheduler` | exact semantics |
| LLQ baseline = fewest in-flight (pending+active); LOR = fewest waiting | `llq` (new), `lor` (Vidur's) | exact |
| vLLM-style memory: incremental KV blocks, evict youngest on OOM, restart from scratch | Vidur's Sarathi scheduler (unchanged) | exact |
| Sarathi chunked prefill, chunk 8192 | `SarathiSchedulerConfig(chunk_size=8192)` | exact |
| 4 x NVIDIA A10, Llama-3-8B-Instruct | **A10 profile is synthesized** (see below); Vidur's `Meta-Llama-3-8B` architecture | approximate |
| ShareGPT (`anon8231489123/ShareGPT_Vicuna_unfiltered`) | same file (V3 cleaned split), Llama-3 tokenizer | exact data; request construction is a choice (see below) |
| 5% prompts and 5% decodes inflated 10x, independently | `scripts/gen_workload.py` | exact |
| 7.5 QPS, log-normal inter-arrivals, sigma = 2 | `scripts/gen_workload.py` (mean gap = 1/7.5 s) | exact |
| 1000 s benchmark, 10 seeds, 90% bootstrap CI | seeds 0-9, arrivals over 1000 s, run to completion | exact |
| HRA router (Fig. 12) | `routers/glia_hra.py` - verbatim | exact |
| FunSearch router (Fig. 13) | `routers/funsearch_fig13.py` - verbatim (+imports) | exact |
| Expert router | not published | - |

### Unavoidable approximations

1. **A10 compute profile.** Vidur ships no A10 profile, and Llama-3-8B is only profiled on A100.
   `scripts/make_a10_profile.py` transfers the A100 Llama-3-8B profile per operator using the
   *measured* A40/A100 ratio from Vidur's Llama-2-7B profiles (looked up by operator work, so
   memory-bound ops get ~2.9x and compute-bound ops ~2.0x), then a 1.2x A40 -> A10 hardware factor
   (600 vs 696 GB/s, 125 vs 150 TFLOPS). Sanity check: batch-1 decode ~31 ms/token and
   8K-token prefill ~1.5 s, consistent with an A10 running Llama-3-8B in fp16.
2. **KV-cache size.** Vidur's memory planner is used as-is (24 GB, 10% margin). Vidur's parameter
   counter omits the 128K-vocab embedding/LM head, so each replica gets 4096 blocks (65K tokens) -
   roughly 2x what vLLM would leave on a real 24 GB A10. A sensitivity run with a realistic KV
   budget is included.
3. **ShareGPT -> requests.** The paper does not say how conversations became requests. Two
   readings are provided: `first_turn` (vLLM benchmark convention: first user message -> first
   reply) and `multi_turn` (every reply is a request whose prompt is the whole history; Sarathi /
   Vidur convention). Hugging Face is not reachable from the build environment; the identical file
   (sha256 `35f0e213...79ba4`) was fetched from a public Git LFS mirror.

## Layout

```
patches/vidur-glia.patch     all simulator changes (LLQ, custom-router loader, completion trigger,
                             pending/active queues, A10 SKU, request telemetry)
scripts/setup.sh             clone Vidur @ pinned commit, apply patch, build A10 profile, warm cache
scripts/make_a10_profile.py  A10 profile synthesis
scripts/tokenize_sharegpt.py ShareGPT -> per-turn Llama-3 token counts
scripts/make_length_tables.py / gen_workload.py   request lengths -> 10-seed traces
data/lengths/                (prompt, decode) length tables derived from ShareGPT
data/traces/<workload>/      the exact traces used (seeds 0-9)
routers/                     glia_hra.py, funsearch_fig13.py, ablation_hra_fifo.py, template_router.py
glia_repro/run.py            run routers x seeds in parallel; prints per-router summary
glia_repro/report.py         aggregate -> results/<tag>/summary.{csv,md}, mean_rt.png
results/                     summaries + per-request CSVs for every run reported here
```

## Testing your own router

```bash
./scripts/setup.sh                                   # once (~30 min: trains Vidur's runtime predictors)
cp routers/template_router.py routers/my_router.py   # implement schedule()
python -m glia_repro.run --routers routers/my_router.py llq routers/glia_hra.py --workload first_turn
python -m glia_repro.report --tag first_turn
```

`template_router.py` documents exactly what state a router may read. As in the paper, a router
may **not** use `request.num_decode_tokens` (the oracle output length).
