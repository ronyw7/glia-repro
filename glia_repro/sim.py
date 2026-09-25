"""Run one Vidur simulation in the Glia paper's setup and summarize it.

Setup (Glia, Sec. 5): 4 replicas of Llama-3-8B(-Instruct) on A10 GPUs, TP=1,
Sarathi (chunked-prefill) replica scheduler with chunk size 8192, trace replay.
"""
import atexit
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDUR = os.path.join(REPO, "third_party", "vidur")
if VIDUR not in sys.path:
    sys.path.insert(0, VIDUR)

BUILTIN_ROUTERS = {"rr": "round_robin", "round_robin": "round_robin", "llq": "llq",
                   "lor": "lor", "random": "random"}


def build_config(router: str, trace_file: str, out_dir: str, *, num_replicas: int = 4,
                 device: str = "a10", network_device: str = "a10_pcie",
                 model: str = "meta-llama/Meta-Llama-3-8B", chunk_size: int = 8192,
                 max_tokens: int = 8192, batch_size_cap: int = 128,
                 num_blocks: Optional[int] = None, memory_margin_fraction: float = 0.1,
                 cache_dir: Optional[str] = None, seed: int = 42):
    from vidur.config import (ClusterConfig, CustomGlobalSchedulerConfig, LLQGlobalSchedulerConfig,
                              LORGlobalSchedulerConfig, MetricsConfig, RandomForrestExecutionTimePredictorConfig,
                              RandomGlobalSchedulerConfig, ReplicaConfig, RoundRobinGlobalSchedulerConfig,
                              SarathiSchedulerConfig, SimulationConfig, TraceRequestGeneratorConfig)

    if router.endswith(".py"):
        gs = CustomGlobalSchedulerConfig(module_path=os.path.abspath(router))
    else:
        gs = {"round_robin": RoundRobinGlobalSchedulerConfig, "llq": LLQGlobalSchedulerConfig,
              "lor": LORGlobalSchedulerConfig, "random": RandomGlobalSchedulerConfig}[BUILTIN_ROUTERS[router]]()

    prof = os.path.join(VIDUR, "data", "profiling")
    predictor = RandomForrestExecutionTimePredictorConfig(
        compute_input_file=os.path.join(prof, "compute/{DEVICE}/{MODEL}/mlp.csv"),
        attention_input_file=os.path.join(prof, "compute/{DEVICE}/{MODEL}/attention.csv"),
        all_reduce_input_file=os.path.join(prof, "network/{NETWORK_DEVICE}/all_reduce.csv"),
        send_recv_input_file=os.path.join(prof, "network/{NETWORK_DEVICE}/send_recv.csv"),
        cpu_overhead_input_file=os.path.join(prof, "cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"),
        prediction_max_prefill_chunk_size=chunk_size,
        prediction_max_tokens_per_request=max_tokens,
        prediction_max_batch_size=batch_size_cap,
    )
    config = SimulationConfig(
        seed=seed,
        log_level="warning",
        cluster_config=ClusterConfig(
            num_replicas=num_replicas,
            replica_config=ReplicaConfig(model_name=model, device=device, network_device=network_device,
                                         memory_margin_fraction=memory_margin_fraction),
            global_scheduler_config=gs,
            replica_scheduler_config=SarathiSchedulerConfig(chunk_size=chunk_size, batch_size_cap=batch_size_cap,
                                                            num_blocks=num_blocks),
        ),
        request_generator_config=TraceRequestGeneratorConfig(trace_file=trace_file, max_tokens=max_tokens),
        execution_time_predictor_config=predictor,
        metrics_config=MetricsConfig(write_metrics=False, enable_chrome_trace=False, store_plots=False,
                                     output_dir=out_dir,
                                     cache_dir=cache_dir or os.path.join(REPO, "cache", "vidur")),
    )
    return config


def summarize(requests, sim_time: float, wall: float, scheduler) -> Dict:
    done = [r for r in requests if r.completed]
    e2e = np.array([r.completed_at - r.arrived_at for r in done])
    ttft = np.array([r._prefill_completed_at - r.arrived_at for r in done])
    routed = np.array([getattr(r, "_routed_at", np.nan) - r.arrived_at for r in done])
    restarts = np.array([r.num_restarts for r in done])
    sched = np.array([r._scheduled_at - r.arrived_at for r in done])  # arrival -> first time on a GPU
    replicas = scheduler._replica_schedulers
    return {
        "num_requests": len(requests),
        "num_completed": len(done),
        "mean_e2e": float(e2e.mean()) if len(e2e) else float("nan"),
        "p50_e2e": float(np.percentile(e2e, 50)) if len(e2e) else float("nan"),
        "p90_e2e": float(np.percentile(e2e, 90)) if len(e2e) else float("nan"),
        "p99_e2e": float(np.percentile(e2e, 99)) if len(e2e) else float("nan"),
        "mean_ttft": float(ttft.mean()) if len(ttft) else float("nan"),
        "p90_ttft": float(np.percentile(ttft, 90)) if len(ttft) else float("nan"),
        "mean_global_queue_delay": float(np.nanmean(routed)) if len(routed) else float("nan"),
        "mean_wait_before_gpu": float(sched.mean()) if len(sched) else float("nan"),
        "mean_service": float((e2e - sched).mean()) if len(e2e) else float("nan"),
        "frac_restarted": float((restarts > 0).mean()) if len(restarts) else float("nan"),
        "mean_restarts": float(restarts.mean()) if len(restarts) else float("nan"),
        "makespan": float(sim_time),
        "num_blocks_per_replica": int(next(iter(replicas.values()))._config.num_blocks),
        "wall_seconds": wall,
    }


class _ScaledExecutionTime:
    """Stand-in for vidur's ExecutionTime with total/model time multiplied by a constant."""

    def __init__(self, et, scale: float):
        self._et, self.total_time, self.model_time = et, et.total_time * scale, et.model_time * scale

    def __getattr__(self, name):
        return getattr(self._et, name)


def _apply_diagnostic_knobs():
    """Env-var knobs for sensitivity experiments (not part of the paper setup):
    GPU_TIME_SCALE  multiply every batch's execution time (0.8 = a 25% faster GPU)
    KVSAVE_SCALE    multiply the attn_kv_cache_save op (its A100 source measurements look anomalous)
    """
    from vidur.execution_time_predictor.base_execution_time_predictor import BaseExecutionTimePredictor
    from vidur.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor

    gpu = float(os.environ.get("GPU_TIME_SCALE", 1.0))
    kvs = float(os.environ.get("KVSAVE_SCALE", 1.0))
    if kvs != 1.0:
        orig_kv = SklearnExecutionTimePredictor._get_attention_kv_cache_save_execution_time
        SklearnExecutionTimePredictor._get_attention_kv_cache_save_execution_time = (
            lambda self, batch: kvs * orig_kv(self, batch))
    if gpu != 1.0:
        orig = BaseExecutionTimePredictor.get_execution_time
        BaseExecutionTimePredictor.get_execution_time = (
            lambda self, batch, stage: _ScaledExecutionTime(orig(self, batch, stage), gpu))


def run_one(router: str, trace_file: str, out_dir: str, env: Optional[Dict[str, str]] = None,
            write_requests: bool = True, **kw) -> Dict:
    """Run a single simulation; returns the summary dict (also written to out_dir/summary.json)."""
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    _apply_diagnostic_knobs()

    os.makedirs(out_dir, exist_ok=True)
    config = build_config(router, trace_file, out_dir, **kw)
    set_seeds(config.seed)
    t0 = time.time()
    sim = Simulator(config)
    atexit.unregister(sim._write_output)
    sim.run()
    wall = time.time() - t0

    s = summarize(sim._requests, sim._time, wall, sim.scheduler)
    s.update({"router": router, "trace": trace_file, "env": env or {}, **{k: v for k, v in kw.items()}})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    if write_requests:
        import pandas as pd
        pd.DataFrame([{
            "id": r.id, "arrived_at": r.arrived_at,
            "num_prefill_tokens": r._orig_num_prefill_tokens, "num_decode_tokens": r._orig_num_decode_tokens,
            "completed": r.completed, "completed_at": r._completed_at if r.completed else np.nan,
            "e2e": (r._completed_at - r.arrived_at) if r.completed else np.nan,
            "ttft": (r._prefill_completed_at - r.arrived_at) if r.completed else np.nan,
            "routed_at": getattr(r, "_routed_at", np.nan), "replica": getattr(r, "_routed_to", -1),
            "first_scheduled_at": r._scheduled_at if r.completed else np.nan,
            "num_restarts": r.num_restarts,
        } for r in sim._requests]).to_csv(os.path.join(out_dir, "requests.csv.gz"), index=False)
    return s
