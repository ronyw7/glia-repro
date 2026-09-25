"""Train + cache Vidur's execution-time predictors for the A10/Llama-3-8B setup (uses all cores)."""
import glob
import os
import time

from glia_repro.sim import REPO, build_config


def main():
    from vidur.execution_time_predictor import ExecutionTimePredictorRegistry

    trace = sorted(glob.glob(os.path.join(REPO, "data/traces/*/*.csv")))[0]
    cfg = build_config("llq", trace, os.path.join(REPO, "results", "_warm"))
    t0 = time.time()
    ExecutionTimePredictorRegistry.get(
        cfg.execution_time_predictor_config.get_type(),
        predictor_config=cfg.execution_time_predictor_config,
        replica_config=cfg.cluster_config.replica_config,
        replica_scheduler_config=cfg.cluster_config.replica_scheduler_config,
        metrics_config=cfg.metrics_config,
    )
    print(f"predictor ready in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
