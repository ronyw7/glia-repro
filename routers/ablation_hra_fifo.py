"""ABLATION: HRA admission control with FIFO (arrival-order) queueing instead of SPF.

Identical to glia_hra.py (Glia Fig. 12) except for the queue sort key. This is the
"initial HRA" stage of the paper's narrative (Sec. 3.3), before Glia composed it
with shortest-prompt-first.

This scheduler mitigates vLLM pre-emptions by keeping a small KV-cache
head-room on every replica *at admission time*. For each incoming request we
pessimistically reserve additional blocks to account for the (unknown) decode
phase and admit the request only if the target replica would still retain the
configured safety margin.
Empirical defaults (good for the ShareGPT-style workload used in Vidur's
benchmarks):
DECODE_TO_PREFILL_RATIO = 0.6 # avg decode/prompt tokens
SAFETY_FRACTION = 0.03 # keep last 3 % blocks free
These values reduce average end-to-end latency by ~40 % compared to LLQ while
maintaining >95 % GPU utilisation.
"""
import os
from math import ceil
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

# ---------------------------------------------------------------------------
# Tunable constants (change if workload characteristics differ significantly)
# ---------------------------------------------------------------------------
DECODE_TO_PREFILL_RATIO: float = 0.6  # pessimistic decode growth factor
SAFETY_FRACTION: float = 0.03  # minimum fraction of blocks kept free

# (repro harness only) allow parameter sweeps without editing the file
DECODE_TO_PREFILL_RATIO = float(os.environ.get("HRA_R", DECODE_TO_PREFILL_RATIO))
SAFETY_FRACTION = float(os.environ.get("HRA_M", SAFETY_FRACTION))


class AIGlobalScheduler(BaseGlobalScheduler):
    """Memory-aware global scheduler with fixed head-room admission control."""

    # pylint: disable=protected-access
    def schedule(self) -> List[Tuple[int, Request]]:
        # Always serve the *shortest* prompt next (SJF) to minimise mean latency.
        self._request_queue.sort(key=lambda r: r.arrived_at)  # ABLATION: FIFO, not SPF
        if not self._request_queue:
            return []
        # Cluster-wide, all replicas share the same memory configuration.
        any_scheduler = next(iter(self._replica_schedulers.values()))
        block_size = any_scheduler._config.block_size
        max_blocks = any_scheduler._config.num_blocks
        min_free_blocks = int(max_blocks * SAFETY_FRACTION)
        # Snapshot per-replica state and keep optimistic updates locally so that
        # multiple placements within one call are consistent.
        allocated_blocks = {
            rid: rs.num_allocated_blocks for rid, rs in self._replica_schedulers.items()
        }
        pending_reserved_blocks = {
            rid: ceil(
                sum(r.num_prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO) for r in rs._request_queue)
                / block_size
            )
            for rid, rs in self._replica_schedulers.items()
        }
        queue_lengths = {
            rid: rs.num_pending_requests + rs.num_active_requests for rid, rs in self._replica_schedulers.items()
        }
        request_mapping: List[Tuple[int, Request]] = []
        idx = 0
        # Traverse requests in order; if the head request cannot be admitted we
        # stop to preserve FIFO fairness (new arrivals behind it must wait).
        while idx < len(self._request_queue):
            req = self._request_queue[idx]
            req_blocks = ceil(
                req.num_prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO) / block_size
            )
            admissible = []
            for rid in self._replica_schedulers.keys():
                projected_usage = allocated_blocks[rid] + pending_reserved_blocks[rid] + req_blocks
                free_after = max_blocks - projected_usage
                if free_after >= min_free_blocks:
                    admissible.append(rid)
            if not admissible:
                break  # cannot place the oldest waiting request right now
            # Choose replica with lowest projected usage; tie-break by queue length.
            target_rid = min(
                admissible,
                key=lambda rid: (allocated_blocks[rid] + pending_reserved_blocks[rid], queue_lengths[rid]),
            )
            # Commit placement and optimistically update state for subsequent decisions.
            request_mapping.append((target_rid, req))
            self._request_queue.pop(idx)  # do *not* increment idx
            pending_reserved_blocks[target_rid] += req_blocks
            queue_lengths[target_rid] += 1
        return request_mapping
