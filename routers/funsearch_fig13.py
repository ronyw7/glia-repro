"""Router produced by FunSearch in the Glia paper (Fig. 13), transcribed verbatim.

Only the imports were added (the figure omits them).
"""
import math
from typing import Any, Dict, List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class CustomGlobalScheduler(BaseGlobalScheduler):  # type: ignore[name-defined]
    """Latency-oriented, eviction-aware global scheduler.
    Key features
    -------------
    1. Decode length prediction per *prefill* bucket (small / mid / large)
    with an online exponential moving average; gives markedly better
    memory-footprint forecasts than a single global estimate.
    2. Looks ahead and keeps a projection of every replica's future state
    (memory blocks, remaining pre-fill backlog, queue length). The
    projection is updated greedily after each assignment so later
    decisions use a consistent view.
    3. Request priority is *rescue-first SJF*: previously evicted jobs first
    (to avoid starvation / wasted work), then smaller **total** expected
    tokens, finally FIFO.
    4. Replica selection minimises a composite cost of projected memory
    utilisation (quadratic), outstanding pre-fill backlog, queue length
    and the *instant* block deficit for the pre-fill of the candidate
    request. Jobs with restarts receive a multiplicative cost discount.
    5. Admission control: a new request is dispatched only if the projected
    utilisation stays below a configurable soft limit. The limit is
    relaxed slightly for restarted jobs so they can finish.
    """

    # ------------- tunables -------------
    _BUCKET_BOUNDS = (128, 512)  # <128 small, 128-512 mid, >512 large
    _EMA_ALPHA = 0.10  # smoothing for bucketed averages
    _INIT_DECODE_EST = 96.0  # bootstrap decode len (tokens)
    _MIN_DECODE = 32.0
    _MAX_DECODE = 1024.0
    _SOFT_UTIL_CAP = 1.03  # ordinary requests must stay under this
    _SOFT_UTIL_CAP_RESTART = 1.10  # restarted jobs may exceed slightly
    # cost weights (should sum ~1)
    _W_UTIL = 0.55  # projected utilisation^2
    _W_BACKLOG = 0.25  # remaining pre-fill backlog fraction
    _W_QUEUE = 0.10  # queue length fairness
    _W_DEFICIT = 0.10  # instantaneous block deficit
    _OVER_CAP_PEN = 12.0  # extra when util>1
    _RESTART_DISCOUNT = 0.6  # multiplicative cost discount per restart
    # ------------------------------------

    def __init__(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        super().__init__(*args, **kwargs)
        # bucketed decode length EWMA statistics
        # structure: (count , avg)
        self._bucket_avg: List[float] = [self._INIT_DECODE_EST] * 3
        self._bucket_cnt: List[int] = [0, 0, 0]
        # global fallback EWMA
        self._global_avg: float = self._INIT_DECODE_EST
        # snapshot of visible requests from previous tick (id -> (pf, processed))
        self._prev_snapshot: Dict[int, Tuple[int, int]] = {}

    # --------------------- helper: bucket index ---------------------
    @classmethod
    def _bucket_idx(cls, prefill: int) -> int:
        if prefill < cls._BUCKET_BOUNDS[0]:
            return 0
        if prefill < cls._BUCKET_BOUNDS[1]:
            return 1
        return 2

    # --------------------- statistics maintenance ------------------
    def _update_decode_statistics(self) -> None:
        """Detect completed requests and update bucket/global decode EWMAs."""
        current: Dict[int, Tuple[int, int]] = {}

        # helper to insert into current snapshot quickly
        def _collect(req):
            current[id(req)] = (req.num_prefill_tokens, req.num_processed_tokens)

        for req in self._request_queue:
            _collect(req)
        for rep in self._replica_schedulers.values():
            for rq in rep.pending_queue:
                _collect(rq)
            for rq in rep.active_queue:
                _collect(rq)
        # detect finished requests
        finished_ids = set(self._prev_snapshot.keys()) - set(current.keys())
        for rid in finished_ids:
            pf_tokens, processed = self._prev_snapshot[rid]
            decode_tokens = max(0, processed - pf_tokens)
            if decode_tokens <= 0:
                continue
            # update bucket stats
            bidx = self._bucket_idx(pf_tokens)
            old_avg = self._bucket_avg[bidx]
            new_avg = (1.0 - self._EMA_ALPHA) * old_avg + self._EMA_ALPHA * decode_tokens
            self._bucket_avg[bidx] = min(max(new_avg, self._MIN_DECODE), self._MAX_DECODE)
            if self._bucket_cnt[bidx] < 1e9:  # avoid overflow
                self._bucket_cnt[bidx] += 1
            # update global average
            g_new = (1.0 - self._EMA_ALPHA) * self._global_avg + self._EMA_ALPHA * decode_tokens
            self._global_avg = min(max(g_new, self._MIN_DECODE), self._MAX_DECODE)
        self._prev_snapshot = current

    # -------------------- decode prediction ------------------------
    def _predict_decode(self, prefill_tokens: int) -> float:
        bidx = self._bucket_idx(prefill_tokens)
        if self._bucket_cnt[bidx] >= 10:  # need some data for bucket-specific
            return self._bucket_avg[bidx]
        return self._global_avg

    # -------------------- utility functions ------------------------
    @staticmethod
    def _ceil_div(a: float, b: int) -> int:
        return int(math.ceil(a / b))

    # ---------------------------- main ------------------------------
    def schedule(self) -> List[Tuple[int, 'Request']]:  # type: ignore[name-defined]
        # housekeeping
        self._update_decode_statistics()
        if not self._request_queue:
            return []
        replicas: Dict[int, 'ReplicaScheduler'] = self._replica_schedulers  # type: ignore[name-defined]
        num_repls: int = max(1, self._num_replicas)

        # ---------- priority sort for global queue ----------
        def _priority(req: 'Request') -> Tuple[int, float, float]:  # type: ignore[name-defined]
            predicted_total = req.num_prefill_tokens + self._predict_decode(req.num_prefill_tokens)
            return (-req.num_restarts, predicted_total, req.arrived_at)

        self._request_queue.sort(key=_priority)
        # ---------- projected replica states (without unrouted) ----------
        proj_blocks: Dict[int, int] = {}
        proj_backlog_tokens: Dict[int, int] = {}
        proj_queue_len: Dict[int, int] = {}
        blk_size: Dict[int, int] = {}
        token_capacity: Dict[int, int] = {}
        for rid, rep in replicas.items():
            bs = rep.block_size
            blk_size[rid] = bs
            token_capacity[rid] = rep.num_blocks * bs
            blocks = rep.num_allocated_blocks  # currently allocated blocks
            backlog_tokens = 0
            qlen = len(rep.active_queue) + len(rep.pending_queue)
            # active requests
            for rq in rep.active_queue:
                # remaining future blocks for this request
                total_tokens_goal = rq.num_prefill_tokens + self._predict_decode(rq.num_prefill_tokens)
                future_blocks = self._ceil_div(total_tokens_goal, bs)
                already_blocks = self._ceil_div(rq.num_processed_tokens, bs)
                blocks += max(0, future_blocks - already_blocks)
                # backlog tokens (remaining prefill)
                if rq.num_processed_tokens < rq.num_prefill_tokens:
                    backlog_tokens += rq.num_prefill_tokens - rq.num_processed_tokens
            # pending requests
            for rq in rep.pending_queue:
                total_tokens_goal = rq.num_prefill_tokens + self._predict_decode(rq.num_prefill_tokens)
                blocks += self._ceil_div(total_tokens_goal, bs)
                backlog_tokens += rq.num_prefill_tokens
            proj_blocks[rid] = blocks
            proj_backlog_tokens[rid] = backlog_tokens
            proj_queue_len[rid] = qlen
        avg_queue_len = (sum(proj_queue_len.values()) / num_repls) + 1e-6
        # ---------- greedy assignment loop ----------
        mapping: List[Tuple[int, 'Request']] = []  # type: ignore[name-defined]
        remaining: List['Request'] = []  # requests we skip this tick
        while self._request_queue:
            req = self._request_queue.pop(0)
            pred_decode = self._predict_decode(req.num_prefill_tokens)
            total_tokens_req = req.num_prefill_tokens + pred_decode
            best_rid: int | None = None
            best_cost: float = float('inf')
            best_util_after: float = 0.0
            for rid, rep in replicas.items():
                bs = blk_size[rid]
                req_blocks = self._ceil_div(total_tokens_req, bs)
                pf_blocks = self._ceil_div(req.num_prefill_tokens, bs)
                free_now_blocks = rep.num_blocks - rep.num_allocated_blocks
                deficit_blocks = max(0, pf_blocks - free_now_blocks)
                util_after = (proj_blocks[rid] + req_blocks) / rep.num_blocks
                backlog_after = proj_backlog_tokens[rid] + req.num_prefill_tokens
                queue_after = proj_queue_len[rid] + 1
                cost = (
                    self._W_UTIL * (util_after ** 2) +
                    self._W_BACKLOG * (backlog_after / (token_capacity[rid] + 1e-6)) +
                    self._W_QUEUE * (queue_after / avg_queue_len) +
                    self._W_DEFICIT * (deficit_blocks / (rep.num_blocks + 1e-6))
                )
                if util_after > 1.0:
                    cost += self._OVER_CAP_PEN * (util_after - 1.0) ** 2
                # discount for restarts
                if req.num_restarts:
                    cost *= (1.0 - self._RESTART_DISCOUNT) ** req.num_restarts
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best_rid = rid
                    best_util_after = util_after
            if best_rid is None:
                remaining.append(req)
                continue
            # ------ admission control ------
            cap = self._SOFT_UTIL_CAP_RESTART if req.num_restarts else self._SOFT_UTIL_CAP
            if best_util_after > cap:
                # keep for next tick
                remaining.append(req)
                continue
            # commit placement
            sel = best_rid
            mapping.append((sel, req))
            bs_sel = blk_size[sel]
            req_blocks_sel = self._ceil_div(total_tokens_req, bs_sel)
            proj_blocks[sel] += req_blocks_sel
            proj_backlog_tokens[sel] += req.num_prefill_tokens
            proj_queue_len[sel] += 1
            avg_queue_len = (sum(proj_queue_len.values()) / num_repls) + 1e-6
        # push remaining requests back to queue (maintain order)
        self._request_queue = remaining + self._request_queue
        return mapping
