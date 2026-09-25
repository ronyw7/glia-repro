"""Template for your own router. Copy this file, implement schedule(), then run:

    python -m glia_repro.run --routers routers/my_router.py llq routers/glia_hra.py

What the router can see (all read-only; do not mutate requests/replicas):

  self._request_queue            requests waiting at the router (list, you may reorder/pop)
  self._replica_schedulers       {replica_id: ReplicaScheduler}
  self._num_replicas

  ReplicaScheduler (per GPU):
    .num_blocks, .block_size     KV-cache capacity (blocks of 16 tokens)
    .num_allocated_blocks        blocks in use right now
    .pending_queue               routed here, no GPU memory yet (includes restarted requests)
    .active_queue                holding KV-cache (prefilling or decoding)
    .num_pending_requests, .num_active_requests

  Request:
    .arrived_at, .num_prefill_tokens, .num_processed_tokens, .num_restarts, .is_prefill_complete
    (.num_decode_tokens exists but is the *oracle* output length -- not allowed, as in the paper)

schedule() is called on every request arrival AND every request completion. Return a list of
(replica_id, request) and pop those requests from self._request_queue. Anything left in the
queue waits until the next call (admission control). Every request must eventually be routed.

Memory model (vLLM / Sarathi in Vidur): a replica admits a pending request only if its prompt's
blocks fit (minus a 1% watermark); decode then grows 1 block per 16 tokens. When a running request
needs a block and none is free, the *youngest* running request is evicted and restarted from
scratch (its prompt becomes prompt+generated-so-far) -- wasted work that HRA is designed to avoid.
"""
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class MyGlobalScheduler(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        # Example: least-loaded queue (LLQ). Replace with your algorithm.
        self._request_queue.sort(key=lambda r: r.arrived_at)
        load = {
            rid: rs.num_pending_requests + rs.num_active_requests
            for rid, rs in self._replica_schedulers.items()
        }
        mapping = []
        while self._request_queue:
            req = self._request_queue.pop(0)
            rid = min(load, key=load.get)
            load[rid] += 1
            mapping.append((rid, req))
        return mapping
