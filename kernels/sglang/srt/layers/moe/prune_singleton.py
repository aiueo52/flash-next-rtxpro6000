"""P1: drop routed-expert routes that are the sole user of their expert.

The routed-expert grouped GEMM is at the expert-weight bandwidth roof and costs
~1.04 us per *distinct* expert per call (GEMM1 = 9.4 + 1.036*D us at T=4, GEMM2
~0.55x that; ``specs/G1_LOG.md``).  At T=16 about half the ~69 distinct experts
of a verify call are **singletons** -- reached by exactly one of the 16 chain
rows (``specs/MOE_SMALLM_SPEC.md`` 8.2, columns x1/x2/>=3).  Dropping a
singleton route removes a whole expert weight read (2.76 MB, ~1.6 us of
GEMM1+GEMM2); dropping a route to an expert some *other* row also uses saves
nothing at all, because that expert's weights are read either way.  A uniform
top-k cut cannot exploit that asymmetry; this can.

``SGLANG_MOE_PRUNE_SINGLETON_TAU=<float>`` (default 0 = off) masks every route
that is, within the current MoE call's batch,

* the only route to its expert, **and**
* below ``tau`` in normalised routing weight, **and**
* not the row's top-1 (``SGLANG_MOE_PRUNE_MIN_RANK``, default 1) -- a row must
  keep at least one expert or FlashInfer's finalize never writes its output.

A masked route gets expert id ``-1`` and weight 0.  FlashInfer treats any id
outside ``[start_expert, end_expert)`` as "not on this node": the fused prologue
sorts it into the ``num_experts_per_node`` bucket, so it lands past
``expert_first_token_offset[num_experts_per_node]`` -- the ``num_valid_tokens``
bound that ``expandInputRowsKernel`` and both finalize paths loop to -- and the
non-filling finalize skips it explicitly (``expert_id < 0`` in
``finalizeMoeRoutingNoFillingKernel``).  Nothing downstream has to change, and
the group-packing patch (``FLASHINFER_MOE_PACK_GROUPS``) keeps working because
its group list bound ``expanded_num_rows = num_tokens * k`` stays an upper
bound.  ``-1`` rather than ``num_experts`` as the sentinel so it can never
collide with a fused shared-expert slot at index ``num_experts``.

Everything runs in one single-CTA Triton kernel over the ``[M, k]`` top-k
tensors, in place, with no host sync and no allocation, so it is CUDA-graph
capturable.  Only ``[min_rows, max_rows]`` batches are touched (default 2..64),
which keeps the T=1 draft/decode calls -- where every route is a singleton by
construction, i.e. a plain top-k cut -- and prefill out of it.

Env:
    SGLANG_MOE_PRUNE_SINGLETON_TAU   weight threshold; 0/unset disables (default 0)
    SGLANG_MOE_PRUNE_MIN_RANK        lowest 0-based within-row rank eligible (default 1)
    SGLANG_MOE_PRUNE_MIN_ROWS        smallest batch to prune (default 2)
    SGLANG_MOE_PRUNE_MAX_ROWS        largest batch to prune (default 64)
    SGLANG_MOE_PRUNE_RENORM          rescale each row's surviving weights to sum 1 (default 0)
    SGLANG_MOE_PRUNE_KEEP_IDS        zero the weight but keep the id (equivalence control, default 0)
    SGLANG_MOE_PRUNE_PAIRWISE        O(N^2) reference path instead of the histogram (default 0)
    SGLANG_MOE_PRUNE_WARPS           num_warps for the kernel (default 4)
    SGLANG_MOE_PRUNE_NUM_EXPERTS     histogram width; must exceed every expert id (default 512)
    SGLANG_MOE_PRUNE_IN_PROLOGUE     P2: hand the mask to FlashInfer's fused routing prologue
                                     (patched csrc, ``FLASHINFER_MOE_PRUNE_SINGLETON_TAU``) and
                                     skip this kernel entirely (default 0)

P2 (``SGLANG_MOE_PRUNE_IN_PROLOGUE=1``): the same rule evaluated inside FlashInfer's
``fusedBuildExpertMapsSortFirstTokenAndStridesKernel`` (private csrc patch
``p2-prune-in-prologue.patch``), which already reads the top-k ids, so the prune
costs no launch and -- the point -- does not cost the prologue the ~54 % of its
time that is otherwise hidden behind the preceding kernels (``specs/P1_LOG.md``
step 6/7).  ``TAU``/``MIN_RANK``/``MIN_ROWS``/``MAX_ROWS`` are exported to the
``FLASHINFER_MOE_PRUNE_*`` variables the csrc reads (an explicitly set
``FLASHINFER_`` value wins), and this module's kernel becomes a no-op.  The csrc
writes the dropped routes back as id -1 / weight 0, the same convention as here.
If the fused prologue declines a call (batch > 32 rows, unsupported quant path,
``FLASHINFER_MOE_FUSED_PROLOGUE=0`` ...), that call is simply not pruned.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait

logger = logging.getLogger(__name__)

SENTINEL = -1


def _f(name: str, default: str) -> float:
    return float(os.environ.get(name, default) or default)


def _i(name: str, default: str) -> int:
    return int(os.environ.get(name, default) or default)


def _b(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


TAU = _f("SGLANG_MOE_PRUNE_SINGLETON_TAU", "0")
MIN_RANK = _i("SGLANG_MOE_PRUNE_MIN_RANK", "1")
MIN_ROWS = _i("SGLANG_MOE_PRUNE_MIN_ROWS", "2")
MAX_ROWS = _i("SGLANG_MOE_PRUNE_MAX_ROWS", "64")
RENORM = _b("SGLANG_MOE_PRUNE_RENORM", "0")
# Equivalence control: zero the weight but keep the expert id.  The MoE output
# is then mathematically identical to real pruning (a zero-weight route adds
# nothing in the finalize) while the grouped GEMM still reads every expert, so
# an A/B against the real thing isolates "is the -1 sentinel handled correctly"
# from "does dropping the route change the answer".
KEEP_IDS = _b("SGLANG_MOE_PRUNE_KEEP_IDS", "0")
PAIRWISE = _b("SGLANG_MOE_PRUNE_PAIRWISE", "0")
NUM_WARPS = _i("SGLANG_MOE_PRUNE_WARPS", "4")
# Upper bound on expert ids, i.e. the histogram width.  512 for this model;
# raised via the env if a wider router ever uses this path.
NUM_EXPERT_SLOTS = _i("SGLANG_MOE_PRUNE_NUM_EXPERTS", "512")
IN_PROLOGUE = _b("SGLANG_MOE_PRUNE_IN_PROLOGUE", "0")
# P2: the C++ side reads its knobs once, on the first MoE call, via getenv; os.environ
# writes reach it through putenv, and this module is imported long before that call.
if TAU > 0.0 and IN_PROLOGUE:
    for _k, _v in (
        ("FLASHINFER_MOE_PRUNE_SINGLETON_TAU", repr(TAU)),
        ("FLASHINFER_MOE_PRUNE_MIN_RANK", str(MIN_RANK)),
        ("FLASHINFER_MOE_PRUNE_MIN_ROWS", str(MIN_ROWS)),
        ("FLASHINFER_MOE_PRUNE_MAX_ROWS", str(MAX_ROWS)),
    ):
        os.environ.setdefault(_k, _v)
    if RENORM or KEEP_IDS:
        logger.warning("moe prune: RENORM/KEEP_IDS are not available in the in-prologue path")
# The Triton kernel runs only when pruning is on *and* not delegated to FlashInfer.
ENABLED = TAU > 0.0 and not IN_PROLOGUE

_warned: set = set()
# Per-device [NUM_EXPERT_SLOTS] int32 histogram scratch, allocated eagerly on the
# first non-capturing call and always left zeroed by the kernel.
_COUNTS: dict = {}


@triton.jit
def _prune_singleton_kernel(
    ids_ptr,
    w_ptr,
    cnt_ptr,
    M,
    TAU_,
    MIN_RANK_,
    K: tl.constexpr,
    MP: tl.constexpr,
    KP: tl.constexpr,
    SENTINEL_: tl.constexpr,
    KEEP_IDS_: tl.constexpr,
    PAIRWISE_: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """One CTA over the [M, k] top-k tile of a single MoE call.

    ``rank`` -- routes in the same row with a strictly larger weight (column
    index breaks ties), i.e. the 0-based within-row rank.  Computed from a
    per-row k x k compare, so it is exact and costs only ``M*k*k`` register ops
    (1 280 at T=16, k=10) -- no dependence on the order ``topk`` happened to
    store the ids in.

    ``cnt`` -- how many rows of this call route to the same expert; ``cnt == 1``
    is a singleton.  Two ways to get it:

    * default: a **device histogram** over the E expert slots in ``cnt_ptr``.
      ``atomic_add`` -> ``__syncthreads`` -> gather -> ``atomic_xchg`` back to
      zero, so the buffer is always zero on entry and never has to be cleared
      with a plain store (a store would land in this SM's L1 while the atomics
      go to L2, and the gather could then read a stale line).  The gather is
      ``.cg`` for the same reason.  O(N).
    * ``PAIRWISE_``: the O(N^2/CHUNK) all-pairs compare, kept as a reference
      for the unit test and as a fallback if the scratch buffer cannot be
      allocated eagerly.

    Everything is one CTA, so ``tl.debug_barrier`` is a plain ``__syncthreads``
    and there is no inter-CTA race between the loads and the in-place stores.
    """
    pdl_wait(USE_PDL)
    r = tl.arange(0, MP)
    c = tl.arange(0, KP)
    off = r[:, None] * K + c[None, :]
    m = (r[:, None] < M) & (c[None, :] < K)
    ids = tl.load(ids_ptr + off, mask=m, other=SENTINEL_)
    w = tl.load(w_ptr + off, mask=m, other=0.0).to(tl.float32)

    # within-row rank: [MP, KP, KP]
    gt = (w[:, None, :] > w[:, :, None]) | (
        (w[:, None, :] == w[:, :, None]) & (c[None, None, :] < c[None, :, None])
    )
    rank = tl.sum((gt & m[:, None, :]).to(tl.int32), axis=2)

    if PAIRWISE_:
        cnt = tl.zeros([MP, KP], tl.int32)
        for rp in tl.static_range(MP):
            orow = tl.load(ids_ptr + rp * K + c, mask=(c < K) & (rp < M),
                           other=SENTINEL_ - 1)
            cnt += tl.sum((ids[:, :, None] == orow[None, None, :]).to(tl.int32),
                          axis=2)
    else:
        tl.atomic_add(cnt_ptr + ids, 1, mask=m)
        tl.debug_barrier()
        cnt = tl.load(cnt_ptr + ids, mask=m, other=0, cache_modifier=".cg")
        tl.debug_barrier()
        tl.atomic_xchg(cnt_ptr + ids, 0, mask=m)

    prune = m & (cnt == 1) & (w < TAU_) & (rank >= MIN_RANK_)
    tl.debug_barrier()          # every load above precedes every store below
    if not KEEP_IDS_:
        tl.store(ids_ptr + off, tl.where(prune, SENTINEL_, ids), mask=m)
    tl.store(w_ptr + off, tl.where(prune, 0.0, w).to(w_ptr.dtype.element_ty), mask=m)
    pdl_trigger(USE_PDL)


def _skip(reason: str) -> None:
    if reason not in _warned:
        _warned.add(reason)
        logger.warning("moe singleton prune disabled for this shape: %s", reason)


def maybe_prune_singleton_routes(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
    """In-place mask of low-weight singleton routes.  No-op unless enabled."""
    if not ENABLED:
        return
    prune_singleton_routes_(topk_ids, topk_weights, TAU, MIN_RANK, MIN_ROWS, MAX_ROWS)


def prune_singleton_routes_(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tau: float,
    min_rank: int = 1,
    min_rows: int = 2,
    max_rows: int = 64,
) -> None:
    """The kernel itself, independent of the env gating (used by the P2 equivalence test)."""
    if topk_ids.dim() != 2 or topk_weights.shape != topk_ids.shape:
        return _skip(f"shape {tuple(topk_ids.shape)}/{tuple(topk_weights.shape)}")
    M, K = topk_ids.shape
    if M < min_rows or M > max_rows:
        return
    if not (topk_ids.is_contiguous() and topk_weights.is_contiguous()):
        return _skip("non-contiguous top-k tensors")
    dev = topk_ids.device
    cnt = _COUNTS.get(dev)
    if cnt is None:
        # The histogram scratch must not be allocated inside a CUDA graph
        # capture (it would land in that graph's private pool).  SGLang runs two
        # eager warmup forwards per batch size before every capture, so this
        # always fires on an eager call; the pairwise path is the safety net.
        if torch.cuda.is_current_stream_capturing():
            _skip("scratch not allocated before capture; using pairwise")
        else:
            cnt = torch.zeros(NUM_EXPERT_SLOTS, dtype=torch.int32, device=dev)
            _COUNTS[dev] = cnt
    MP = triton.next_power_of_2(M)
    KP = triton.next_power_of_2(K)
    _prune_singleton_kernel[(1,)](
        topk_ids,
        topk_weights,
        cnt if cnt is not None else topk_ids,   # unused when PAIRWISE_
        M,
        tau,
        min_rank,
        K=K,
        MP=MP,
        KP=KP,
        SENTINEL_=SENTINEL,
        KEEP_IDS_=KEEP_IDS,
        PAIRWISE_=PAIRWISE or cnt is None,
        USE_PDL=PDL,
        num_warps=NUM_WARPS,
        launch_pdl=PDL,
    )
    if RENORM:
        topk_weights.div_(
            topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        )


def describe() -> str:
    return (
        f"tau={TAU} min_rank={MIN_RANK} rows=[{MIN_ROWS},{MAX_ROWS}] "
        f"renorm={RENORM} keep_ids={KEEP_IDS} pairwise={PAIRWISE} "
        f"warps={NUM_WARPS} in_prologue={IN_PROLOGUE}"
    )
