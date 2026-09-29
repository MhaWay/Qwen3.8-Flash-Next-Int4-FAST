# SPDX-License-Identifier: Apache-2.0
"""flashnext-int4-b12x: optional end-of-prompt hidden-state capture.

ONE Qwen instance, MTP and every B12X optimization untouched. OFF BY DEFAULT:
serving only gains this behavior when launched with

    --worker-extension-cls=vllm_hidden_state.HiddenStateExtension

which serve.sh's HIDDEN_CAPTURE=1 adds on the b12x path. The extension gates
BOTH sides: the worker capture hook and the read route. No VLLM_SERVER_DEV_MODE,
no general /collective_rpc client, no second model.

Installed into the container's dist-packages AS vllm_hidden_state.py by
patch_b12x.py (importable by both the API and worker processes; workers copy the
mod folder and patch in every node).

Capture rule -- exactly one vector per request, at END OF PROMPT BEFORE DECODE:
record only at a step that ENDS that request's prompt (is_prefilling and
num_computed + scheduled >= prefill_len: covers plain and chunked prefills).
Purely-decode steps record nothing; the draft/MTP path is never touched.
A permanently-active flag stops capture on an unexpected build layout so
serving is immune to capture bugs.

Optional HN_TRACE_CAPTURE=1 also clones all decoder hidden rows of a cold,
single-step prefill with at most HN_TRACE_MAX_TOKENS tokens. This diagnostic
trace is served at /flashnext/hidden_state/trace; it is not an image-only
embedding and it is never collected for chunked or cache-hit prefills.
"""

import base64
import os
import re
import time

# Bounded and configurable memory. A stored vector is one bf16 row of the model
# hidden size; on this build that is 2560 x 2 bytes = ~5 KB, so the defaults cap
# captured data at ~5 MB per worker (~5 KB x 1024).
#   HN_MAX_ENTRIES (default 1024) = most requests held at once (LRU ceiling)
#   HN_TTL_SECONDS (default 300)  = a captured vector lives 5 min after capture
# Raise only while a probe needs a longer/wider window (8192 entries are ~40 MiB of bf16 values).
MAX_ENTRIES = int(os.environ.get("HN_MAX_ENTRIES", "1024"))
TTL_SECONDS = float(os.environ.get("HN_TTL_SECONDS", "300"))
# Diagnostic only: bounded full-prompt capture. 512 x 2560 x 2 bytes is
# 2.5 MiB per request in GPU memory; at most 8 requests are retained for 120s.
# Hard ceilings protect the server if someone sets larger values accidentally.
TRACE_ON = os.environ.get("HN_TRACE_CAPTURE") == "1"
TRACE_MAX_TOKENS = min(2048, max(1, int(os.environ.get("HN_TRACE_MAX_TOKENS", "512"))))
TRACE_MAX_ENTRIES = min(16, max(1, int(os.environ.get("HN_TRACE_MAX_ENTRIES", "8"))))
TRACE_TTL_SECONDS = min(300.0, max(1.0, float(os.environ.get("HN_TRACE_TTL_SECONDS", "120"))))
_MARKER = "HiddenStateExtension"

# The pinned server emits chatcmpl-<16 hex> (observed in live responses).
# Also accept 32-hex and canonical UUID forms for compatible vLLM builds.
# The engine may append one 8-hex suffix; partial IDs remain invalid.
_BASE_ID = (r"(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{32}|"
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")
_CANON = re.compile(r"^(?:chatcmpl-)?(" + _BASE_ID + r")(?:-([0-9a-fA-F]{8}))?$")


def _canon(req_id):
    """Parse a request id into (uuid, suffix-or-None) or None if not canonical.

    Accepted forms: `chatcmpl-<16hex>`, optional `-<8hex>`, and equivalent
    32-hex or hyphenated UUID forms (what HTTP callers and engines hold); the bare
    engine-side forms are also accepted. A truncated or malformed string returns
    None, so it can never match a stored capture. The prefix is optional only
    because the same uuid appears on both sides of the engine's id rewrite; the
    match is by parsed (uuid, suffix) pair, never by string prefix.
    """
    if not isinstance(req_id, str):
        return None
    m = _CANON.match(req_id)
    return (m.group(1), m.group(2)) if m else None

def active_for_extension(cls_str) -> bool:
    """Single source of truth for 'the feature is on': the worker extension name
    present in the launch (parallel_config or CLI args). Same flag on both sides."""
    return bool(cls_str) and _MARKER in str(cls_str)


class _Store:
    """engine_req_id -> cloned end-of-prompt vector, LRU + TTL."""
    __slots__ = ("vecs", "order", "max_entries", "ttl", "disabled")

    def __init__(self, max_entries=MAX_ENTRIES, ttl=TTL_SECONDS):
        self.vecs = {}    # req_id -> (tensor clone, last_touch)
        self.order = []   # req_id, oldest -> newest
        self.max_entries = max_entries
        self.ttl = ttl
        self.disabled = False

    def put(self, req_id, row):
        # clone immediately: hidden_states belongs to the current step and the
        # read happens later, so storing a view would alias other requests' data.
        if _canon(req_id) is None:
            return  # never store a non-canonical id: it could only ever match ambiguously
        if req_id in self.vecs:
            self.order.remove(req_id)
        elif len(self.order) >= self.max_entries:
            self._evict()
        self.order.append(req_id)
        self.vecs[req_id] = (row.detach().clone(), time.time())

    def _evict(self):
        now = time.time()
        victim = next((k for k in self.order if now - self.vecs[k][1] > self.ttl), None)
        if victim is None:
            victim = self.order[0]
        self.order.remove(victim)
        self.vecs.pop(victim, None)

    def read(self, req_id):
        now = time.time()
        for k in [k for k, (_, a) in self.vecs.items() if now - a > self.ttl]:
            self.vecs.pop(k, None)
            if k in self.order:
                self.order.remove(k)
        # Strict id matching (no ambiguous prefixes). A caller may hand either
        # the exact stored `chatcmpl-<uuid>-<suffix>` or just `chatcmpl-<uuid>`,
        # in which case a unique suffix-bearing match is resolved. Two stored
        # requests sharing one uuid with different suffixes, or anything that is
        # not one of these two exact forms, resolve to nothing rather than guess.
        # Only one request is exported - there is no read-all path.
        want = _canon(req_id)
        if want is None:
            return None
        base, base_suf = want
        hit = None
        for k in self.vecs:
            pair = _canon(k)
            if pair is None:
                continue
            uuid, suf = pair
            if uuid != base:
                continue
            if base_suf is not None:               # caller pinned the suffix
                if suf != base_suf:
                    continue
            elif suf is not None and hit is not None:
                return None  # base matched several suffixed keys: stay ambiguous
            if hit is not None:
                return None
            hit = k
        if hit is None:
            return None
        vec, _ = self.vecs[hit]
        self.vecs[hit] = (vec, now)  # a read refreshes TTL/LRU
        self.order.remove(hit)
        self.order.append(hit)
        import torch

        cpu = vec.detach().to("cpu", copy=True)
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.to(torch.float32)  # stable wire format for JSON clients
        raw = cpu.contiguous().numpy()
        return {
            "req_id": hit,
            "shape": list(raw.shape),
            "hidden_size": int(raw.shape[-1]),
            "dtype": str(raw.dtype),  # numpy name ("float32"), not a torch repr
            "b64": base64.b64encode(raw.tobytes()).decode(),
        }


def get_store(runner) -> _Store:
    st = getattr(runner, "_hn_store", None)
    if st is None:
        st = runner._hn_store = _Store()
    return st


def get_trace_store(runner) -> _Store:
    st = getattr(runner, "_hn_trace_store", None)
    if st is None:
        st = runner._hn_trace_store = _Store(TRACE_MAX_ENTRIES, TRACE_TTL_SECONDS)
    return st


def capture_step(runner, input_batch, hidden_states) -> None:
    """The per-step hook the runner patch installs (see patch_b12x.py). Returns
    instantly unless the extension is active; disables itself permanently and
    with a single warning on any unexpected build layout so serving is never
    disturbed. Never runs during drafting."""
    vcfg = getattr(runner, "vllm_config", None)
    pconf = getattr(vcfg, "parallel_config", None) if vcfg is not None else None
    if not active_for_extension(getattr(pconf, "worker_extension_cls", "")):
        return
    st = get_store(runner)
    if st.disabled:
        return
    try:
        _capture(st, input_batch, hidden_states, runner)
    except Exception:  # noqa: BLE001
        if not st.disabled:
            st.disabled = True
            import logging

            logging.getLogger("vllm_hidden_state").warning(
                "hidden-state capture disabled: unexpected vLLM layout", exc_info=True
            )


def _capture(st, ib, hidden_states, runner=None) -> None:
    """Store the end-of-prompt row for every request whose prompt this step ends.

    One vector per request, captured BEFORE decode:
      * the step that FINISHES a live (possibly chunked) prefill: is_prefilling
        and computed + this step >= prefill_len -- the last chunk's logits row.
    A full prefix-cache hit must recompute a prompt token while is_prefilling;
    if computed == prefill_len and is_prefilling is false, the sampler kernel
    may already overwrite its input with the first generated token. That row
    must never be reported as the end-of-prompt state. Decode/spec-decode and
    incomplete prefill steps store nothing.

    The row taken is hidden_states[logits_indices[cu_num_logits[i+1] - 1]] for
    request i: the last position of that request's logits window for the step.
    """
    import numpy as np
    n = ib.num_reqs
    if n == 0:
        return
    is_pf = np.asarray(ib.is_prefilling_np[:n], dtype=bool)
    computed = np.asarray(ib.num_computed_tokens_np[:n])
    ub = np.asarray(ib.seq_lens_cpu_upper_bound[:n])
    plen = np.asarray(ib.prefill_len_np[:n])

    want = is_pf & (computed < plen) & (ub >= plen)
    if not bool(np.any(want)):
        return
    cu = np.asarray(ib.cu_num_logits_np, dtype=np.int64)[: n + 1]
    for i, req in enumerate(ib.req_ids[:n]):
        if not want[i]:
            continue
        if cu[i + 1] <= cu[i]:
            continue  # this request has no logits row in the current step
        row = int(cu[i + 1] - 1)
        st.put(req, hidden_states[ib.logits_indices[row]])
    if TRACE_ON and runner is not None:
        trace_store = get_trace_store(runner)
        if not trace_store.disabled:
            try:
                _capture_trace(trace_store, ib, hidden_states, is_pf, computed, ub, plen)
            except Exception:  # never disable the ordinary one-vector capture
                trace_store.disabled = True
                import logging

                logging.getLogger("vllm_hidden_state").warning(
                    "hidden-state prompt trace disabled: unexpected layout", exc_info=True
                )


def _capture_trace(st, ib, hidden_states, is_pf, computed, ub, plen) -> None:
    """Keep every forward row of a single-step cold prefill, in prompt order.

    Skip chunked and cache-hit requests: their full prompt is not simultaneously
    in hidden_states. Never include generated or speculative draft positions.
    Query boundaries are the same batch row boundaries used to build positions.
    """
    import numpy as np

    n = ib.num_reqs
    boundaries = np.asarray(ib.query_start_loc_np[: n + 1], dtype=np.int64)
    if len(boundaries) != n + 1:
        raise ValueError("query boundaries missing")
    for i, req_id in enumerate(ib.req_ids[:n]):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        length = int(plen[i])
        if not (is_pf[i] and computed[i] == 0 and ub[i] >= length):
            continue
        if not (0 < length <= TRACE_MAX_TOKENS and end - start == length):
            continue
        if not (0 <= start < end <= hidden_states.shape[0]):
            raise ValueError("prompt rows out of bounds")
        st.put(req_id, hidden_states[start:end])


class HiddenStateExtension:
    """Worker mixin, mixed in ONLY when launched with
    --worker-extension-cls=vllm_hidden_state.HiddenStateExtension. No __init__
    (vLLM injects it into Worker.__bases__); method names are prefixed so they
    never collide (assertion in worker_base.py)."""

    def hidden_state_read(self, req_id: str):
        """Return the captured end-of-prompt vector of ONE request, or None. This
        is the only data path the feature exposes through vLLM's RPC."""
        st = getattr(getattr(self, "model_runner", None), "_hn_store", None)
        if st is None:
            return None
        return st.read(req_id)

    def hidden_state_trace_read(self, req_id: str):
        """Read the optional, bounded [prompt_tokens, hidden_size] trace."""
        st = getattr(getattr(self, "model_runner", None), "_hn_trace_store", None)
        return st.read(req_id) if st is not None else None


def attach_read_route(app, engine_selector=None):
    """The read surface: GET /flashnext/hidden_state/read?req_id=<chatcmpl-uuid>.
    One endpoint, one purpose, no dev mode, no general RPC surface.

    engine_selector(request) -> engine client exposing .collective_rpc(); by
    default app.state.engine_client as every other vLLM route uses it.
    """
    from fastapi import APIRouter, HTTPException, Request

    router = APIRouter()

    @router.get("/flashnext/hidden_state/read")
    async def _read(req_id: str, request: Request):
        if _canon(req_id) is None:
            # 400 (not 404) distinguishes a bad id from the route not existing,
            # which is what restart-hidden-capture.sh verifies.
            raise HTTPException(
                status_code=400,
                detail="req_id must be chatcmpl-<16hex> (or UUID) with optional -<8hex>",
            )
        engine = (engine_selector(request) if engine_selector
                  else request.app.state.engine_client)
        results = await engine.collective_rpc("hidden_state_read", args=(req_id,))
        got = next((r for r in results if r is not None), None)
        if got is None:
            raise HTTPException(
                status_code=404,
                detail="no hidden state captured for req_id (off, never captured, "
                "or expired)",
            )
        return got

    if TRACE_ON:
        @router.get("/flashnext/hidden_state/trace")
        async def _trace(req_id: str, request: Request):
            if _canon(req_id) is None:
                raise HTTPException(status_code=400, detail="invalid req_id")
            engine = (engine_selector(request) if engine_selector
                      else request.app.state.engine_client)
            results = await engine.collective_rpc("hidden_state_trace_read", args=(req_id,))
            got = next((r for r in results if r is not None), None)
            if got is None:
                raise HTTPException(status_code=404, detail="no full cold-prefill trace for req_id")
            return got

    app.include_router(router)
