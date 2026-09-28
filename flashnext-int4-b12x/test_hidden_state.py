"""CPU-only checks for the optional hidden-state capture module.

Run from this repo root:
    python flashnext-int4-b12x/test_hidden_state.py

No vLLM and no GPU needed. The same numpy/Torch path the runner hook calls is
exercised against fake InputBatch arrays shaped like the V2 model runner builds
them. This covers the capture RULE and store semantics; the runner/API wiring is
covered by the restart smoke test and the CLM-head validation, not here.

A note the tests make explicit: capture width is an arbitrary 16 here only to
keep the CPU check fast. The real model hides a 2560-d vector; width is not what
this file is checking, and nothing here depends on it.
"""

import base64
import time
from types import SimpleNamespace

import numpy as np
import torch

import hidden_state as hs


def _uuid(i):
    raw = f"{i:032x}"
    return raw[0:8] + "-" + raw[8:12] + "-" + raw[12:16] + "-" + raw[16:20] + "-" + raw[20:32]


def _ib(n=1, is_pf=None, computed=None, ub=None, plen=None,
        cu=None, n_logits=None, req_ids=None, suffix=True):
    """A stand-in for InputBatch for one step, with only what _capture reads."""
    if is_pf is None:
        is_pf = np.ones(n, dtype=bool)
    if computed is None:
        computed = np.zeros(n, dtype=np.int64)
    if plen is None:
        plen = np.full(n, 32, dtype=np.int64)
    if ub is None:
        ub = np.asarray(plen, dtype=np.int64).copy()
    if n_logits is None:
        n_logits = [1 for _ in range(n)]
    if cu is None:
        cu = np.concatenate([[0], np.cumsum(n_logits)]).astype(np.int64)
    rows = int(cu[-1])
    if req_ids is None:
        req_ids = [f"chatcmpl-{_uuid(i)}" + ("-%08x" % (1234567 + i * 7) if suffix else "")
                   for i in range(n)]
    ib = SimpleNamespace(
        num_reqs=n,
        is_prefilling_np=np.asarray(is_pf, dtype=bool)[:n],
        num_computed_tokens_np=np.asarray(computed, dtype=np.int64)[:n],
        seq_lens_cpu_upper_bound=np.asarray(ub, dtype=np.int64)[:n],
        prefill_len_np=np.asarray(plen, dtype=np.int64)[:n],
        cu_num_logits_np=np.asarray(cu, dtype=np.int64),
        # the real InputBatch keeps this as a device tensor of row indices
        logits_indices=torch.arange(rows, dtype=torch.long),
        req_ids=list(req_ids),
    )
    return ib


def new_runner():
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(worker_extension_cls=hs._MARKER)))


# logits_indices must index rows of hidden_states; _capture picks hidden[logits[cu-1]].
# Rows start at 1 so every row is non-zero (row 0 would be all zeros and the
# "captured something" assertions would not be meaningful).
def fake_hidden(rows, width):
    h = torch.arange(1, rows + 1, dtype=torch.float32)[:, None].repeat(1, width)
    return h


def decoded(vec):
    raw = np.frombuffer(base64.b64decode(vec["b64"]), dtype=getattr(np, vec["dtype"]))
    return torch.from_numpy(raw.copy())


def test_id_and_miss():
    st = hs._Store()
    uuid = _uuid(1)
    full = "chatcmpl-%s-abcd1234" % uuid
    st.put(full, torch.randn(7))
    assert st.read(full)["shape"] == [7], "exact read lost the vector"
    assert st.read("chatcmpl-" + uuid)["req_id"] == full, "bare http id must resolve unique suffix"
    assert st.read("%s-abcd1234" % uuid)["req_id"] == full, "engine-side form must resolve too"
    assert st.read("chatcmpl-%s-abc0ffff" % _uuid(77)) is None, "unknown id returned data"
    assert st.read("chatcmpl-abc12") is None, "partial/garbage id was accepted"
    assert st.read("5") is None, "bare non-uuid id was accepted"
    assert st.read("") is None and st.read(None) is None, "empty/None accepted"
    print("id grammar strictness OK")


def test_two_ids_same_uuid_is_ambiguous():
    # Two stored captures sharing a uuid with different random suffixes: the
    # suffix-less caller must get NOTHING -- refusing beats guessing here.
    st = hs._Store()
    uuid = _uuid(3)
    st.put("chatcmpl-%s-aaaa1111" % uuid, torch.ones(7))
    st.put("chatcmpl-%s-bbbb2222" % uuid, torch.ones(7) * 2)
    assert st.read("chatcmpl-" + uuid) is None, "ambiguous base resolved by guessing"
    exact = "chatcmpl-%s-aaaa1111" % uuid
    assert st.read(exact)["req_id"] == exact, "exact id must resolve"
    assert st.read("%s-bbbb2222" % uuid) is not None, "exact engine-side id must resolve"
    print("same-uuid two-suffix ambiguity refused OK")


def test_ttl_and_lru():
    st = hs._Store(max_entries=2, ttl=0.05)
    a, b, c, d, e = (f"chatcmpl-{_uuid(i)}" for i in range(100, 105))
    st.put(a, torch.ones(1))
    st.put(b, torch.ones(1) * 2)
    assert len(st.vecs) == 2 and st.read(a) is not None, "initial inserts failed"
    time.sleep(0.07)
    assert st.read(a) is None and st.read(b) is None, "expired entries were retained"
    st.put(c, torch.ones(1) * 3)
    st.put(d, torch.ones(1) * 4)
    assert st.read(c) is not None, "new capture not stored"
    st.put(e, torch.ones(1) * 5)
    assert st.read(d) is None and st.read(c) is not None and st.read(e) is not None, "LRU evicted the wrong capture"
    assert len(st.vecs) == 2, "LRU exceeded its ceiling"
    print("ttl + lru OK")


def test_keeps_prompt_vector_against_later_mutation():
    r = new_runner()
    ib = _ib(n=1, plen=[12], ub=[12], n_logits=[1])
    hidden = fake_hidden(1, 16)
    hs.capture_step(r, ib, hidden)
    st = hs.get_store(r)
    row = hidden[0].clone()  # the stored value must match this BEFORE any rewriting
    assert torch.equal(decoded(st.read(ib.req_ids[0])), row), "wrong row captured"
    hidden[0] = 0.0  # future steps rewrite this buffer in place
    assert torch.equal(decoded(st.read(ib.req_ids[0])), row), "store aliases the live buffer"
    print("prompt vector survives later buffer writes (clone) OK")


def test_last_window_row_per_request():
    # req0 occupies one logits row, req1 two: the row kept for req1 must be the SECOND
    # row of its window (last generated position), not the window start.
    r = new_runner()
    ib = _ib(n=2, plen=[10, 12], ub=[10, 12], n_logits=[1, 2])
    hidden = fake_hidden(3, 16)
    hs.capture_step(r, ib, hidden)
    st = hs.get_store(r)
    assert torch.equal(decoded(st.read(ib.req_ids[0])), hidden[0]), "wrong row for single-row window"
    assert torch.equal(decoded(st.read(ib.req_ids[1])), hidden[2]), "wrong row for multi-row window"
    print("last logits-window row OK")


def test_cache_hit_then_decode_no_touch():
    r = new_runner()
    ib = _ib(n=1, is_pf=[False], computed=[32], plen=[32], ub=[32], n_logits=[1])
    hidden = fake_hidden(1, 16)
    hs.capture_step(r, ib, hidden)
    st = hs.get_store(r)
    first = decoded(st.read(ib.req_ids[0]))
    assert bool(first.abs().sum()), "cache-hit first step not captured"
    # first decode step on this request: not prefilling, computed past prompt -> capture
    # nothing; stored prompt vector untouched.
    ib2 = _ib(n=1, is_pf=[False], computed=[33], plen=[32], ub=[33], n_logits=[1],
              req_ids=ib.req_ids)
    hs.capture_step(r, ib2, torch.zeros_like(hidden))
    assert torch.equal(decoded(st.read(ib.req_ids[0])), first), "decode changed the prompt vector"
    print("cache hit captured, decode leaves it alone OK")


def test_chunked_prefill_only_at_the_end():
    r = new_runner()
    ib = _ib(n=1, is_pf=[True], computed=[0], plen=[128], ub=[64], n_logits=[1])
    hs.capture_step(r, ib, fake_hidden(1, 16))
    assert hs.get_store(r).read(ib.req_ids[0]) is None, "mid-chunk must not capture"
    ib2 = _ib(n=1, is_pf=[True], computed=[64], plen=[128], ub=[128], n_logits=[1],
              req_ids=ib.req_ids)
    hs.capture_step(r, ib2, fake_hidden(1, 16))
    assert hs.get_store(r).read(ib.req_ids[0]) is not None, "last chunk must capture"
    print("chunked prefill captures at the end OK")


def test_off_is_inert():
    r = new_runner()
    r.vllm_config.parallel_config.worker_extension_cls = ""
    ib = _ib()
    hs.capture_step(r, ib, fake_hidden(1, 16))
    assert getattr(r, "_hn_store", None) is None, "disabled feature still built a store"
    print("flag off is inert OK")


def test_unexpected_layout_fails_safe():
    r = new_runner()
    ib = _ib()
    ib.prefill_len_np = None  # a vLLM layout we do not know yet
    hs.capture_step(r, ib, fake_hidden(1, 16))
    assert hs.get_store(r).disabled is True, "bad layout did not disable capture"
    # an already-disabled store must be a complete no-op, including no raise
    hs.capture_step(r, ib, fake_hidden(1, 16))
    print("unexpected layout disables, never crashes serving OK")


if __name__ == "__main__":
    test_id_and_miss()
    test_two_ids_same_uuid_is_ambiguous()
    test_ttl_and_lru()
    test_keeps_prompt_vector_against_later_mutation()
    test_last_window_row_per_request()
    test_cache_hit_then_decode_no_touch()
    test_chunked_prefill_only_at_the_end()
    test_off_is_inert()
    test_unexpected_layout_fails_safe()
    print("ALL CPU TESTS PASSED")
