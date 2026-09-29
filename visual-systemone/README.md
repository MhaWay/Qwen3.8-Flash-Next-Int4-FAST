# Visual System One for Qwen Flash Next

An **opt-in experimental companion** to this repository's current vLLM server. It stores the latest JPEG per session and reads Qwen's logprobs for typed `choice`, `noul`, and `score` questions. Its own process uses CPU and never loads a second Qwen. The image still traverses the multimodal Qwen forward path for each decision. This is **not yet** the contrastive CLM head or a 10 Hz inference engine.

## Execution order

1. On the Spark, use your **existing** model cache and Eugr b12x installation. This companion does not call `setup.sh` or download model weights. If your current Qwen service is already running with b12x, leave it running and confirm `curl -sS http://127.0.0.1:8000/v1/models`. If it is stopped, start it from this repository with `BACKEND=b12x ./serve.sh` (or `BACKEND=b12x ./serve.sh -d` for background operation). This requires the Eugr recipe/mod and pinned image to be installed already; `serve.sh` checks for the cached model, table, recipe and image and exits if they are absent. Do not start a second Qwen process for the companion.
2. Your existing image use establishes that the current serving stack handles images. On the Spark, the **new** check for System One is candidate logprobs on an image: `python3 visual-systemone/scripts/probe.py frame.jpg --url http://127.0.0.1:8000`. The probe also repeats a small image request as a sanity check. The final response must contain both requested `token_id` values in the first `top_logprobs` entry. If it does not, inspect this server build's logprob implementation before starting the companion. This gateway accepts individual JPEG frames; it does not currently ingest video files or streams.
3. Install into a Python virtual environment: `python3 -m venv .venv && .venv/bin/pip install -e './visual-systemone[test]'` from the repository root. Start `VISUAL_VLLM_URL=http://127.0.0.1:8000 .venv/bin/visual-systemone`. It binds **127.0.0.1:8088** by default. For a Windows client, use an SSH tunnel or private network binding with access control; never expose this unauthenticated prototype publicly.
4. For an image decision, upload a recent JPEG: `python3 visual-systemone/scripts/push_frame.py frame.jpg`, then query `/v1/systemone` with `state_id` (example below). For a text-only decision, omit `state_id` or set it to `null`; no frame is needed. A supplied `state_id` still requires a current frame. The gateway keeps one current frame per session, no video backlog and no image history. The default TTL is 10 seconds, 8 sessions, 300 KB per JPEG.
5. Benchmark visual correctness and end-to-end p50/p95 latency on your Spark before connecting a controller. At most one inference runs per session. A newer frame invalidates an in-flight decision with HTTP 409; HTTP 429 means one is already in flight. On any error, release all game controls.

The [NVIDIA forum update by azampatti](https://forums.developer.nvidia.com/t/up-to-70tok-s-qwen3-8-flash-next-int4-autoround/382733?page=7) explicitly says the September 23 Eugr integration needs **no model update** when the latest weights are already cached. The [later pinned-image discussion](https://forums.developer.nvidia.com/t/up-to-70tok-s-qwen3-8-flash-next-int4-autoround/382733?page=10) recommends `./eugr-setup.sh` rather than overwriting `vllm-node-b12x` with Eugr's moving `latest` via `build-and-copy.sh`. If the mod or pinned image is missing, run `./eugr-setup.sh` once; it may download the **container image**, but does not re-download model weights. Eugr's b12x stack can use more Spark RAM than the classic image; reduce `KV_BYTES` only if loading fails due to memory pressure. Keep your existing working service untouched until the companion probe is ready.

```bash
curl -sS http://127.0.0.1:8088/v1/systemone -H 'Content-Type: application/json' -d '{"model":"qwen-visual-choice-v0","state_id":"minecraft","state":"Goal: reach the tree. Stop if a hostile mob appears.","questions":{"move":{"type":"choice","instructions":"What should I do for the next 200 ms?","criteria":{"forward":"Move forward","left":"Turn left","right":"Turn right","stop":"Release all controls"}}}}'
```

This extends the Jev wire shape with `state_id`, as the original API has no visual session reference. `probabilities` are a softmax over listed options, **not calibrated success probabilities**. `confidence` is normalized inverse entropy and likewise not a safety guarantee. `stop` should be checked against independent safeguards before any game action.

Text-only request (no frame or visual session):

```bash
curl -sS http://127.0.0.1:8088/v1/systemone -H 'Content-Type: application/json' -d '{"model":"qwen-visual-choice-v0","state":"The customer was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this a billing issue?"}}}'
```

In this mode `state_id` and `seq` are `null` in the response; the same running Qwen instance handles both modes.

## Architecture and the CLM path

```text
frame capture (desktop/RTX 5080) -> latest JPEG gateway -> existing Qwen/B12X (DGX Spark)
                                          |                    |
                                          |                   text+image forward, candidate logprobs
                                          +---- Jev-shaped choice -> guarded game controller
```

The [CLM reference project](https://github.com/Contrastive-LM/CLM) embeds states and candidate actions using frozen Qwen3-8B plus separately trained state/action projection heads. Its published heads are trained for *textual Qwen3-8B last-token embeddings*. For this fork we need two gated investigations before promising visual CLM on a single copy of Flash Next:

- Establish whether the **specific pinned B12X vLLM generation instance** can also return usable *multimodal last-token state embeddings*, with deterministic pooling. Running a second pooling `vllm serve` for the same model duplicates weights and defeats the one-copy requirement. If dual task serving is unavailable, a version-pinned runner extension is needed; do not add a speculative patch to the default recipe.
- Once state and action vectors are available, freeze Qwen and train fresh contrastive heads on `(frame, goal, action)` pairs with hard negatives; record checkpoint/config tied to the exact model revision, template and pooling method. Training the small heads can be done on the RTX 5080; feature extraction still needs the Spark and the model. The existing CLM head is not drop-in compatible with these features.

The present implementation provides a measured baseline and a safe integration seam. Add a CLM backend *after* embedding extraction is validated. The Spark retains Qwen's regular `/v1/chat/completions` for slow brain planning. A future game controller may combine screenshots with reliable HUD/telemetry and release buttons on timeout, stale output or low confidence.

For training data, run `python3 visual-systemone/scripts/record_example.py frame.jpg --goal 'Reach the tree' --question 'Next action?' --actions forward left right stop --chosen forward --split train`. It copies the image and appends a labeled record under `visual-dataset/` (ignored by Git). Use `val` and `test` splits from different play sessions or worlds to avoid near-duplicate frames crossing splits. The dataset is a preparation artifact; do not train heads until the multimodal embedding path is measured and fixed.

## Tests and boundaries

Run `.venv/bin/pytest -q visual-systemone/tests`. Unit tests mock vLLM and never require model weights. Live candidate-logprob and embedding checks require the user's own Spark and image; no claim about System One latency or visual CLM support is made before those checks pass. The process stores frames only in RAM and does not persist history, but the normal Qwen service will still process and cache its own request state according to vLLM's scheduler.
