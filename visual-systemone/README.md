# Visual System One for Qwen Flash Next

An **opt-in experimental companion** to this repository's current vLLM server. It stores the latest JPEG per session and reads Qwen's logprobs for typed `choice`, `noul`, and `score` questions. Its own process uses CPU and never loads a second Qwen. The image still traverses the multimodal Qwen forward path for each decision. This is **not yet** the contrastive CLM head or a 10 Hz inference engine.

## Execution order

1. On your Spark, keep the existing `./serve.sh` setup. Do not change its recipe or download weights again. Confirm `/v1/models` on port 8000. A pinned B12X image and the exact model snapshot must be tested with images, even if a different version of Qwen supports vision.
2. On the machine where you run the companion, test image + candidate logprobs against the *already running* server: `python3 visual-systemone/scripts/probe.py frame.jpg --url http://127.0.0.1:8000`. The final response must contain both requested `token_id` values in the first `top_logprobs` entry. If it does not, stop here and inspect the server's logprob implementation.
3. Install into a Python virtual environment: `python3 -m venv .venv && .venv/bin/pip install -e './visual-systemone[test]'` from the repository root. Start `VISUAL_VLLM_URL=http://127.0.0.1:8000 .venv/bin/visual-systemone`. It binds **127.0.0.1:8088** by default. For a Windows client, use an SSH tunnel or private network binding with access control; never expose this unauthenticated prototype publicly.
4. Upload a recent JPEG: `python3 visual-systemone/scripts/push_frame.py frame.jpg`. Query `/v1/systemone` (example below). The gateway keeps one current frame per session, no video backlog and no image history. The default TTL is 10 seconds, 8 sessions, 300 KB per JPEG.
5. Benchmark visual correctness and end-to-end p50/p95 latency on your Spark before connecting a controller. At most one inference runs per session. A newer frame invalidates an in-flight decision with HTTP 409; HTTP 429 means one is already in flight. On any error, release all game controls.

```bash
curl -sS http://127.0.0.1:8088/v1/systemone -H 'Content-Type: application/json' -d '{"model":"qwen-visual-choice-v0","state_id":"minecraft","state":"Goal: reach the tree. Stop if a hostile mob appears.","questions":{"move":{"type":"choice","instructions":"What should I do for the next 200 ms?","criteria":{"forward":"Move forward","left":"Turn left","right":"Turn right","stop":"Release all controls"}}}}'
```

This extends the Jev wire shape with `state_id`, as the original API has no visual session reference. `probabilities` are a softmax over listed options, **not calibrated success probabilities**. `confidence` is normalized inverse entropy and likewise not a safety guarantee. `stop` should be checked against independent safeguards before any game action.

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

Run `.venv/bin/pytest -q visual-systemone/tests`. Unit tests mock vLLM and never require model weights. Live checks require the user's own Spark and image; no claim about FPS or image support of this exact build is made before those checks pass. The process stores frames only in RAM and does not persist history, but the normal Qwen service will still process and cache its own request state according to vLLM's scheduler.
