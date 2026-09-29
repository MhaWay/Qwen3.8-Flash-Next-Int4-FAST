# ForJev prototype

ForJev is an OpenJEV backend for **one existing Qwen model in vLLM**. The
backend reads candidate next-token logprobs; each question still requires a
full prompt prefill and one sampling/logit step. It does not load weights or
use the hidden-state capture endpoint.

This lives on `feature/forjev` of the Qwen serving fork while its integration
and runtime contract are tested. The intended destination is an OpenJEV fork
and, after verification, a contribution that lists ForJev among its backends.

## Integration

Install OpenJEV from a pinned upstream checkout in a separate Python
environment with FastAPI, httpx and uvicorn. Put this repository on
`PYTHONPATH`; `forjev.openjev_app` calls `register()` before creating the
OpenJEV app. It inserts `forjev` into `ENGINES` and `ENCODER_MODELS`, so
OpenJEV's existing schema, auth, model aliases, errors and answer formatting
continue to apply. Set `OPENJEV_BACKEND=forjev`, `OPENJEV_UPSTREAM` to the
already running vLLM URL and `OPENJEV_UPSTREAM_MODEL` to its served name.

The intended **single public port** deployment puts OpenJEV/ForJev on `:8000`
and the same pinned Eugr vLLM image on an internal loopback port, e.g. `:8001`.
`/v1/systemone` is answered by ForJev; `/v1/chat/completions` is streamed
through to the same vLLM instance. This requires a planned restart of the
existing container, without pulling a new Eugr image. Do not switch live
serving before the tests against the pinned image and a rollback are ready.

## Status and limits

- Backend registration, structured state/instructions, OpenJEV question schema,
  images in the OpenJEV request, text/image choice logprobs and chat passthrough
  are implemented in this branch.
- Label IDs are discovered from the live tokenizer. This code has not yet been
  tested against 255-option requests on the pinned b12x server; missing tokens
  or candidate logprobs fail closed. Throughput, calibration and latency must
  be measured independently for Qwen.
- `steps`, multi-`samples`, `think` and `sequential` are OpenJEV extensions
  specific to its DiffusionGemma read mode. ForJev currently rejects them when
  non-default. This is not yet full OpenJEV feature parity.
- `GET /v1/models` returns OpenJEV's `models` and an OpenAI-style `data` list,
  including the Qwen chat model, so both client styles can discover it.
- Qwen4 compatibility is architectural intent. Tokenizer, served chat template,
  image encoding, candidate logprob support and label quality must be verified
  against the actual future model and vLLM build.

No second model download is required for this backend. JevK5's checkpoint,
prompt package and calibration temperature are deliberately not used.
