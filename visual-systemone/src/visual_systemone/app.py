"""A bounded visual snapshot store and Qwen token-logprob decision adapter.

One in-memory process, local/private network only. The vLLM server remains the
only process loading Qwen. This is a conditional-choice prototype, not CLM.
"""

import asyncio
import base64
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

VLLM_URL = os.environ.get("VISUAL_VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("VISUAL_MODEL", os.environ.get("FLASHNEXT_SYSTEMONE_MODEL", "qwen3.8-flash-next-a5b"))
MAX_BYTES = 300_000
MAX_SESSIONS = 8
FRAME_TTL = 10.0
LETTERS = "ABCDEFGHIJKLMNOPQRST"


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    seq: int
    received: float
    revision: int


class FrameStore:
    def __init__(self) -> None:
        self.frames: dict[str, Frame] = {}
        self.lock = asyncio.Lock()
        self.next_revision = 0

    async def put(self, session: str, jpeg: bytes) -> Frame:
        async with self.lock:
            now = time.monotonic()
            self.frames = {k: v for k, v in self.frames.items() if now - v.received < FRAME_TTL}
            if session not in self.frames and len(self.frames) >= MAX_SESSIONS:
                raise HTTPException(429, "Session limit reached")
            self.next_revision += 1
            frame = Frame(jpeg, self.frames[session].seq + 1 if session in self.frames else 1,
                          now, self.next_revision)
            self.frames[session] = frame  # replace; never append a visual history
            return frame

    async def get(self, session: str) -> Frame:
        async with self.lock:
            frame = self.frames.get(session)
            if frame is None or time.monotonic() - frame.received >= FRAME_TTL:
                raise HTTPException(409, "No current frame")
            return frame


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=512)
    criteria: dict[str, str] | list[str] | None = None


class SystemOneRequest(BaseModel):
    model: str = "qwen-visual-choice-v0"
    state: str = Field(max_length=2048)
    # Omit or set null for text-only decisions. A supplied id still requires a
    # current JPEG frame, preserving the existing session semantics.
    state_id: str | None = Field(default=None, min_length=1, max_length=80)
    questions: dict[str, Question] = Field(min_length=1, max_length=4)


def choices(q: Question) -> tuple[list[str], list[str]]:
    if q.type == "noul":
        return ["true", "false"], ["Yes", "No"]
    if q.type == "choice" and isinstance(q.criteria, dict):
        keys = list(q.criteria)
        descriptions = [q.criteria[k] or k for k in keys]
    elif q.type == "score" and isinstance(q.criteria, list):
        keys = [str(i) for i in range(len(q.criteria))]
        descriptions = q.criteria
    else:
        raise HTTPException(400, "Invalid criteria for question type")
    if not 2 <= len(keys) <= len(LETTERS) or any(not x for x in descriptions):
        raise HTTPException(400, "Use 2-20 nonempty options")
    return keys, descriptions


def distribution(rows: list[dict[str, Any]], ids: list[int]) -> list[float]:
    logp: dict[int, float] = {}
    for row in rows:
        token = str(row.get("token", ""))
        if token.startswith("token_id:"):
            token_id = int(token.split(":", 1)[1])
            logp[token_id] = float(row["logprob"])
    if any(token_id not in logp or not math.isfinite(logp[token_id]) for token_id in ids):
        raise ValueError("Candidate logprobs missing or non-finite")
    values = [logp[token_id] for token_id in ids]
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    return [w / sum(weights) for w in weights]


def answer(q: Question, keys: list[str], probs: list[float]) -> dict[str, Any]:
    if q.type == "noul":
        return {"type": "noul", "noul": probs[0]}
    entropy = -sum(p * math.log(p) for p in probs if p)
    confidence = max(0.0, 1.0 - entropy / math.log(len(probs)))
    if q.type == "choice":
        return {"type": "choice", "choice": keys[max(range(len(probs)), key=probs.__getitem__)],
                "probabilities": dict(zip(keys, probs)), "confidence": confidence}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
            "legend": dict(enumerate(q.criteria or [])),
            "probabilities": dict(zip(keys, probs)), "confidence": confidence}


async def upstream(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"vLLM request failed: {type(exc).__name__}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=httpx.Timeout(90, connect=5), trust_env=False)
    app.state.store = FrameStore()
    app.state.gates = {}
    app.state.letter_ids = None
    yield
    await app.state.client.aclose()


app = FastAPI(title="Qwen visual System One prototype", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/v1/vision/{state_id}/frame")
async def add_frame(state_id: str, request: Request):
    if not 1 <= len(state_id) <= 80 or not all(c.isalnum() or c in "-_" for c in state_id):
        raise HTTPException(400, "Invalid state_id")
    if request.headers.get("content-type", "").split(";")[0] != "image/jpeg":
        raise HTTPException(415, "Expected image/jpeg")
    jpeg = await request.body()
    if not 4 <= len(jpeg) <= MAX_BYTES or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise HTTPException(413, "Invalid or oversized JPEG")
    frame = await _service(request.app).store.put(state_id, jpeg)
    return {"state_id": state_id, "seq": frame.seq}


def _service(parent: FastAPI):
    return parent.state if parent is app else parent.state.systemone_service


async def letter_ids(client: httpx.AsyncClient, state) -> list[int]:
    if state.letter_ids is None:
        ids = []
        for letter in LETTERS:
            tokens = (await upstream(client, "/tokenize", {
                "model": MODEL, "prompt": letter, "add_special_tokens": False})).get("tokens", [])
            if len(tokens) != 1:
                raise HTTPException(502, f"Label {letter} is not one token")
            ids.append(int(tokens[0]))
        if len(set(ids)) != len(ids):
            raise HTTPException(502, "Labels have overlapping token IDs")
        state.letter_ids = ids
    return state.letter_ids


@app.post("/v1/systemone")
async def systemone(body: SystemOneRequest, request: Request):
    if body.model != "qwen-visual-choice-v0":
        raise HTTPException(400, "Unknown model")
    state = _service(request.app)
    store: FrameStore = state.store
    frame = await store.get(body.state_id) if body.state_id is not None else None
    # One request per session: the image can change while Qwen is answering.
    gate = (state.gates.setdefault(body.state_id, asyncio.Lock())
            if body.state_id is not None else asyncio.Lock())
    if gate.locked():
        raise HTTPException(429, "Previous decision still running")
    async with gate:
        # Standalone gateway owns a persistent client in its lifespan. When
        # attached to vLLM's FastAPI app, use a scoped loopback client instead:
        # vLLM owns that app's lifespan and we must not replace it.
        client = (request.app.state.client if request.app is app else
                  getattr(request.app.state, "systemone_client", None))
        if client is None:
            async with httpx.AsyncClient(base_url=VLLM_URL, timeout=httpx.Timeout(90, connect=5), trust_env=False) as scoped:
                return await _evaluate(body, request, frame, store, scoped)
        return await _evaluate(body, request, frame, store, client)


async def _evaluate(body: SystemOneRequest, request: Request, frame: Frame | None,
                    store: FrameStore, client: httpx.AsyncClient):
        ids = await letter_ids(client, _service(request.app))
        image_content = ([{"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode()}}]
            if frame is not None else [])
        answers: dict[str, Any] = {}
        input_tokens = 0
        for name, q in body.questions.items():
            keys, descriptions = choices(q)
            options = "\n".join(f"{LETTERS[i]}. {desc}" for i, desc in enumerate(descriptions))
            prompt = f"Current objective and memory: {body.state}\nQuestion: {q.instructions}\nOptions:\n{options}\nAnswer with one letter only:"
            result = await upstream(client, "/v1/chat/completions", {
                "model": MODEL, "messages": [
                    {"role": "system", "content": "Evaluate the current image. Select exactly one listed action or answer."},
                    {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 1,
                "logprob_token_ids": ids[:len(keys)], "return_tokens_as_token_ids": True})
            try:
                rows = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                probs = distribution(rows, ids[:len(keys)])
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                raise HTTPException(502, "Candidate logprobs unavailable; stop the controller") from exc
            answers[name] = answer(q, keys, probs)
            input_tokens += result.get("usage", {}).get("prompt_tokens", 0)
        if frame is not None:
            current = await store.get(body.state_id)
            if current.revision != frame.revision:
                raise HTTPException(409, "New frame arrived during decision; retry")
        return {"model": body.model, "state_id": body.state_id,
                "seq": frame.seq if frame is not None else None,
                "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 1 * len(answers)}}


def attach_to_vllm(parent: FastAPI) -> None:
    """Register two paths on the existing vLLM API process and port.

    No extra Qwen engine, proxy listener, or replacement of vLLM's lifespan.
    This is called by the pinned b12x router patch before the server starts.
    """
    paths = {"/v1/systemone", "/v1/vision/{state_id}/frame"}
    if any(route.path in paths for route in parent.routes):
        raise RuntimeError("SystemOne route collision")
    parent.state.systemone_service = SimpleNamespace(store=FrameStore(), gates={}, letter_ids=None)
    parent.add_api_route("/v1/vision/{state_id}/frame", add_frame, methods=["POST"],
                         name="systemone_frame")
    parent.add_api_route("/v1/systemone", systemone, methods=["POST"],
                         name="systemone_decision")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("VISUAL_PORT", "8088")))
