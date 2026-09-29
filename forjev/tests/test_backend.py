import asyncio
import json
import sys
import types
from types import SimpleNamespace

import httpx
import pytest

from forjev import forjev_backend


class SchemaError(Exception):
    def __init__(self, message, loc=None):
        super().__init__(message)
        self.loc = loc


class EncoderEngine:
    max_choices = 255

    def build_schema(self, questions):
        qs, forced = [], {}
        for name, q in questions.items():
            criteria = q.get("criteria") or {}
            if q["type"] == "noul":
                options = [("yes", "Yes"), ("no", "No")]
            elif q["type"] == "score":
                options = [(str(i), label) for i, label in enumerate(criteria)]
            else:
                options = list(criteria.items())
            if len(options) > self.max_choices:
                raise SchemaError("Too many choices")
            qs.append({"key": name, "choices": options, "instructions": q["instructions"],
                       "type": q["type"]})
        return qs, forced


def installed(monkeypatch, responses):
    package = types.ModuleType("openjev")
    package.__path__ = []
    config = types.ModuleType("openjev.config")
    config.ENCODER_MODELS = {}
    encoders = types.ModuleType("openjev.encoders")
    encoders.EncoderEngine = EncoderEngine
    encoders.ENGINES = {}
    engine = types.ModuleType("openjev.engine")
    engine.Overloaded = type("Overloaded", (Exception,), {})
    engine.SchemaError = SchemaError
    engine.Upstream = type("Upstream", (Exception,), {})
    engine.model_ns = SimpleNamespace(get=lambda: None)
    engine.to_answer = lambda q, p: {"choice": q["choices"][p.index(max(p))][0],
                                     "probabilities": dict(zip([x[0] for x in q["choices"]], p))}
    for name, module in {"openjev": package, "openjev.config": config,
                         "openjev.encoders": encoders, "openjev.engine": engine}.items():
        monkeypatch.setitem(sys.modules, name, module)

    sent = []

    def respond(req):
        body = json.loads(req.content)
        sent.append((req.url.path, body))
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(body["prompt"])]})
        ids = body["logprob_token_ids"]
        top = [{"token": f"token_id:{i}", "logprob": -0.01 if i == ids[-1] else -5.0} for i in ids]
        if responses.get("missing"):
            top.pop()
        return httpx.Response(200, json={"choices": [{"logprobs": {
            "content": [{"top_logprobs": top}]}}], "usage": {"prompt_tokens": 50}})

    client = httpx.AsyncClient
    monkeypatch.setattr(forjev_backend.httpx, "AsyncClient", lambda **kw: client(
        transport=httpx.MockTransport(respond), base_url=kw["base_url"]))
    backend = forjev_backend.register()
    assert encoders.ENGINES["forjev"] is backend
    assert config.ENCODER_MODELS["forjev"]["name"] == forjev_backend.MODEL_NAME
    return backend, sent


def test_structured_state_and_image_use_the_same_qwen(monkeypatch):
    backend, sent = installed(monkeypatch, {})

    async def run():
        service = backend(SimpleNamespace(upstream="http://localhost:8001", upstream_model="qwen-a5b",
                                          max_inflight=3, max_queue=8))
        try:
            answers, tokens, output = await service.decide(
                {"action": {"type": "choice", "instructions": "Next action?",
                            "criteria": {"wait": "Wait", "click": "Click"}}},
                {"screen": "button"}, 0,
                images=[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}}])
            assert answers["action"]["choice"] == "click"
            assert sum(answers["action"]["probabilities"].values()) == pytest.approx(1)
            assert (tokens, output) == (50, 0)
            calls = [body for path, body in sent if path == "/v1/chat/completions"]
            assert len(calls) == 1 and calls[0]["model"] == "qwen-a5b"
            assert calls[0]["max_tokens"] == 1 and len(calls[0]["logprob_token_ids"]) == 2
            assert calls[0]["messages"][1]["content"][0]["type"] == "image_url"
            assert '"screen": "button"' in calls[0]["messages"][1]["content"][1]["text"]
        finally:
            await service.close()

    asyncio.run(run())


def test_missing_candidate_logprob_fails_closed(monkeypatch):
    backend, _ = installed(monkeypatch, {"missing": True})

    async def run():
        service = backend(SimpleNamespace(upstream="http://localhost:8001", upstream_model="qwen-a5b",
                                          max_inflight=3, max_queue=8))
        try:
            with pytest.raises(Exception, match="incomplete candidate logprobs"):
                await service.decide({"q": {"type": "choice", "instructions": "Choose",
                                            "criteria": {"left": "Left", "right": "Right"}}}, "state", 0)
        finally:
            await service.close()

    asyncio.run(run())
