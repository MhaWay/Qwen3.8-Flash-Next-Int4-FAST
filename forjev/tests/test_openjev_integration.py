"""Run with a pinned OpenJEV checkout on PYTHONPATH (see README)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient


def test_real_openjev_contract_with_forjev(monkeypatch):
    pytest.importorskip("openjev.api")
    from forjev.forjev_backend import register

    register()
    from openjev.api import create_app
    from openjev.config import Settings

    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(body["prompt"])]})
        ids = body["logprob_token_ids"]
        top = [{"token": f"token_id:{i}", "logprob": 0.0 if i == ids[-1] else -5.0} for i in ids]
        return httpx.Response(200, json={"choices": [{"logprobs": {
            "content": [{"top_logprobs": top}]}}], "usage": {"prompt_tokens": 34}})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(respond), base_url=kw.get("base_url", "http://qwen")))
    settings = Settings(backend="forjev", upstream="http://qwen", upstream_model="qwen-a5b", warmup=False)
    with TestClient(create_app(settings)) as client:
        body = {"model": "jev-latest", "state": {"health": 20, "threat": "spider"},
                "questions": {"move": {"type": "choice", "instructions": {"goal": "survive"},
                                       "criteria": {"wait": {"action": "wait"}, "flee": {"action": "flee"}}}}}
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["answers"]["move"]["choice"] == "flee"
        assert response.json()["model"] == "forjev-qwen-next"
        assert response.json()["usage"] == {"input_tokens": 34, "output_tokens": 0}
        assert any(p == "/v1/chat/completions" for p, _ in calls)

        with_image = {**body, "images": ["data:image/jpeg;base64,/9j/2Q=="]}
        image_response = client.post("/v1/systemone", json=with_image)
        assert image_response.status_code == 200, image_response.text
        _, request_json = next((p, data) for p, data in reversed(calls) if p == "/v1/chat/completions")
        assert request_json["messages"][1]["content"][0]["type"] == "image_url"


def test_single_public_app_forwards_chat_stream(monkeypatch):
    pytest.importorskip("openjev.api")
    monkeypatch.setenv("OPENJEV_BACKEND", "forjev")
    monkeypatch.setenv("OPENJEV_UPSTREAM", "http://qwen")
    monkeypatch.setenv("OPENJEV_UPSTREAM_MODEL", "qwen-a5b")

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True and body["tools"][0]["function"]["name"] == "look"
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=httpx.ByteStream(b'data: {"delta":"ok"}\n\ndata: [DONE]\n\n'))

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(respond), base_url=kw.get("base_url", "http://qwen")))
    from forjev.openjev_app import app

    with TestClient(app) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert {m["name"] for m in models.json()["models"]} >= {"forjev-qwen-next", "qwen-a5b"}
        assert {m["id"] for m in models.json()["data"]} >= {"forjev-qwen-next", "qwen-a5b"}
        response = client.post("/v1/chat/completions", json={
            "model": "qwen-a5b", "stream": True, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "look", "parameters": {"type": "object"}}}],
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.content == b'data: {"delta":"ok"}\n\ndata: [DONE]\n\n'
