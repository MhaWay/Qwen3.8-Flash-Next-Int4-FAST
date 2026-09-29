import asyncio
from dataclasses import replace

import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from visual_systemone.app import FrameStore, app, attach_to_vllm, distribution


JPEG = b"\xff\xd8visual-test\xff\xd9"


def test_latest_frame_replaces_previous():
    async def scenario():
        store = FrameStore()
        first = await store.put("s", b"one")
        second = await store.put("s", b"two")
        assert second.seq == first.seq + 1
        assert (await store.get("s")).jpeg == b"two"
        assert len(store.frames) == 1
    asyncio.run(scenario())


def test_expired_session_restarts_seq_but_new_frame_has_new_identity():
    async def scenario():
        store = FrameStore()
        first = await store.put("s", b"one")
        store.frames["s"] = replace(first, received=first.received - 100)
        second = await store.put("s", b"two")
        assert first.seq == second.seq == 1
        assert first.revision != second.revision
    asyncio.run(scenario())


def test_candidate_probabilities_fail_closed():
    row = [{"token": "token_id:10", "logprob": -0.1},
           {"token": "token_id:20", "logprob": -2.1}]
    probs = distribution(row, [10, 20])
    assert probs[0] > probs[1]
    assert sum(probs) == pytest.approx(1)
    with pytest.raises(ValueError, match="missing"):
        distribution(row[:1], [10, 20])


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeVLLM:
    async def post(self, path, json):
        if path == "/tokenize":
            return FakeResponse({"tokens": [ord(json["prompt"])]})
        assert path == "/v1/chat/completions"
        assert json["logprob_token_ids"] == [65, 66]
        assert json["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return FakeResponse({"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "token_id:65", "logprob": -0.1},
            {"token": "token_id:66", "logprob": -2.1}]}]}}],
            "usage": {"prompt_tokens": 25}})

    async def aclose(self):
        pass


class MissingCandidate(FakeVLLM):
    async def post(self, path, json):
        response = await super().post(path, json)
        if path == "/v1/chat/completions":
            response.data["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        return response


class TextOnlyVLLM(FakeVLLM):
    async def post(self, path, json):
        if path == "/tokenize":
            return await super().post(path, json)
        assert path == "/v1/chat/completions"
        content = json["messages"][1]["content"]
        assert len(content) == 1 and content[0]["type"] == "text"
        assert "Which category?" in content[0]["text"]
        return FakeResponse({"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "token_id:65", "logprob": -0.1},
            {"token": "token_id:66", "logprob": -2.1}]}]}}],
            "usage": {"prompt_tokens": 25}})


class FrameReplacedAfterExpiry(FakeVLLM):
    async def post(self, path, json):
        if path == "/v1/chat/completions":
            store = app.state.store
            current = store.frames["minecraft"]
            store.frames["minecraft"] = replace(current, received=current.received - 100)
            await store.put("minecraft", JPEG)
        return await super().post(path, json)


def test_end_to_end_choice_one_current_frame():
    with TestClient(app) as client:
        app.state.client = FakeVLLM()
        upload = client.post("/v1/vision/minecraft/frame", data=JPEG,
                             headers={"Content-Type": "image/jpeg"})
        assert upload.status_code == 200
        body = {"state_id": "minecraft", "state": "Go to the tree", "questions": {
            "move": {"type": "choice", "instructions": "Next move?",
                     "criteria": {"forward": "Move forward", "stop": "Stop"}}}}
        result = client.post("/v1/systemone", json=body)
        assert result.status_code == 200, result.text
        answer = result.json()["answers"]["move"]
        assert answer["choice"] == "forward"
        assert result.json()["seq"] == upload.json()["seq"]
        assert client.post("/v1/vision/minecraft/frame", data=JPEG,
                           headers={"Content-Type": "image/jpeg"}).json()["seq"] == 2


@pytest.mark.parametrize("explicit_null", [False, True])
def test_text_only_systemone_without_frame(explicit_null):
    with TestClient(app) as client:
        app.state.client = TextOnlyVLLM()
        body = {"state": "The customer was billed twice.", "questions": {
            "team": {"type": "choice", "instructions": "Which category?",
                     "criteria": {"billing": "Billing", "other": "Other"}}}}
        if explicit_null:
            body["state_id"] = None
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["answers"]["team"]["choice"] == "billing"
        assert response.json()["state_id"] is None and response.json()["seq"] is None
        assert not app.state.store.frames


def test_supplied_session_still_requires_frame():
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json={
            "state_id": "absent", "state": "Test", "questions": {
                "team": {"type": "noul", "instructions": "Is this urgent?"}}})
        assert response.status_code == 409


def test_missing_candidate_returns_error_instead_of_action():
    with TestClient(app) as client:
        app.state.client = MissingCandidate()
        client.post("/v1/vision/minecraft/frame", content=JPEG,
                    headers={"Content-Type": "image/jpeg"})
        response = client.post("/v1/systemone", json={
            "state_id": "minecraft", "state": "Test", "questions": {
                "move": {"type": "choice", "instructions": "Move?",
                         "criteria": {"forward": "Forward", "stop": "Stop"}}}})
        assert response.status_code == 502


def test_expired_frame_replacement_during_decision_returns_conflict():
    with TestClient(app) as client:
        app.state.client = FrameReplacedAfterExpiry()
        client.post("/v1/vision/minecraft/frame", content=JPEG,
                    headers={"Content-Type": "image/jpeg"})
        response = client.post("/v1/systemone", json={
            "state_id": "minecraft", "state": "Test", "questions": {
                "move": {"type": "choice", "instructions": "Move?",
                         "criteria": {"forward": "Forward", "stop": "Stop"}}}})
        assert response.status_code == 409


def test_vllm_api_hosts_chat_and_systemone_on_one_port():
    parent = FastAPI()

    @parent.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "chat works"}}]}

    attach_to_vllm(parent)
    parent.state.systemone_client = TextOnlyVLLM()
    with TestClient(parent) as client:
        assert client.post("/v1/chat/completions").status_code == 200
        result = client.post("/v1/systemone", json={"state": "The customer was billed twice.",
            "questions": {"team": {"type": "choice", "instructions": "Which category?",
                                   "criteria": {"billing": "Billing", "other": "Other"}}}})
        assert result.status_code == 200, result.text
        assert result.json()["answers"]["team"]["choice"] == "billing"
        upload = client.post("/v1/vision/minecraft/frame", content=JPEG,
                             headers={"Content-Type": "image/jpeg"})
        assert upload.status_code == 200
        assert parent.state.systemone_service.store.frames["minecraft"].jpeg == JPEG
        parent.state.systemone_client = FakeVLLM()
        visual = client.post("/v1/systemone", json={"state_id": "minecraft", "state": "Look at the frame.",
            "questions": {"move": {"type": "choice", "instructions": "Move?",
                                   "criteria": {"forward": "Forward", "stop": "Stop"}}}})
        assert visual.status_code == 200, visual.text
        assert visual.json()["answers"]["move"]["choice"] == "forward"
    assert parent.state.systemone_service.letter_ids == [65, 66, 67, 68, 69, 70, 71, 72]


def test_vllm_route_collision_fails_before_serving():
    parent = FastAPI()

    @parent.post("/v1/systemone")
    async def already_registered():
        return {}

    with pytest.raises(RuntimeError, match="collision"):
        attach_to_vllm(parent)


def test_systemone_calls_chat_on_its_own_asgi_app():
    parent = FastAPI()

    @parent.post("/tokenize")
    async def tokenize(body: dict):
        return {"tokens": [ord(body["prompt"])]}

    @parent.post("/v1/chat/completions")
    async def chat(body: dict):
        assert len(body["messages"][1]["content"]) == 1  # text-only
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "token_id:65", "logprob": -0.1},
            {"token": "token_id:66", "logprob": -2.1}]}]}}],
            "usage": {"prompt_tokens": 25}}

    attach_to_vllm(parent)
    # ASGITransport simulates loopback on one server without binding another
    # port or spawning a model.
    async def run():
        transport = httpx.ASGITransport(app=parent)
        async with httpx.AsyncClient(transport=transport, base_url="http://self") as upstream:
            parent.state.systemone_client = upstream
            response = await upstream.post("/v1/systemone", json={"state": "A customer was billed twice.",
                "questions": {"team": {"type": "choice", "instructions": "Which category?",
                                       "criteria": {"billing": "Billing", "other": "Other"}}}})
            assert response.status_code == 200, response.text
            assert response.json()["answers"]["team"]["choice"] == "billing"
    asyncio.run(run())
