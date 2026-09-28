import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from visual_systemone.app import FrameStore, app, distribution


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
