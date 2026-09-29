"""Serve OpenJEV SystemOne and transparently forward Qwen chat on one public port.

Start with OPENJEV_BACKEND=forjev OPENJEV_UPSTREAM=http://127.0.0.1:8001
OPENJEV_UPSTREAM_MODEL=<vLLM served name> uvicorn forjev.openjev_app:app --host 0.0.0.0 --port 8000
The vLLM server must already be running on the private upstream port.
"""

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .forjev_backend import register

register()

from openjev.api import create_app  # noqa: E402: registration must precede API import
from openjev.config import Settings, served_models  # noqa: E402

settings = Settings()
if settings.backend != "forjev":
    raise RuntimeError("Set OPENJEV_BACKEND=forjev")
app = create_app(settings)


# OpenJEV returns {models: [...]}; OpenAI clients expect {object: list,
# data: [...]}. Preserve the former while also exposing the real Qwen chat
# model to existing clients on the single public origin.
for route in list(app.router.routes):
    if route.path == "/v1/models" and "GET" in getattr(route, "methods", set()):
        app.router.routes.remove(route)


@app.get("/v1/models")
async def combined_models():
    _, _, own = served_models(settings.backend)
    entries = own + [
        {"name": name, "description": "Routed System One backend", "release_date": ""}
        for name in settings.model_routes if name not in {m["name"] for m in own}
    ]
    entries.append({"name": settings.upstream_model,
                    "description": "Qwen chat model, also used by ForJev decisions", "release_date": ""})
    return {"models": entries, "object": "list", "data": [
        {"id": entry["name"], "object": "model", "owned_by": "forjev"}
        for entry in entries
    ]}


async def forward_chat(request: Request):
    """Keep vLLM's OpenAI chat payload/stream intact, including tools and usage."""
    client = httpx.AsyncClient(base_url=settings.upstream.rstrip("/"), trust_env=False,
                               timeout=httpx.Timeout(300, connect=5))
    try:
        headers = {name: value for name, value in request.headers.items()
                   if name.lower() not in {"host", "content-length", "transfer-encoding"}}
        upstream = await client.send(client.build_request("POST", "/v1/chat/completions",
                                    headers=headers, content=await request.body()), stream=True)
    except BaseException:
        await client.aclose()
        raise

    async def close():
        await upstream.aclose()
        await client.aclose()

    response_headers = {name: value for name, value in upstream.headers.items()
                        if name.lower() not in {"content-length", "transfer-encoding", "connection", "content-encoding"}}
    return StreamingResponse(upstream.aiter_bytes(), status_code=upstream.status_code,
                             headers=response_headers, background=BackgroundTask(close))


app.add_api_route("/v1/chat/completions", forward_chat, methods=["POST"], name="forjev_chat")
