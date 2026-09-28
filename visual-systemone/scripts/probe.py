#!/usr/bin/env python3
"""Read-only probes against the already running vLLM; never loads weights."""
import argparse
import base64
import json
import time
import urllib.error
import urllib.request


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        print(path, "HTTP", exc.code, exc.read(400).decode(errors="replace"))
        return None
    print(path, f"{(time.perf_counter()-started)*1000:.0f} ms", json.dumps(result, ensure_ascii=False)[:900])
    return result


parser = argparse.ArgumentParser()
parser.add_argument("frame", help="a small local JPEG with visually recognizable content")
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="qwen3.8-flash-next-a5b")
args = parser.parse_args()
base = args.url.rstrip("/")
image = "data:image/jpeg;base64," + base64.b64encode(open(args.frame, "rb").read()).decode()
messages = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": image}},
    {"type": "text", "text": "What is visible? Reply with one short sentence."}]}]
post(base, "/v1/chat/completions", {"model": args.model, "messages": messages,
                                    "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 40})
ids = []
for letter in "AB":
    data = post(base, "/tokenize", {"model": args.model, "prompt": letter,
                                     "add_special_tokens": False})
    if data and len(data.get("tokens", [])) == 1:
        ids.append(data["tokens"][0])
if len(ids) == 2:
    messages[0]["content"][1]["text"] = "Is there a visible character? A = yes, B = no. Letter:"
    post(base, "/v1/chat/completions", {"model": args.model, "messages": messages,
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 1,
        "logprob_token_ids": ids, "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False}})
print("Pooling probe intentionally omitted: /v1/embeddings is not evidence of image pooling unless its actual response to multimodal input is checked on the pinned server.")
