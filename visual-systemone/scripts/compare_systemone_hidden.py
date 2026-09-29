#!/usr/bin/env python3
"""Compare question-conditioned end-of-prompt states for B, D, B, D.

Uses the same messages and decision prompt as visual_systemone.app.systemone.
Only the per-request cache_salt differs, isolating prefix-cache reuse without
changing any prompt tokens. Does not modify the server or require a restart.
"""

import argparse
import array
import base64
import math
import secrets
import urllib.parse
from pathlib import Path

from compare_hidden_images import distance, fetch_json


STATE = "Classify the uppercase letter visible in the image."
QUESTION = "Which uppercase letter is visible?"
SYSTEM = "Evaluate the current image. Select exactly one listed action or answer."
PROMPT = (f"Current objective and memory: {STATE}\nQuestion: {QUESTION}\nOptions:\n"
          "A. The letter B\nB. The letter D\nAnswer with one letter only:")


def capture(base, model, letter, image, option_ids):
    encoded = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encoded}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
        "logprob_token_ids": option_ids,
        "return_tokens_as_token_ids": True,
        "cache_salt": secrets.token_hex(16),
    }
    reply = fetch_json(base + "/v1/chat/completions", request)
    req_id = reply["id"]
    route = base + "/flashnext/hidden_state/read?" + urllib.parse.urlencode({"req_id": req_id})
    result = fetch_json(route)
    if result["shape"] != [2560] or result["dtype"] != "float32":
        raise RuntimeError(f"Unexpected hidden-state shape/dtype: {result['shape']}, {result['dtype']}")
    vector = array.array("f")
    vector.frombytes(base64.b64decode(result["b64"], validate=True))
    if len(vector) != 2560 or not all(math.isfinite(x) for x in vector):
        raise RuntimeError("Invalid hidden vector")
    response = reply["choices"][0]["message"]["content"]
    tokens = reply.get("usage", {}).get("prompt_tokens")
    print(f"{letter}: id={req_id}, answer={response!r}, prompt_tokens={tokens}")
    return vector, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next-a5b")
    parser.add_argument("--b", type=Path, default=Path("systemone-B-clear.jpg"))
    parser.add_argument("--d", type=Path, default=Path("systemone-D-clear.jpg"))
    args = parser.parse_args()
    if not args.b.is_file() or not args.d.is_file():
        parser.error("B and D JPEG files must exist")
    base = args.url.rstrip("/")
    option_ids = []
    for label in "AB":
        ids = fetch_json(base + "/tokenize", {
            "model": args.model, "prompt": label, "add_special_tokens": False,
        })["tokens"]
        if len(ids) != 1:
            raise RuntimeError(f"Option label {label} must be one token")
        option_ids.append(int(ids[0]))
    data = [capture(base, args.model, label, path, option_ids)
            for label, path in (("B", args.b), ("D", args.d), ("B", args.b), ("D", args.d))]
    if len({tokens for _, tokens in data}) != 1:
        raise RuntimeError("Prompt token counts differ; do not compare positions")
    b, d, b_repeat, d_repeat = (item[0] for item in data)
    for name, a, other in (("B/D", b, d), ("B/B", b, b_repeat),
                           ("D/D", d, d_repeat), ("D/B", d_repeat, b_repeat)):
        cosine, relative_l2 = distance(a, other)
        print(f"{name}: cosine_distance={cosine:.8g}, relative_l2={relative_l2:.8g}")
    print("These are end-of-prompt states conditioned on the same SystemOne question.")


if __name__ == "__main__":
    main()
