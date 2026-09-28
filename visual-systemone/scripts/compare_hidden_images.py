#!/usr/bin/env python3
"""Compare end-of-prompt Qwen states for A, B, A without restarting vLLM.

Requires the opt-in hidden capture route on the existing B12X server. Uses only
the Python standard library; never saves the image or the returned vectors.
"""

import argparse
import array
import base64
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_json(url, data=None):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url}: HTTP {exc.code}: {exc.read(300).decode(errors='replace')}") from exc


def capture(base, model, path, prompt):
    content = [{"type": "text", "text": prompt}]
    if path is not None:
        image = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": image}})
    reply = fetch_json(base + "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    req_id = reply["id"]
    route = base + "/flashnext/hidden_state/read?" + urllib.parse.urlencode({"req_id": req_id})
    result = fetch_json(route)
    if result["dtype"] != "float32" or result["shape"] != [2560]:
        raise RuntimeError(f"Unexpected vector format: {result['dtype']} {result['shape']}")
    vector = array.array("f")
    vector.frombytes(base64.b64decode(result["b64"], validate=True))
    if len(vector) != 2560 or not all(math.isfinite(value) for value in vector):
        raise RuntimeError("Invalid hidden vector")
    print(f"{path.name if path else 'text-only'}: {req_id}, shape={result['shape']}, dtype={result['dtype']}, output={reply['choices'][0]['message']['content']!r}")
    return vector


def distance(a, b):
    aa = sum(x * x for x in a)
    bb = sum(y * y for y in b)
    dot = sum(x * y for x, y in zip(a, b))
    cos = 1 - dot / math.sqrt(aa * bb)
    relative_l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / aa)
    return cos, relative_l2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, nargs="?", help="first JPEG")
    parser.add_argument("second", type=Path, nargs="?", help="different JPEG")
    parser.add_argument("--text-only", action="store_true", help="repeat one text prompt three times without images")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next-a5b")
    args = parser.parse_args()
    if args.text_only:
        if args.first is not None or args.second is not None:
            parser.error("Do not pass image paths with --text-only")
    else:
        if args.first is None or args.second is None:
            parser.error("Pass two JPEG paths or use --text-only")
        if args.first.resolve() == args.second.resolve():
            parser.error("Use two different image files")
        for image in (args.first, args.second):
            if not image.is_file():
                parser.error(f"Missing image: {image}")
    prompt = "Describe the visible shape in one short sentence."
    base = args.url.rstrip("/")
    first = capture(base, args.model, args.first, prompt)
    second = capture(base, args.model, args.second, prompt)
    repeat = capture(base, args.model, args.first, prompt)
    ab = distance(first, second)
    aa = distance(first, repeat)
    print(f"A vs B: cosine_distance={ab[0]:.8g}, relative_l2={ab[1]:.8g}")
    print(f"A vs A: cosine_distance={aa[0]:.8g}, relative_l2={aa[1]:.8g}")
    print("A vs A is a repeatability control; compare it with A vs B. No automatic threshold is assumed.")


if __name__ == "__main__":
    main()
