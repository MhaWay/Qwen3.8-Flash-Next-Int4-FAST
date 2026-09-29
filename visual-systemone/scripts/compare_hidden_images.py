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
import secrets
import statistics
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


def capture(base, model, path, prompt, fresh_cache=False, describe=False, trace=False):
    content = [{"type": "text", "text": prompt}]
    if path is not None:
        image = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": image}})
    request = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 96 if describe else 1,
        "temperature": 0.7 if describe else 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if describe:
        request.update({"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})
    if fresh_cache:
        # Each request has its own first-block hash, so KV blocks from earlier
        # requests cannot be reused. This does not change the prompt tokens.
        request["cache_salt"] = secrets.token_hex(16)
    reply = fetch_json(base + "/v1/chat/completions", request)
    req_id = reply["id"]
    endpoint = "trace" if trace else "read"
    route = base + "/flashnext/hidden_state/" + endpoint + "?" + urllib.parse.urlencode({"req_id": req_id})
    result = fetch_json(route)
    shape = result["shape"]
    valid_shape = (len(shape) == 2 and 0 < shape[0] <= 2048 and shape[1] == 2560) if trace else shape == [2560]
    if result["dtype"] != "float32" or not valid_shape:
        raise RuntimeError(f"Unexpected vector format: {result['dtype']} {result['shape']}")
    vector = array.array("f")
    vector.frombytes(base64.b64decode(result["b64"], validate=True))
    if len(vector) != math.prod(shape) or not all(math.isfinite(value) for value in vector):
        raise RuntimeError("Invalid hidden vector")
    print(f"{path.name if path else 'text-only'}: {req_id}, shape={result['shape']}, dtype={result['dtype']}, output={reply['choices'][0]['message']['content']!r}")
    if trace:
        tail_result = fetch_json(
            base + "/flashnext/hidden_state/read?" + urllib.parse.urlencode({"req_id": req_id})
        )
        if tail_result["shape"] != [2560] or tail_result["dtype"] != "float32":
            raise RuntimeError("Unexpected single-vector format during trace cross-check")
        tail = array.array("f")
        tail.frombytes(base64.b64decode(tail_result["b64"], validate=True))
        last_row = vector[-2560:]
        if len(tail) != 2560:
            raise RuntimeError("Invalid single-vector width during trace cross-check")
        max_delta = max(abs(x - y) for x, y in zip(last_row, tail))
        print(f"  trace[-1] vs /read: max_abs_delta={max_delta:.8g}")
    return vector, (shape[0] if trace else 1)


def distance(a, b):
    aa = sum(x * x for x in a)
    bb = sum(y * y for y in b)
    dot = sum(x * y for x, y in zip(a, b))
    cos = 1 - dot / math.sqrt(aa * bb)
    relative_l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / aa)
    return cos, relative_l2


def trace_summary(first, second, repeat, text_only=False):
    a, na = first
    b, nb = second
    again, nr = repeat
    if na != nr:
        raise RuntimeError(f"Same image changed prompt length: {na} vs {nr}")
    print(f"Prompt positions: A={na}, B={nb}, A-repeat={nr}; hidden_size=2560")
    if na != nb:
        print("A/B token counts differ: positions cannot be aligned directly; compare equal-size images.")
        return
    ab, aa = [], []
    for i in range(na):
        lo, hi = i * 2560, (i + 1) * 2560
        ab.append(distance(a[lo:hi], b[lo:hi])[0])
        aa.append(distance(a[lo:hi], again[lo:hi])[0])
    print(f"Per-position cosine distance: A/B mean={statistics.mean(ab):.6g} median={statistics.median(ab):.6g}; A/A mean={statistics.mean(aa):.6g} median={statistics.median(aa):.6g}")
    interesting = sorted(range(na), key=lambda i: ab[i] - aa[i], reverse=True)[:8]
    noisy = sorted(range(na), key=lambda i: aa[i], reverse=True)[:8]
    label = "Largest between-request differences" if text_only else "Largest image-specific differences"
    print(f"{label} (position: A/B, A/A):", ", ".join(f"{i}: {ab[i]:.4g}, {aa[i]:.4g}" for i in interesting))
    print("Least repeatable positions (position: A/B, A/A):", ", ".join(f"{i}: {ab[i]:.4g}, {aa[i]:.4g}" for i in noisy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, nargs="?", help="first JPEG")
    parser.add_argument("second", type=Path, nargs="?", help="different JPEG")
    parser.add_argument("--text-only", action="store_true", help="repeat one text prompt three times without images")
    parser.add_argument("--text-fill", type=int, default=0, help="with --text-only, prepend this many simple words to match a longer prompt")
    parser.add_argument("--fresh-cache", action="store_true", help="use a unique cache_salt per request")
    parser.add_argument("--describe", action="store_true", help="generate a short description at Qwen's non-thinking temperature 0.7")
    parser.add_argument("--trace", action="store_true", help="compare every prompt hidden row from the optional trace route")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next-a5b")
    args = parser.parse_args()
    if args.text_only:
        if args.first is not None or args.second is not None:
            parser.error("Do not pass image paths with --text-only")
        if not 0 <= args.text_fill <= 400:
            parser.error("--text-fill must be between 0 and 400")
    else:
        if args.text_fill:
            parser.error("--text-fill requires --text-only")
        if args.first is None or args.second is None:
            parser.error("Pass two JPEG paths or use --text-only")
        if args.first.resolve() == args.second.resolve():
            parser.error("Use two different image files")
        for image in (args.first, args.second):
            if not image.is_file():
                parser.error(f"Missing image: {image}")
    prompt = "Describe the visible shape in one short sentence."
    if args.text_fill:
        prompt = " ".join("object" for _ in range(args.text_fill)) + "\n" + prompt
    base = args.url.rstrip("/")
    print("prefix_cache:", "isolated per request" if args.fresh_cache else "normal")
    first = capture(base, args.model, args.first, prompt, args.fresh_cache, args.describe, args.trace)
    second = capture(base, args.model, args.second, prompt, args.fresh_cache, args.describe, args.trace)
    repeat = capture(base, args.model, args.first, prompt, args.fresh_cache, args.describe, args.trace)
    if args.trace:
        trace_summary(first, second, repeat, args.text_only)
    else:
        ab = distance(first[0], second[0])
        aa = distance(first[0], repeat[0])
        print(f"A vs B: cosine_distance={ab[0]:.8g}, relative_l2={ab[1]:.8g}")
        print(f"A vs A: cosine_distance={aa[0]:.8g}, relative_l2={aa[1]:.8g}")
    print("A vs A is a repeatability control; compare it with A vs B. No automatic threshold is assumed.")


if __name__ == "__main__":
    main()
