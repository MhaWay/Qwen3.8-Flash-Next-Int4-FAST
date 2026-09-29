#!/usr/bin/env python3
"""Compare question-conditioned end-of-prompt states for two letter images.

Uses the same messages and decision prompt as visual_systemone.app.systemone.
Omits logprobs (not needed for hidden-state comparison) because this pinned
vLLM build can fail while formatting them. A unique per-request cache_salt
isolates prefix reuse without changing prompt tokens. No restart is required.
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
def make_prompt(state, question, first_option, second_option):
    return (f"Current objective and memory: {state}\nQuestion: {question}\nOptions:\n"
            f"A. {first_option}\nB. {second_option}\nAnswer with one letter only:")


def capture(base, model, letter, image, prompt):
    encoded = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encoded}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 1,
        "temperature": 0,
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
    parser.add_argument("--ambiguous-a", action="store_true",
                        help="compare the old ambiguous symbol with the clear A using the original question")
    args = parser.parse_args()
    if args.ambiguous_a:
        first_name, second_name = "A-clear", "A-ambiguous"
        first_path, second_path = Path("systemone-A-clear.jpg"), Path("systemone-A.jpg")
        state = "Diagnostic image"
        question = "È visibile una A?"
        first_option, second_option = "Sì, è visibile una A", "No, non è visibile una A"
    else:
        first_name, second_name = "B", "D"
        first_path, second_path = args.b, args.d
        state, question = STATE, QUESTION
        first_option, second_option = "The letter B", "The letter D"
    if not first_path.is_file() or not second_path.is_file():
        parser.error(f"JPEG files must exist: {first_path}, {second_path}")
    prompt = make_prompt(state, question, first_option, second_option)
    print(f"Option A = {first_option}; option B = {second_option}")
    base = args.url.rstrip("/")
    data = [capture(base, args.model, label, path, prompt)
            for label, path in ((first_name, first_path), (second_name, second_path),
                                (first_name, first_path), (second_name, second_path))]
    if len({tokens for _, tokens in data}) != 1:
        raise RuntimeError("Prompt token counts differ; do not compare positions")
    first, second, first_repeat, second_repeat = (item[0] for item in data)
    for name, a, other in ((f"{first_name}/{second_name}", first, second),
                           (f"{first_name}/{first_name}", first, first_repeat),
                           (f"{second_name}/{second_name}", second, second_repeat),
                           (f"{second_name}/{first_name}", second_repeat, first_repeat)):
        cosine, relative_l2 = distance(a, other)
        print(f"{name}: cosine_distance={cosine:.8g}, relative_l2={relative_l2:.8g}")
    print("These are end-of-prompt states conditioned on the same SystemOne question.")


if __name__ == "__main__":
    main()
