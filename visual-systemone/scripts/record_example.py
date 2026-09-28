#!/usr/bin/env python3
"""Capture a labeled state/action pair for future CLM-style head training.

Run on the desktop with the screenshot stored locally. The image path is copied
into a dataset directory; the Spark is not needed for this collection step.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
parser.add_argument("--dataset", type=Path, default=Path("visual-dataset"))
parser.add_argument("--goal", required=True)
parser.add_argument("--question", required=True)
parser.add_argument("--actions", nargs="+", required=True)
parser.add_argument("--chosen", required=True)
parser.add_argument("--split", choices=["train", "val", "test"], required=True)
args = parser.parse_args()
if len(args.actions) < 2 or args.chosen not in args.actions:
    parser.error("chosen must be one of two or more actions")
raw = args.image.read_bytes()
if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
    parser.error("image must be a JPEG")
digest = hashlib.sha256(raw).hexdigest()
out = args.dataset / "frames" / f"{digest}.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(raw)
row = {"image": f"frames/{digest}.jpg", "goal": args.goal, "question": args.question,
       "actions": args.actions, "chosen": args.chosen, "split": args.split}
with (args.dataset / "labels.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(out)
