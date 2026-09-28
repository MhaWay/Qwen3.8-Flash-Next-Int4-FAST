#!/usr/bin/env python3
"""Send one local JPEG to the private snapshot endpoint."""
import argparse
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("frame")
parser.add_argument("--session", default="minecraft")
parser.add_argument("--url", default="http://127.0.0.1:8088")
args = parser.parse_args()
req = urllib.request.Request(f"{args.url.rstrip('/')}/v1/vision/{args.session}/frame",
                             open(args.frame, "rb").read(), {"Content-Type": "image/jpeg"})
print(urllib.request.urlopen(req, timeout=10).read().decode())
