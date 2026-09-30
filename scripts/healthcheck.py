"""Container health probe: passes only once the API reports ready.

Readiness flips after the application's startup self-check (geometry kernel
+ a minimal end-to-end adjudication), so this gate cannot pass before the
adjudication endpoint can accept requests.
"""
import json
import os
import sys
import urllib.request

port = os.environ.get("APP_PORT", "8000")
try:
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health/ready", timeout=2) as resp:
        if resp.status == 200 and json.load(resp).get("status") == "ready":
            sys.exit(0)
except Exception:  # noqa: BLE001 - any failure means "not healthy yet"
    pass
sys.exit(1)
