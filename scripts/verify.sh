#!/usr/bin/env bash
# One-shot verification run inside the "verify" Compose service.
# Aggregates: unit tests, application build/import check and a collision-aware
# API smoke test (including single-target-loss takeover certification).
# Exits non-zero if any stage fails.
set -u

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
status=0

echo "== [1/3] unit tests =="
"$PYTHON" -m pytest tests/ -q
pytest_rc=$?
[ "$pytest_rc" -eq 0 ] || { echo "unit tests failed ($pytest_rc)"; status=1; }

echo "== [2/3] application build / import check =="
"$PYTHON" -m compileall -q app scripts && "$PYTHON" -c "import app.main; print('import ok')"
build_rc=$?
[ "$build_rc" -eq 0 ] || { echo "build check failed ($build_rc)"; status=1; }

echo "== [3/3] API smoke (collision constraints + takeover certification) =="
"$PYTHON" scripts/smoke_api.py
smoke_rc=$?
[ "$smoke_rc" -eq 0 ] || { echo "API smoke failed ($smoke_rc)"; status=1; }

if [ "$status" -eq 0 ]; then
    echo "VERIFY: ALL CHECKS PASSED"
else
    echo "VERIFY: FAILURE"
fi
exit "$status"
