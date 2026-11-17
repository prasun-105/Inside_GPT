#!/usr/bin/env sh
# Start the chat server and open it in the browser (macOS / Linux).
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" server.py --open "$@"
