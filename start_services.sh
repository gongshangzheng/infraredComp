#!/usr/bin/env bash
# 一键启动 infraredComp 后端(FastAPI :8091)+ 前端(Vite :3001)
# 用法: bash start_services.sh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 选择可用的 node(pnpm 需要 Node 22+;系统默认 node 20 会崩)
pick_node() {
  for c in /opt/homebrew/opt/node@25/bin/node \
           /opt/homebrew/opt/node@24/bin/node \
           /opt/homebrew/opt/node@22/bin/node; do
    if [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  echo "$(command -v node)"
}
NODE_BIN="$(pick_node)"
export PATH="$(dirname "$NODE_BIN"):$PATH"
echo "[start] node: $("$NODE_BIN" -v)  pnpm: $(command -v pnpm)"

# 后端
echo "[start] backend: uv run uvicorn server.main:app --port 8091 (logs: backend.log)"
uv run uvicorn server.main:app --host 0.0.0.0 --port 8091 > backend.log 2>&1 &
BACKEND_PID=$!
echo "[start] backend pid=$BACKEND_PID"

# 前端
echo "[start] frontend: cd web && pnpm dev (port 3001)"
( cd web && pnpm dev --port 3001 ) > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "[start] frontend pid=$FRONTEND_PID"

trap "echo '[stop] terminating'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT INT TERM

echo ""
echo "  backend  -> http://localhost:8091  (docs /api/docs)"
echo "  frontend -> http://localhost:3001/infraredComp/"
echo "  logs     -> backend.log  frontend.log"
echo ""
wait
