#!/usr/bin/env bash
# The 27B as its own reranker (rr27.py), after run6.sh has chosen the layer.
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}
"$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 512 -np 1 -c 8192 -ctk q4_0 -ctv q4_0 -m "$BASE" > out/rr27-server.log 2>&1 & SRV=$!
until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 out/rr27-server.log; exit 1; }; sleep 3; done
$PY rr27.py ask $1 2>&1 | grep -v pynvml; kill $SRV; wait $SRV 2>/dev/null
$PY rr27.py score 2>&1 | grep -v pynvml; echo done
