#!/usr/bin/env bash
# End to end with one model (e2e27.py). Waits for run8.sh. The answering server is the usual one (launcher :8797, model :8081).
cd "$(dirname "$0")"; PY=python3; BASE=${MODEL_GGUF}
BIN=~/models/llama.cpp/src/build/bin/llama-server
up() { curl -s -X POST "http://127.0.0.1:8797/start?mode=long&minutes=300" > /dev/null; until curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 5; done; }
down() { curl -s -X POST http://127.0.0.1:8797/stop > /dev/null; until ! curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 3; done; sleep 5; }
if [ ! -f out/e2e/meta.json ]; then $PY e2e27.py prompts 2>&1 | grep -v pynvml
PROBE_PROMPTS=out/e2e/in.jsonl PROBE_OUT=out/e2e PROBE_NAMES=l_out PROBE_LAYERS=36 ../tools/probe -m "$BASE" -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe: done" -e rror
fi
"$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 2048 -b 4096 -np 1 -c 4096 --cache-ram 0 -ctk q4_0 -ctv q4_0 -m "$BASE" > out/rr27-server.log 2>&1 & SRV=$!
until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 3; done
$PY e2e27.py orders 2>&1 | grep -v pynvml; kill $SRV; wait $SRV 2>/dev/null; sleep 5
up
for m in states12 own12 own24; do $PY e2e27.py run $m 2>&1 | grep -v pynvml; done; down
$PY e2e27.py score 2>&1 | grep -v pynvml; echo done
