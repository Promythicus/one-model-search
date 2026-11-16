#!/usr/bin/env bash
# Speed of the yes/no step: a shorter prompt (no system prompt, the clause's name only), on every third held-out contract,
# against the full prompt on the same contracts (from the stored scores). One slot, RAM prompt cache off, calls ordered by passage.
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}
RR27_EVERY=3 $PY rr27.py score 2>&1 | grep -v pynvml
"$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 2048 -b 4096 -np 1 -c 4096 --cache-ram 0 -ctk q4_0 -ctv q4_0 -m "$BASE" > out/rr27-server.log 2>&1 & SRV=$!
until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 3; done
RR27_VARIANT=short RR27_EVERY=3 $PY rr27.py ask 2>&1 | grep -v pynvml; kill $SRV; wait $SRV 2>/dev/null
RR27_VARIANT=short RR27_EVERY=3 $PY rr27.py score 2>&1 | grep -v pynvml; echo done
