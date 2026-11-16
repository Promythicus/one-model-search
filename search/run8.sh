#!/usr/bin/env bash
# Speed of the 27B's yes/no step when a question's 30 passages are sent together, by number of server slots.
cd "$(dirname "$0")"; PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BASE=${MODEL_GGUF}
for np in 1 4 8 16 30; do
  "$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 2048 -b 4096 -np $np -c $((np * 1024)) -ctk q4_0 -ctv q4_0 -m "$BASE" > out/rr27-server.log 2>&1 & SRV=$!
  until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV 2>/dev/null || { echo "server died np=$np"; tail -3 out/rr27-server.log; break; }; sleep 3; done
  kill -0 $SRV 2>/dev/null && $PY rr27.py speed "slots=$np" 2>&1 | grep -v pynvml; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 4
done; echo done
