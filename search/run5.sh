#!/usr/bin/env bash
# The 27B's own states as a search model (own27.py). Waits for run4.sh.
cd "$(dirname "$0")"; PY=python3; BASE=${MODEL_GGUF}
until grep -q "^done" out/run4.log; do sleep 60; done
echo "$(date +%H:%M) start reranker/run5.sh" >> overnight.log
for f in mean eol; do
  PROBE_PROMPTS=out/own27/$f/in.jsonl PROBE_OUT=out/own27/$f PROBE_NAMES=l_out PROBE_LAYERS=24,32,40,48,56,63 ../tools/probe -m "$BASE" -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe: done" -e rror
done
$PY own27.py score 2>&1 | grep -v pynvml
echo "$(date +%H:%M) end reranker/run5.sh" >> overnight.log; echo done
