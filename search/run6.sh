#!/usr/bin/env bash
# The 27B's own states as the search model: layer chosen on the practice contracts, then read on the held-out ones.
cd "$(dirname "$0")"; PY=python3; BASE=${MODEL_GGUF}
$PY own27.py prompts2 2>&1 | grep -v pynvml
for f in practice held; do
  PROBE_PROMPTS=out/own27/pick/$f/in.jsonl PROBE_OUT=out/own27/pick/$f PROBE_NAMES=l_out PROBE_LAYERS=24,28,32,36,38,40,42,44,48,52,56,60 ../tools/probe -m "$BASE" -ngl 99 -c 4096 -b 4096 -ub 512 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep -e "probe: done" -e rror
done
$PY own27.py pick 2>&1 | grep -v pynvml
echo done
