#!/usr/bin/env bash
# The general, released CLM head (no fitting on contracts). Waits for the night queue and for the downloads.
cd "$(dirname "$0")"; PY=python3
until grep -q "all done" overnight.log && grep -q "downloads done" ../models/dl-clm.log; do sleep 60; done
echo "$(date +%H:%M) start reranker/run4.sh" >> overnight.log
$PY clm.py embed 2>&1 | grep -v -e pynvml -e "Loading weights"; $PY clm.py score 2>&1 | grep -v -e pynvml -e "Loading weights"
echo "$(date +%H:%M) end reranker/run4.sh" >> overnight.log; echo done
