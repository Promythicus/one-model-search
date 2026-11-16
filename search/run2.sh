#!/usr/bin/env bash
# The three rungs on the embedder, after rr.py has finished.
cd "$(dirname "$0")"; PY=python3
until grep -q "^done" out/run.log; do sleep 30; done
for s in base linear train; do $PY head.py $s 2>&1 | grep -v -e pynvml -e "Loading weights"; done; echo done
