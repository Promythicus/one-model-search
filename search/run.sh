#!/usr/bin/env bash
cd "$(dirname "$0")"; PY=python3
[ -n "$WAIT_FOR" ] && until grep -q -e "^done" -e "server died" "$WAIT_FOR"; do sleep 30; done
$PY rr.py feats 2>&1 | grep -v -e pynvml -e "Loading weights"; $PY rr.py fit 2>&1 | grep -v pynvml; echo done
