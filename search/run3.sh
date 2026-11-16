#!/usr/bin/env bash
# Honesty check on the closed-form add-on: does it help on kinds of question it was never fitted on?  -> out/types.json
cd "$(dirname "$0")"; python3 head.py types 2>&1 | grep -v -e pynvml -e "Loading weights"; echo done
