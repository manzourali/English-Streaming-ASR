#!/usr/bin/env bash
source /kaggle/working/English-Streaming-ASR/scripts/surt_base/env.sh
export MINI=/kaggle/working/surt-mini-1b
test -f "$MINI/active-run.txt"
export MINI_RUN="$(cat "$MINI/active-run.txt")"
[[ "$MINI_RUN" == "$MINI"/runs/* ]]
test -d "$MINI_RUN"
export PYTHONPATH="$PROJECT/scripts/surt_mini:$PYTHONPATH"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
