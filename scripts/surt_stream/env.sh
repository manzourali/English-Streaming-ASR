#!/usr/bin/env bash
# Source in each cell; do not inherit notebook shell state.
source /kaggle/working/English-Streaming-ASR/scripts/surt_base/env.sh
export STREAM=/kaggle/working/surt-stream-1c
export MINI_PRIOR=/kaggle/working/surt-mini-1b/runs/mini-20261119-d
export PYTHONPATH="$PROJECT/scripts/surt_stream:$PROJECT/scripts/surt_mini:$PYTHONPATH"
test -s "$STREAM/active-run.txt"
export STREAM_RUN="$(cat "$STREAM/active-run.txt")"
[[ "$STREAM_RUN" == "$STREAM"/runs/* ]]
test -d "$STREAM_RUN/logs"
