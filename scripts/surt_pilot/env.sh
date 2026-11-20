#!/usr/bin/env bash
# Source independently in every notebook shell cell.
export PROJECT=/kaggle/working/English-Streaming-ASR
export PILOT=/kaggle/working/surt-pilot-2
source /kaggle/working/surt-venv-py312/bin/activate
export PYTHONPATH="$PROJECT/scripts/surt_pilot:$PROJECT/scripts/surt_stream:$PROJECT/scripts/surt_mini:$PROJECT/src"
