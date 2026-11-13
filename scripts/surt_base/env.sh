#!/usr/bin/env bash
# Sourced in every Kaggle cell; no shell state is assumed to persist.
export WORK=/kaggle/working
export PROJECT="$WORK/English-Streaming-ASR"
export ICEFALL="$WORK/icefall-surt-base"
export RECIPE="$ICEFALL/egs/libricss/SURT"
export BASE="$WORK/surt-base-1a"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT/scripts/surt_base:$ICEFALL:$RECIPE/dprnn_zipformer"
source "$WORK/surt-venv-py312/bin/activate"
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, sysconfig; p=pathlib.Path(sysconfig.get_path("purelib")); assert (p/"nvidia/cuda_nvrtc/lib/libnvrtc.so.12").is_file(); print(":".join(str(d) for d in sorted((p/"nvidia").glob("*/lib"))))')"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
