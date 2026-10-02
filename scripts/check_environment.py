#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"Kaggle: {os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'not detected')}")
    try:
        import streaming_asr
        print(f"REQUIRED streaming_asr: OK ({streaming_asr.__version__})")
    except Exception as exc:
        print(f"REQUIRED streaming_asr: NOT IMPORTABLE ({exc})")
        return 1
    for package in ("numpy", "yaml", "pytest", "soundfile", "webrtcvad", "torch", "transformers"):
        status = "installed" if importlib.util.find_spec(package) else "not installed"
        label = "REQUIRED" if package in {"numpy", "yaml", "pytest"} else "OPTIONAL"
        print(f"{label} {package}: {status}")
    for path in (Path("outputs"), Path("checkpoints")):
        print(f"Writable {path}: {path.exists() and os.access(path, os.W_OK)}")
    try:
        import torch
        print(f"GPU available: {torch.cuda.is_available()}")
        print(f"CUDA: {torch.version.cuda or 'not available'}")
    except ImportError:
        print("GPU/CUDA: unavailable (PyTorch not installed)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
