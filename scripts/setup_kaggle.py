#!/usr/bin/env python3
"""Install lightweight notebook extras while constraining the existing ML stack.

Run with the notebook kernel's sys.executable. No CUDA wheels or model runtime
are installed here. A resolver conflict is an error requiring investigation.
"""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "torch", "torchaudio", "torchvision", "tensorflow", "numpy", "pandas",
    "scipy", "scikit-learn", "transformers", "opencv-python",
    "opencv-python-headless", "Pillow", "soundfile", "librosa",
    "huggingface_hub", "tokenizers", "safetensors", "sentencepiece", "PyYAML",
)


def installed_constraints() -> dict[str, str]:
    versions = {}
    for name in PROTECTED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/kaggle/working/asr_artifacts"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    before = installed_constraints()
    constraints = args.artifacts.resolve() / "installed-stack-constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in before.items()), encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--only-binary=:all:",
        "--constraint", str(constraints), "--editable", f"{ROOT}[kaggle,dev]",
    ], check=True)
    after = installed_constraints()
    changed = {name: (version, after.get(name)) for name, version in before.items()
               if after.get(name) != version}
    if changed:
        raise RuntimeError(f"Protected packages changed; start a fresh session: {changed}")
    print(f"Notebook extras installed; {len(before)} existing stack versions preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
