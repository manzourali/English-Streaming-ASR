#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def _package(name: str) -> dict[str, str]:
    module = importlib.util.find_spec(name)
    if not module:
        return {"status": "not installed"}
    distribution = {"yaml": "PyYAML", "webrtcvad": "webrtcvad-wheels"}.get(name, name)
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"status": "installed", "version": version}


def environment_report() -> tuple[dict[str, object], bool]:
    report: dict[str, object] = {"python": sys.version.split()[0], "os": platform.platform(), "kaggle": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "not detected"), "packages": {}, "writable": {}, "gpu": {}}
    importable = True
    try:
        import streaming_asr
        report["project_import"] = {"status": "OK", "version": streaming_asr.__version__}
    except Exception as exc:
        report["project_import"] = {"status": "NOT IMPORTABLE", "error": str(exc)}; importable = False
    for package in ("numpy", "yaml", "pytest", "soundfile", "webrtcvad", "torch", "transformers", "datasets", "peft", "huggingface_hub"):
        report["packages"][package] = _package(package)
    required_ok = all(report["packages"][package]["status"] == "installed" for package in ("numpy", "yaml", "pytest"))
    for path in (Path("outputs"), Path("checkpoints")):
        report["writable"][str(path)] = path.exists() and os.access(path, os.W_OK)
    try:
        import torch
        report["gpu"] = {"torch": torch.__version__, "available": torch.cuda.is_available(), "cuda": torch.version.cuda or "not available", "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        report["gpu"] = {"status": "unavailable (PyTorch not installed)"}
    return report, importable and required_ok


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Require optional model/dataset packages as well as project core packages")
    args = parser.parse_args()
    report, core_ok = environment_report()
    optional = ("soundfile", "webrtcvad", "torch", "transformers", "datasets", "huggingface_hub")
    strict_ok = core_ok and all(report["packages"][package]["status"] == "installed" for package in optional)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python: {report['python']}\nOS: {report['os']}\nKaggle: {report['kaggle']}\nProject import: {report['project_import']['status']}")
        for package, details in report["packages"].items(): print(f"{'REQUIRED' if package in {'numpy', 'yaml', 'pytest'} else 'OPTIONAL'} {package}: {details['status']}{' (' + details['version'] + ')' if 'version' in details else ''}")
        print(f"GPU: {report['gpu']}")
    return 0 if (strict_ok if args.strict else core_ok) else 1

if __name__ == "__main__":
    raise SystemExit(main())
