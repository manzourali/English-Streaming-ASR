#!/usr/bin/env python3
"""Check actual imports and record all visible GPUs without changing the stack."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGES = {
    "numpy": "numpy", "yaml": "PyYAML", "pytest": "pytest",
    "soundfile": "soundfile", "sentencepiece": "sentencepiece",
    "webrtcvad": "webrtcvad-wheels", "torch": "torch", "torchaudio": "torchaudio",
    "transformers": "transformers", "datasets": "datasets", "peft": "peft",
    "huggingface_hub": "huggingface_hub", "pandas": "pandas",
    "tensorflow": "tensorflow", "librosa": "librosa", "sklearn": "scikit-learn",
    "cv2": "opencv-python", "PIL": "Pillow", "k2": "k2", "kaldifeat": "kaldifeat",
    "lhotse": "lhotse",
}
CORE = ("numpy", "yaml", "pytest")
KAGGLE = CORE + ("soundfile", "sentencepiece", "huggingface_hub")


def _package(name: str, *, check_import: bool = False) -> dict[str, str]:
    try:
        version = importlib.metadata.version(PACKAGES[name])
    except importlib.metadata.PackageNotFoundError:
        if name == "cv2":
            try:
                version = importlib.metadata.version("opencv-python-headless")
            except importlib.metadata.PackageNotFoundError:
                return {"status": "not installed"}
        else:
            return {"status": "not installed"}
    result = {"status": "installed", "version": version, "import": "not checked"}
    if check_import:
        try:
            importlib.import_module(name)
            result["import"] = "OK"
        except Exception as exc:
            result.update(status="broken", error=f"{type(exc).__name__}: {exc}")
    return result


def _gpu_report() -> dict[str, object]:
    try:
        import torch
        available = torch.cuda.is_available()
        devices = []
        for index in range(torch.cuda.device_count() if available else 0):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name,
                            "memory_mib": props.total_memory // (1024 * 1024),
                            "compute_capability": list(torch.cuda.get_device_capability(index)),
                            "native_bfloat16": props.major >= 8})
        return {"torch": str(torch.__version__), "available": available,
                "cuda": torch.version.cuda, "count": len(devices), "devices": devices}
    except Exception as exc:
        return {"available": False, "count": 0, "devices": [],
                "error": f"{type(exc).__name__}: {exc}"}


def _writable(path: Path) -> bool:
    # Missing directories are usable when their nearest ancestor is writable.
    while not path.exists() and path != path.parent:
        path = path.parent
    return path.is_dir() and os.access(path, os.W_OK)


def environment_report(*, required: tuple[str, ...] = CORE,
                       require_gpu: bool = False) -> tuple[dict[str, object], bool]:
    kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir())
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "os": platform.platform(), "machine": platform.machine(), "cwd": str(Path.cwd()),
        "kaggle": kaggle,
        "platform_signals": {"kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
                             "kaggle_working_exists": Path("/kaggle/working").is_dir(),
                             "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG")},
        "packages": {name: _package(name, check_import=name in required) for name in PACKAGES},
        "writable": {str(path): _writable(path) for path in (Path("outputs"), Path("checkpoints"))},
        "gpu": _gpu_report(), "warnings": [], "errors": [],
    }
    try:
        import streaming_asr
        report["project_import"] = {"status": "OK", "version": streaming_asr.__version__}
    except Exception as exc:
        report["project_import"] = {"status": "NOT IMPORTABLE", "error": str(exc)}
        report["errors"].append("Project import failed")
    for name in required:
        if report["packages"][name]["status"] != "installed":
            report["errors"].append(f"Required package unavailable or broken: {name}")
    if not all(report["writable"].values()):
        report["errors"].append("Output/checkpoint locations are not writable")
    if require_gpu and not report["gpu"]["available"]:
        report["errors"].append("CUDA GPU required but unavailable")
    torch_version = report["packages"]["torch"].get("version")
    audio_version = report["packages"]["torchaudio"].get("version")
    if torch_version and audio_version and torch_version != audio_version:
        report["warnings"].append("Torch/torchaudio versions differ; verify their binary compatibility")
    reference = json.loads((ROOT / "configs/kaggle_environment.json").read_text())
    report["reference_source"] = reference["source"]
    if kaggle:
        if report["python"] != reference["python"]:
            report["warnings"].append(f"Python differs from recorded snapshot: {reference['python']}")
        for module, distribution in PACKAGES.items():
            expected = reference["packages"].get(distribution)
            actual = report["packages"][module].get("version")
            if expected and actual != expected:
                report["warnings"].append(f"Snapshot drift: {distribution}: expected {expected}, got {actual}")
        if report["gpu"]["count"] != reference["gpu"]["count"]:
            report["warnings"].append("Visible GPU count differs from the two-GPU snapshot")
    return report, not report["errors"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="Save the JSON report for this run")
    parser.add_argument("--profile", choices=("core", "kaggle"), default="core")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Also import optional model/dataset packages")
    args = parser.parse_args()
    required = KAGGLE if args.profile == "kaggle" else CORE
    if args.strict:
        required = tuple(dict.fromkeys((*required, "soundfile", "webrtcvad", "torch",
                                       "torchaudio", "transformers", "datasets", "huggingface_hub")))
    report, ok = environment_report(required=required, require_gpu=args.require_gpu)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(f"Python: {report['python']}\nOS: {report['os']}\nKaggle: {report['kaggle']}\nProject import: {report['project_import']['status']}")
        for package, details in report["packages"].items():
            print(f"{'REQUIRED' if package in required else 'OPTIONAL'} {package}: {details}")
        print(f"GPU: {report['gpu']}")
        for message in report["warnings"]:
            print(f"WARNING: {message}")
        for message in report["errors"]:
            print(f"ERROR: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
