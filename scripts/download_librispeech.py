#!/usr/bin/env python3
"""Download verified OpenSLR archives for an explicit pilot or standard run."""
import argparse
import hashlib
from pathlib import Path
import tarfile
from urllib.request import urlopen, urlretrieve

BASE = "https://openslr.trmal.net/resources/12/"


def extract_audio_archive(archive, destination):
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        handle.extractall(destination, members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["pilot", "standard"], default="pilot")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE + "md5sum.txt", timeout=60) as response:
        checksums = {Path(line.split()[1].lstrip("*")).name: line.split()[0]
                     for line in response.read().decode().splitlines() if line.strip()}
    subsets = ["dev-clean"] if args.layout == "pilot" else ["train-clean-100", "dev-clean", "test-clean"]
    for subset in subsets:
        marker = output / f".{subset}.verified"
        if marker.is_file() and (output / "LibriSpeech" / subset).is_dir():
            print(f"Reusing verified {subset}")
            continue
        name = f"{subset}.tar.gz"
        archive = output / name
        if not archive.is_file():
            partial = output / (name + ".partial")
            print(f"Downloading {BASE + name}", flush=True)
            urlretrieve(BASE + name, partial)
            partial.replace(archive)
        checksum = hashlib.md5()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != checksums[name]:
            raise ValueError(f"Checksum mismatch: remove {archive} and retry")
        extract_audio_archive(archive, output)
        marker.write_text(checksum.hexdigest(), encoding="utf-8")
        archive.unlink()
    print(output / "LibriSpeech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
