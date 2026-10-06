from streaming_asr.utils.provenance import reproducibility_manifest, sha256_file


def test_sha256_and_manifest_distinguish_present_and_missing_files(tmp_path):
    present = tmp_path / "present.txt"; present.write_text("streaming", encoding="utf-8")
    assert len(sha256_file(present)) == 64
    manifest = reproducibility_manifest(tmp_path, ["present.txt", "missing.txt"])
    assert len(manifest["sha256"]["present.txt"]) == 64
    assert manifest["sha256"]["missing.txt"] == "MISSING"
