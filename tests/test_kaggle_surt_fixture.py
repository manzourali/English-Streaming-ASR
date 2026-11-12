"""Lightweight contract tests; these do not build the Kaggle fixture."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_surt_fixture.py"
SPEC = importlib.util.spec_from_file_location("kaggle_surt_fixture", SCRIPT)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def test_recipe_labels_and_manifest_names():
    assert tuple(fixture.LABELS) == ("0L", "0S", "OV10", "OV20", "OV30", "OV40")
    assert len(fixture.MANIFEST_NAMES) == 5
    assert fixture.LABELS["OV40"] == 0.4


def test_empty_group_probe_records_recipe_failure():
    def fail(batches, params, model, tokenizer):
        assert batches == []
        assert params.decoding_method == "greedy_search"
        raise UnboundLocalError("masks_dict")

    assert "masks_dict" in fixture.probe_empty_decode_group(SimpleNamespace(decode_dataset=fail))


def test_empty_group_probe_does_not_hide_other_errors():
    def fail(*args):
        raise ValueError("unexpected failure")

    with pytest.raises(ValueError, match="unexpected failure"):
        fixture.probe_empty_decode_group(SimpleNamespace(decode_dataset=fail))
