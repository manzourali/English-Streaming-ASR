import pytest

from streaming_asr.evaluation import Benchmark, EvaluationRecord, Evaluator, computational_table, main_results_table, read_evaluation_records, write_evaluation_records


def _record(system, sample_id, value, *, errors=None, units=None, regime="medium", oracle=False):
    return EvaluationRecord(system, sample_id, "test", "synthetic", "pi_wer", value, "lower", errors, units, regime, 0.5, 0.0, "partial", oracle, 2.0, 1.0, 0.2, 0.1, 0.05, 100.0, ("speaker_miss",) if value else ())


def test_weighted_summary_bootstrap_and_stratification():
    records = [_record("base", "a", 0.5, errors=5, units=10), _record("base", "b", 0.25, errors=5, units=20, regime="high")]
    evaluator = Evaluator(100, 7)
    summaries = evaluator.summarize(records)
    assert summaries[0].value == pytest.approx(10 / 30)
    assert summaries[0].confidence_interval_95 is not None
    strata = evaluator.stratify(records, "overlap_regime")
    assert set(strata) == {"high", "medium"}
    assert evaluator.error_taxonomy(records) == {"speaker_miss": 2}


def test_paired_improvement_requires_comparable_identical_samples():
    evaluator = Evaluator(10)
    result = evaluator.paired_improvement([_record("base", "a", 0.5), _record("base", "b", 0.4)], [_record("candidate", "a", 0.3), _record("candidate", "b", 0.2)])
    assert result["absolute_improvement"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        evaluator.paired_improvement([_record("base", "a", 0.5)], [_record("candidate", "b", 0.3)])


def test_record_roundtrip_benchmark_and_programmatic_tables(tmp_path):
    records = [_record("base", "a", 0.5), _record("candidate", "a", 0.25)]
    path = tmp_path / "records.jsonl"; write_evaluation_records(records, path)
    loaded = read_evaluation_records(path)
    summaries = Evaluator(10).summarize(loaded); benchmarks = Benchmark().run(loaded)
    assert benchmarks[0].rtf == pytest.approx(0.5)
    assert "candidate" in main_results_table(summaries, ["base", "candidate", "missing"])
    assert "NOT MEASURED" in computational_table(benchmarks, ["base", "missing"])
