"""Kaggle-only 1C-B full-model acceptance runner. No training or downloads."""
import argparse
import os
import resource
import random
import signal
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path

from support import BASE_PINS, MINI_PINS, PINS, CANONICAL, ROOT, digest, dump, read, kaggle_only


def verify(run):
    from dependencies import audit, require
    kaggle_only()
    if sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError('Use reviewed Python 3.12.10')
    report = audit()
    require(report)
    icefall = Path(os.environ['ICEFALL'])
    revision = subprocess.check_output(['git', '-C', str(icefall), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != BASE_PINS['icefall_revision']:
        raise RuntimeError('Icefall revision changed')
    changed = set(subprocess.check_output(
        ['git', '-C', str(icefall), 'diff', 'HEAD', '--name-only'], text=True).splitlines())
    if changed != {'egs/libricss/SURT/dprnn_zipformer/' + n for n in ('asr_datamodule.py', 'pretrained.py')}:
        raise RuntimeError(f'Unexpected upstream edits: {changed}')
    for name, expected in BASE_PINS['patched_source_sha256'].items():
        if digest(icefall / 'egs/libricss/SURT/dprnn_zipformer' / name) != expected:
            raise RuntimeError(f'Inherited patch changed: {name}')
    for name, expected in MINI_PINS['source_sha256'].items():
        if digest(icefall / name) != expected:
            raise RuntimeError(f'Recipe source changed: {name}')
    # Includes the resolved symlink targets, not just recipe pathnames.
    evidence = read(ROOT / 'configs/surt_stream/native-source.json')
    for name, expected in evidence['source_hashes'].items():
        if digest(icefall / name) != expected:
            raise RuntimeError(f'1C-A source drift: {name}')
    before = read(run / 'restored-inputs.json')['files']
    for name, expected in before.items():
        if digest(CANONICAL / name) != expected:
            raise RuntimeError(f'1B input drift: {name}')
    return report, before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    kaggle_only()
    run = args.run
    # Exclusive create happens before model/test work; never retry a started attempt.
    dump(run / 'started.json', dict(stage='1C-B', argv=sys.argv, wall_budget_seconds=900))
    def deadline(signum, frame):
        raise TimeoutError('900s budget exceeded')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        dependencies, before = verify(run)
        import numpy as np
        import torch
        import soundfile as sf
        import sentencepiece as spm
        import train
        from lhotse.features.kaldifeat import KaldifeatFbank, KaldifeatFbankConfig
        from run import make_model, audit_model
        from native import NativeSURT
        from native_checks import drive, ids_from_events

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA GPU required')
        torch.manual_seed(PINS['seed'])
        torch.cuda.manual_seed_all(PINS['seed'])
        np.random.seed(PINS['seed'])
        random.seed(PINS['seed'])
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        # Reduce partition-dependent arithmetic variation without changing weights.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, params = make_model(torch, train)
        audit_model(model, run, native_adapter='1C-B adapters implemented; acceptance evidence must be reviewed')
        extractor = KaldifeatFbank(KaldifeatFbankConfig(device='cpu'))
        previous = read(CANONICAL / 'feature-contract.json')['config']
        if {k: v for k, v in extractor.config.to_dict().items() if k != 'device'} != {
                k: v for k, v in previous.items() if k != 'device'}:
            raise RuntimeError('Frontend contract drift')
        row = next(r for r in read(CANONICAL / 'mixtures.json') if r['id'] == 'train-0')
        wave, rate = sf.read(row['audio'], dtype='float32')
        if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
            raise RuntimeError('Expected finite reviewed mono train mixture')
        tokenizer = spm.SentencePieceProcessor(model_file=str(CANONICAL / 'tokenizer/bpe.model'))
        if tokenizer.get_piece_size() != params.vocab_size or tokenizer.unk_id() != params.unk_id:
            raise RuntimeError('Tokenizer drift')
        dump(run / 'runtime.json', dict(dependencies=dependencies, python=sys.version,
             gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, precision='FP32',
             tf32=False, source_contract=read(ROOT / 'configs/surt_stream/native-source.json'),
             model_arguments=read(CANONICAL / 'model-arguments.json'),
             checkpoint_sha256=PINS['checkpoint_sha256'], tokenizer_sha256=PINS['tokenizer_sha256'],
             new_downloads=[], cuda_synchronization=True))
        # Execute the actual GPU model unittest suite in this process; reuse one load.
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_surt_stream_native import NativeAcceptance
        NativeAcceptance.fixture = (model, extractor, wave)
        NativeAcceptance.reports = []
        with torch.inference_mode():
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeAcceptance)
            with (run / 'logs/native-tests.log').open('x') as log:
                result = unittest.TextTestRunner(stream=log, verbosity=2, failfast=True).run(suite)
            dump(run / 'acceptance-tests.json', dict(run=result.testsRun, failures=len(result.failures),
                 errors=len(result.errors), skipped=len(result.skipped), cases=NativeAcceptance.reports))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError('Native acceptance tests failed; see logs/native-tests.log')
            stream = NativeSURT(model, extractor)
            for _ in range(2):
                stream.reset()
                drive(stream, wave, [1600], retain=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            measurements = []
            for _ in range(5):
                stream.reset()
                start = time.perf_counter()
                observed = drive(stream, wave, [1600])
                torch.cuda.synchronize()
                measurements.append(dict(push_compute_seconds=observed['seconds'],
                                         wall_seconds=time.perf_counter()-start))
            transcript = ids_from_events(observed['events'])
            pre_eof = [e for t in observed['timeline'] if t['before_eof'] for e in t['tokens']]
            acoustic_pre_eof = any(t['before_eof'] and t['output_frames'] for t in observed['timeline'])
            streaming_evidence = acoustic_pre_eof and bool(pre_eof)
            dump(run / 'streaming-report.json', dict(
                status='measured', warmup=2, measured=5, measurements=measurements,
                audio_seconds=len(wave)/16000,
                median_compute_rtf=statistics.median(m['push_compute_seconds'] for m in measurements)/(len(wave)/16000),
                median_wall_rtf=statistics.median(m['wall_seconds'] for m in measurements)/(len(wave)/16000),
                hypotheses=[tokenizer.decode(ids) for ids in transcript], branch_token_ids=transcript,
                references=row['references'], raw_mixture_metadata=row,
                branch_order='channel-major 0,1; not persistent speaker identities',
                timeline=observed['timeline'], eof_frames=observed['eof_frames'],
                eof_compute_seconds=observed['eof_compute_seconds'], eof_events=observed['eof_events'],
                acoustic_state_layout=dict(
                    mask_columns=[[list(t.shape) for t in pair] for pair in stream.mask.columns],
                    zipformer_by_branch=[[dict(shape=list(t.shape), dtype=str(t.dtype)) for t in states]
                                         for states in stream.recognition.zip_states],
                    joint=[list(t.shape) for t in stream.recognition.joint],
                    decoder_context=stream.recognition.context),
                emitted_acoustic_before_file_end=acoustic_pre_eof,
                emitted_token_before_file_end=bool(pre_eof),
                peak_acoustic_state_bytes=observed['peak_acoustic_state_bytes'],
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                latency=dict(packet_ms=100, zipformer_stride_features=32,
                    zipformer_input_features=39, mask_segment_features=32, mask_stride_features=16,
                    frontend_window_ms=25, first_acoustic_required_samples=10360,
                    nominal_first_acoustic_context_ms=647.5, zipformer_left_context_subsampled=64,
                    endpoint_ms=None, actual_live_delivery_ms=None,
                    protocol='unpaced packet replay; timeline uses max(previous_ready, received_time)+push_compute; not measured live latency'),
                quality_claim=False))
        for name, expected in before.items():
            if digest(CANONICAL / name) != expected:
                raise RuntimeError(f'1B input modified: {name}')
        dump(run / 'result.json', dict(
            status='native_streaming_checks_completed',
            gate_1c='REVIEW_REQUIRED' if streaming_evidence else 'OPEN',
            reason='Review numerical/context, EOF, stress and timing evidence' if streaming_evidence else
                   'No nonblank token before file end; do not claim complete demonstrated streaming',
            actual_gpu_tests_passed=True, prior_inputs_unchanged=True,
            quality_claim=False, full_training_claim=False))
        print('1C-B evidence written; gate requires review of result.json and streaming-report.json')
    except Exception as error:
        dump(run / 'failure.json', dict(exception=type(error).__name__, message=str(error), gate_1c='OPEN'))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
