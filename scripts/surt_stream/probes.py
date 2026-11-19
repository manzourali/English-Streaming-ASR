"""Kaggle-only component evidence, explicitly separate from full-model streaming."""
import time

from frontend import BufferedFbank
from support import CANONICAL, PINS, dump, read


def frontend_probe(run):
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet
    from lhotse.features.kaldifeat import KaldifeatFbank, KaldifeatFbankConfig

    extractor = KaldifeatFbank(KaldifeatFbankConfig(device='cpu'))
    config = extractor.config.to_dict()
    prior_config = read(CANONICAL / 'feature-contract.json')['config']
    if {k: v for k, v in config.items() if k != 'device'} != {k: v for k, v in prior_config.items() if k != 'device'}:
        raise RuntimeError('Feature config changed relative to 1B')
    rows, comparisons = [], []
    for row in read(CANONICAL / 'selection.json') + read(CANONICAL / 'mixtures.json'):
        if row['split'] == 'train':
            rows.append(row)
    packets = ([1600], [173, 997, 3200, 1, 511])
    wave_for_stress = None

    def compare(label, wave, reference):
        for partition in packets:
            stream = BufferedFbank(extractor, PINS['max_packet_samples'])
            outputs, cursor, index = [], 0, 0
            begin = time.perf_counter()
            while cursor < len(wave):
                size = partition[index % len(partition)]
                outputs.append(stream.push(wave[cursor:cursor + size]))
                cursor += size
                index += 1
            outputs.append(stream.push([], final=True))
            actual = np.concatenate(outputs, axis=0)
            if actual.shape != reference.shape or not np.allclose(
                    actual, reference, atol=PINS['feature_atol'], rtol=PINS['feature_rtol']):
                raise RuntimeError(f'Incremental feature parity failed: {label}, packets={partition}')
            error = float(np.max(np.abs(actual - reference))) if actual.size else 0.0
            if len(stream.push([], final=True)):
                raise RuntimeError('EOF emitted frames twice')
            stream.reset()
            if stream.emitted or stream.total or len(stream.buffer) or stream.finished:
                raise RuntimeError('Feature reset retained recording state')
            # Reuse the reset instance, not just a fresh object.
            reset_outputs = []
            for start in range(0, len(wave), PINS['max_packet_samples']):
                reset_outputs.append(stream.push(wave[start:start + PINS['max_packet_samples']]))
            reset_outputs.append(stream.push([], final=True))
            reset_actual = np.concatenate(reset_outputs, axis=0)
            if reset_actual.shape != reference.shape or not np.allclose(
                    reset_actual, reference, atol=PINS['feature_atol'], rtol=PINS['feature_rtol']):
                raise RuntimeError('Feature reset/replay parity failed')
            comparisons.append(dict(input=label, samples=len(wave), frames=len(actual),
                                    packets=partition, max_absolute_error=error,
                                    peak_buffer_samples=stream.peak_samples,
                                    seconds_including_reset_replay=time.perf_counter() - begin))

    for row in rows:
        wave, rate = sf.read(row['audio'], dtype='float32')
        if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
            raise RuntimeError('Unexpected mini waveform')
        compare(row['audio'].split('/')[-1], wave, extractor.extract(wave, 16000))
        if row.get('id') == 'train-0':
            wave_for_stress = wave
    if wave_for_stress is None:
        raise RuntimeError('Missing real train mixture')
    # Boundary cases are prefixes of the approved train audio, not new ASR evidence.
    for length in (0, 1, 79, 80, 159, 160, 399, 400, 5119, 5120, 5361):
        wave = wave_for_stress[:length]
        reference = (extractor.extract(wave, 16000) if (len(wave) + 80) // 160
                     else np.empty((0, 80), dtype=np.float32))
        compare(f'train-prefix-{length}', wave, reference)
    stored = []
    for cut in CutSet.from_file(CANONICAL / 'features/train.jsonl.gz'):
        row = next(row for row in rows if row.get('id') == cut.id)
        wave, _ = sf.read(row['audio'], dtype='float32')
        reference, actual = cut.load_features(), extractor.extract(wave, 16000)
        if actual.shape != reference.shape or not np.allclose(
                actual, reference, atol=PINS['feature_atol'], rtol=PINS['feature_rtol']):
            raise RuntimeError(f'CPU frontend differs from stored 1B features: {cut.id}')
        stored.append(dict(cut_id=cut.id, max_absolute_error=float(np.max(np.abs(actual - reference)))))
    stream = BufferedFbank(extractor, PINS['max_packet_samples'])
    target, emitted = PINS['stress_seconds'] * 16000, 0
    begin = time.perf_counter()
    for start in range(0, target, PINS['packet_samples']):
        indices = np.arange(start, min(start + PINS['packet_samples'], target)) % len(wave_for_stress)
        emitted += len(stream.push(wave_for_stress[indices]))
    emitted += len(stream.push([], final=True))
    if emitted != (target + 80) // 160:
        raise RuntimeError('Feature stress test lost or duplicated frames')
    stress_compute_seconds = time.perf_counter() - begin
    report = dict(status='component_checks_completed', config=config, comparisons=comparisons,
                  stored_1b_comparisons=stored,
                  stress=dict(seconds_of_repeated_train_audio=PINS['stress_seconds'],
                              emitted_frames=emitted, peak_buffer_samples=stream.peak_samples,
                              compute_seconds=stress_compute_seconds,
                              compute_rtf=stress_compute_seconds / PINS['stress_seconds'],
                              purpose='feature-state stress only; not long-session recognition'),
                  latency=dict(declared_packet_duration_ms=100, frame_center_right_context_ms=12.5,
                               first_frame_required_samples=280,
                               model_right_context_ms=None, endpoint_ms=None,
                               end_to_end_streaming_ms=None),
                  scope='Bounded incremental features only; native model state remains unimplemented')
    dump(run / 'frontend-report.json', report)
    return report


def lstm_probe(torch, joint, encoded, run):
    """Observe the actual wrapper, then probe only its existing recurrent core.

    This deliberately does not claim a complete joint-wrapper export, since
    surrounding reshapes/projections and padded tails need a separate adapter.
    """
    cores = [(name, module) for name, module in joint.named_modules()
             if isinstance(module, torch.nn.LSTM)] if joint is not None else []
    if len(cores) != 1:
        report = dict(status='blocked', reason='Expected exactly one inspectable joint LSTM',
                      discovered=[name for name, module in cores])
        dump(run / 'joint-lstm-report.json', report)
        return report
    name, core = cores[0]
    observed = {}

    def capture(module, args, output):
        if isinstance(args[0], torch.Tensor) and isinstance(output[0], torch.Tensor):
            observed['input'] = args[0].detach().clone()
            observed['output'] = output[0].detach().clone()

    handle = core.register_forward_hook(capture)
    with torch.inference_mode():
        joint(encoded)
    handle.remove()
    if not observed or core.bidirectional:
        report = dict(status='blocked', reason='No tensor observation or bidirectional joint core')
        dump(run / 'joint-lstm-report.json', report)
        return report
    x = observed['input']
    axis = 1 if core.batch_first else 0
    outputs, state, maximum = [], None, 0
    with torch.inference_mode():
        for piece in x.split(7, dim=axis):
            output, state = core(piece, state)
            outputs.append(output)
            maximum = max(maximum, sum(value.numel() * value.element_size() for value in state))
        actual = torch.cat(outputs, dim=axis)
        reset, _ = core(x, None)
    if not torch.allclose(actual, observed['output'], atol=2e-4, rtol=1e-4):
        raise RuntimeError('Joint LSTM core chunk parity failed')
    if not torch.allclose(reset, observed['output'], atol=2e-4, rtol=1e-4):
        raise RuntimeError('Joint LSTM core reset parity failed')
    report = dict(status='recurrent_core_parity_completed', core=name, input_shape=list(x.shape),
                  time_axis=axis, state_bytes=maximum,
                  max_absolute_error=(actual - observed['output']).abs().max().item(),
                  scope='Only existing recurrent core; not wrapper/branch identity/end-to-end streaming proof')
    dump(run / 'joint-lstm-report.json', report)
    return report
