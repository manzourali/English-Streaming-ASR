"""Independent finite-recording reference and Kaggle acceptance checks."""
import time
import numpy as np
import torch
from frontend import extract_with_kaldi_edges
from native import NativeSURT

ATOL, RTOL = 2e-4, 1e-4


def close(actual, expected, label):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f'{label}: nonfinite tensor')
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        error = (actual - expected).abs().max().item() if actual.shape == expected.shape and actual.numel() else None
        raise AssertionError(f'{label}: shapes={actual.shape}/{expected.shape}, max_error={error}')
    return (actual - expected).abs().max().item() if actual.numel() else 0.0


@torch.inference_mode()
def reference(model, extractor, wave):
    """Offline upstream DPRNN, fixed-context native Zipformer, whole joint/greedy.

    Deliberately does not call MaskStream/RecognitionStream. Zipformer offline
    forward has different convolution/pooling semantics; it is NOT this reference.
    """
    from beam_search import greedy_search_batch
    x = torch.as_tensor(extract_with_kaldi_edges(extractor, wave),
                        device=next(model.parameters()).device).unsqueeze(0)
    f = x.shape[1]
    dim = model.joint_encoder_layer[1].input_size // 2
    empty = x.new_empty(2, 0, dim)
    if f == 0:
        return empty, [[], []], x.new_empty(2, 0, 80), [[], []]
    masks = model.mask_encoder(x).view(1, f, 80, 2).unbind(-1)
    masked = torch.cat([x * m for m in masks], 0)
    valid = (max(0, (f - 7) // 2) + 1) // 2
    if not valid:
        return empty, [[], []], masked, [[], []]
    encoded = []
    for branch in range(2):
        states = model.encoder.get_init_state(x.device)
        pieces = []
        for start in range(0, ((valid + 7) // 8) * 32, 32):
            piece = masked[branch:branch+1, start:start+39]
            piece = torch.nn.functional.pad(piece, (0, 0, 0, 39 - piece.shape[1]), value=-23.025850929940457)
            y, lens, states = model.encoder.streaming_forward(
                piece, torch.tensor([39], device=x.device), states)
            assert y.shape[1] == lens.item() == 8
            pieces.append(y)
        encoded.append(torch.cat(pieces, 1)[:, :valid])
    tied = model.joint_encoder_layer(torch.cat(encoded, 0))
    decoded = greedy_search_batch(model, tied, torch.tensor([valid, valid], device=x.device),
                                  return_timestamps=True)
    return tied, decoded.hyps, masked, decoded.timestamps


def ids_from_events(events):
    return [[e['token'] for e in events if e['branch'] == branch] for branch in range(2)]


@torch.inference_mode()
def drive(stream, wave, packets, retain=True, final_on_last=False):
    outputs, events, timeline = [], [], []
    cursor = index = frames = 0
    compute = 0.0
    simulated_ready_ms = 0.0
    while cursor < len(wave):
        end = min(len(wave), cursor + packets[index % len(packets)])
        torch.cuda.synchronize()
        begin = time.perf_counter()
        y, emitted = stream.push(wave[cursor:end], final=final_on_last and end == len(wave))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        compute += elapsed
        queue_ms = max(0.0, simulated_ready_ms - end/16)
        simulated_ready_ms = max(simulated_ready_ms, end/16) + elapsed*1000
        if retain:
            outputs.append(y)
            events.extend(emitted)
        if y.shape[1] or emitted:
            # Earliest masked frames for output block k need raw frame
            # 32*k+38; its containing DPRNN segment ends at next stride boundary.
            block = frames // 8
            needed_features = ((32 * block + 38) // 16 + 1) * 16 + 16
            required_samples = (needed_features - 1) * 160 + 280
            timeline.append(dict(received_samples=end, before_eof=end < len(wave),
                                 output_start_frame=frames, output_frames=y.shape[1], tokens=emitted,
                                 compute_seconds=elapsed, cumulative_compute_seconds=compute,
                                 required_samples=required_samples,
                                 packet_excess_ms=max(0, end-required_samples) / 16,
                                 # Nominal output frame anchor (40ms grid), not a word alignment.
                                 anchor_ms=40 * frames,
                                 context_wait_ms=required_samples / 16 - 40 * frames,
                                 received_audio_ms=end / 16,
                                 simulated_queue_ms=queue_ms,
                                 simulated_ready_ms=simulated_ready_ms,
                                 simulated_delivery_from_anchor_ms=simulated_ready_ms - 40*frames))
        frames += y.shape[1]
        cursor = end
        index += 1
    torch.cuda.synchronize()
    begin = time.perf_counter()
    y, emitted = stream.push([], final=True)
    torch.cuda.synchronize()
    eof_compute = time.perf_counter() - begin
    if retain:
        outputs.append(y)
        events.extend(emitted)
    frames += y.shape[1]
    before = frames
    repeated, duplicate = stream.push([], final=True)
    assert repeated.shape[1] == 0 and not duplicate
    try:
        stream.push(wave[:1])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Post-EOF input accepted')
    empty = next(stream.model.parameters()).new_empty(2, 0, stream.model.joint_encoder_layer[1].input_size // 2)
    return dict(encoded=torch.cat(outputs, 1) if outputs else empty, events=events,
                frames=before, timeline=timeline, seconds=compute + eof_compute,
                eof_frames=y.shape[1], eof_events=emitted, eof_compute_seconds=eof_compute,
                peak_acoustic_state_bytes=stream.peak_state_bytes,
                final_state_bytes=stream.acoustic_bytes())


@torch.inference_mode()
def check_case(model, extractor, wave, label):
    from native import MaskStream
    expected, expected_ids, expected_masked, expected_timestamps = reference(model, extractor, wave)
    # Full upstream DPRNN is a separate reference for the stateful segment adapter.
    features = torch.as_tensor(extract_with_kaldi_edges(extractor, wave),
                               device=next(model.parameters()).device).unsqueeze(0)
    mask = MaskStream(model.mask_encoder)
    pieces = [mask.push(p) for p in features.split(13, 1) if p.shape[1]]
    pieces.append(mask.push(features[:, :0], final=True))
    mask_error = close(torch.cat(pieces, 1), expected_masked, label + ':mask')
    stream = NativeSURT(model, extractor)
    reports = []
    canonical_events = None
    for packets in ([1600], [173, 997, 3200, 1, 511]):
        stream.reset()  # reuse the same object, after a different recording/partition
        actual = drive(stream, wave, packets)
        error = close(actual['encoded'], expected, label + ':composite')
        if canonical_events is None:
            canonical_events = actual['events']
        assert actual['events'] == canonical_events, label + ': token frame/branch changed with partition'
        assert ids_from_events(actual['events']) == expected_ids, label + ': branch IDs differ'
        assert [[e['frame'] for e in actual['events'] if e['branch'] == b] for b in range(2)] == expected_timestamps, label + ': reference token frames differ'
        assert actual['frames'] == (max(0, (features.shape[1]-7)//2)+1)//2
        assert actual['peak_acoustic_state_bytes'] < 16 * 1024 * 1024
        reports.append({k: v for k, v in actual.items() if k != 'encoded'} |
                       dict(packets=packets, max_absolute_error=error, ids=expected_ids))
    # Abandon a live recording and reset: no stale states from incomplete EOF.
    stream.reset()
    stream.push(wave[:min(3200, len(wave))])
    stream.reset()
    replay = drive(stream, wave, [3200])
    close(replay['encoded'], expected, label + ':reset-live')
    assert ids_from_events(replay['events']) == expected_ids
    assert replay['events'] == canonical_events, label + ': token frame changed after reset'
    stream.reset()
    final_packet = drive(stream, wave, [173, 997, 3200, 1, 511], final_on_last=True)
    close(final_packet['encoded'], expected, label + ':final-on-last-packet')
    assert final_packet['events'] == canonical_events, label + ': EOF packet changed tokens'
    return dict(label=label, samples=len(wave), feature_frames=features.shape[1],
                output_frames=expected.shape[1], mask_max_absolute_error=mask_error,
                reports=reports)
