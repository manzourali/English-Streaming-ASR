"""Stage 1C component audit and bounded Kaggle resource probe.

Native end-to-end streaming is intentionally a separate, still-open gate.
Never route these weights through the incompatible generic streaming frontend.
"""
import argparse
import inspect
import math
import os
import random
import signal
import statistics
import subprocess
import time
from pathlib import Path

from probes import frontend_probe, lstm_probe
from support import BASE_PINS, CANONICAL, MINI_PINS, PINS, digest, dump, kaggle_only, read


def timed(torch, action):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return result, time.perf_counter() - begin


def audit_model(model, run):
    source_dir = run / 'upstream-source'
    source_dir.mkdir(exist_ok=False)
    modules, files = {}, {}
    for role, module in [('mask', model.mask_encoder), ('encoder', model.encoder),
                         ('branch_tying', model.joint_encoder_layer),
                         ('decoder', model.decoder), ('joiner', model.joiner)]:
        if module is None:
            modules[role] = None
            continue
        signatures = {}
        for method in ('forward', 'forward_chunk', 'streaming_forward', 'initial_state',
                       'get_init_state', 'get_init_states', 'get_init_state_tensor'):
            target = getattr(module, method, None)
            if callable(target):
                signatures[method] = str(inspect.signature(target))
        recurrent = []
        for name, child in module.named_modules():
            path = Path(inspect.getfile(type(child))).resolve()
            # Preserve recipe/icefall source, never installed Torch's large tree.
            icefall = Path(os.environ['ICEFALL']).resolve()
            if path.is_relative_to(icefall) and path.suffix == '.py':
                key = str(path.relative_to(icefall))
                if key not in files:
                    destination = source_dir / key
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(path.read_bytes())
                    files[key] = digest(path)
            if hasattr(child, 'bidirectional'):
                recurrent.append(dict(name=name, class_name=type(child).__name__,
                                      bidirectional=bool(child.bidirectional),
                                      batch_first=getattr(child, 'batch_first', None)))
        modules[role] = dict(class_name=type(module).__name__,
                             source=inspect.getfile(type(module)), methods=signatures,
                             recurrent=recurrent)
    recipe = Path(os.environ['ICEFALL']) / 'egs/libricss/SURT/dprnn_zipformer'
    for name in ('train.py', 'model.py', 'beam_search.py', 'dprnn.py',
                 'zipformer.py', 'subsampling.py', 'scaling.py'):
        path = (recipe / name).resolve()
        if path.is_file():
            key = str(path.relative_to(Path(os.environ['ICEFALL']).resolve()))
            if key not in files:
                destination = source_dir / key
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
                files[key] = digest(path)
    # Method presence is an inventory, not permission to guess its state contract.
    report = dict(icefall_revision=BASE_PINS['icefall_revision'], modules=modules,
                  source_hashes=files, model_arguments=read(CANONICAL / 'model-arguments.json'),
                  native_adapter='not implemented; exact API and context review required',
                  blockers=[
                      'DPRNN segment overlap/merge, bidirectional intra-segment context and inter-RNN state',
                      'Zipformer subsampling, attention/convolution caches and output frame index mapping',
                      'Branch-tying wrapper temporal reshape, hidden/cell state and padded-tail exclusion',
                      'Persistent greedy decoder context and one-symbol-per-frame parity',
                      'Composite reset/EOF, packet partition parity and bounded full acoustic state',
                  ],
                  engineering_order=['Review this exact source/API packet',
                                     'Implement and verify mask/encoder/joint state adapters individually',
                                     'Compose with validated frontend and persistent decoder',
                                     'Measure short/long end-to-end reset/EOF/context/latency parity'],
                  engineering_effort='Not quantified before source/API review; no schedule promise')
    dump(run / 'api-audit.json', report)
    return report


def make_model(torch, train):
    args = read(CANONICAL / 'model-arguments.json')
    params = train.get_params()
    params.update(vars(train.get_parser().parse_args([])))
    params.update(args['architecture'])
    for name in ('context_size', 'vocab_size', 'blank_id', 'unk_id', 'model_warm_step',
                 'prune_range', 'am_scale', 'lm_scale'):
        params[name] = args[name]
    params.ctc_loss_scale = args['loss_scales']['ctc']
    params.heat_loss_scale = args['loss_scales']['heat']
    params.batch_idx_train = 0
    checkpoint = torch.load(CANONICAL / 'mini-one-update.pt', map_location='cpu')
    if checkpoint['step'] != 1 or checkpoint['tokenizer_sha256'] != PINS['tokenizer_sha256']:
        raise RuntimeError('1B checkpoint metadata mismatch')
    model = train.get_surt_model(params)
    model.load_state_dict(checkpoint['model'], strict=True)
    if any(value.is_floating_point() and not torch.isfinite(value).all().item()
           for value in model.state_dict().values()):
        raise RuntimeError('Nonfinite restored model')
    del checkpoint
    model.to('cuda').eval()
    train.set_batch_count(model, 0)
    return model, params


def resources(torch, train, model, params, run, gpu_hours):
    import numpy as np
    import sentencepiece as spm
    from lhotse import CutSet
    from lhotse.dataset import K2SurtDataset
    from beam_search import greedy_search_batch

    tokenizer = spm.SentencePieceProcessor(model_file=str(CANONICAL / 'tokenizer/bpe.model'))
    if (tokenizer.get_piece_size() != params.vocab_size or tokenizer.unk_id() != params.unk_id
            or params.blank_id != 0):
        raise RuntimeError('Restored tokenizer/model mismatch')
    # Training data only: stage 1C adds no new validation/test tuning exposure.
    cuts = list(CutSet.from_file(CANONICAL / 'features/train.jsonl.gz'))
    if len(cuts) != 2:
        raise RuntimeError('Expected the two approved mini training mixtures')
    dataset = K2SurtDataset(return_sources=True, return_cuts=True, strict=True, num_channels=2)
    batch = dataset[CutSet.from_cuts([cuts[0]])]
    if batch['inputs'].shape[0] != 1 or batch['inputs'].shape[1] > 1600:
        raise RuntimeError('Resource probe exceeds its single short-mixture budget')
    feature = batch['inputs'].to('cuda')
    lengths = batch['input_lens'].to('cuda')
    encoded_for_joint = None

    def decode():
        nonlocal encoded_for_joint
        with torch.inference_mode():
            size, frames, bins = feature.shape
            masks = model.mask_encoder(feature).view(size, frames, bins, 2).unbind(-1)
            separated = torch.cat([feature * mask for mask in masks], dim=0)
            encoded, encoded_lengths = model.encoder(x=separated, x_lens=lengths.repeat(2))
            encoded_for_joint = encoded.detach()
            if model.joint_encoder_layer is not None:
                encoded = model.joint_encoder_layer(encoded)
            if not all(torch.isfinite(value).all().item() for value in (*masks, encoded)):
                raise RuntimeError('Nonfinite mask/encoder inference output')
            ids = greedy_search_batch(model=model, encoder_out=encoded, encoder_out_lens=encoded_lengths)
            if len(ids) != 2:
                raise RuntimeError('Expected exactly two decoded branches')
            return [tokenizer.decode(tokens) for tokens in ids]

    model.eval()
    for unused in range(PINS['warmup_passes']):
        decode()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    decode_seconds = []
    for unused in range(PINS['measured_passes']):
        hypotheses, seconds = timed(torch, decode)
        decode_seconds.append(seconds)
    inference = dict(protocol='whole-utterance recipe greedy; NOT incremental streaming',
                     warmup=PINS['warmup_passes'], seconds=decode_seconds,
                     median_seconds=statistics.median(decode_seconds),
                     utterance_seconds=cuts[0].duration,
                     median_rtf=statistics.median(decode_seconds) / cuts[0].duration,
                     references=batch['text'][0], hypotheses=hypotheses,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(run / 'inference-report.json', inference)
    joint = lstm_probe(torch, model.joint_encoder_layer, encoded_for_joint, run)
    del feature, lengths, encoded_for_joint
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=MINI_PINS['optimizer']['lr'])
    params.exp_dir = run
    steps = []
    groups = ('mask_encoder.', 'encoder.', 'joint_encoder_layer.', 'decoder.', 'joiner.',
              'ctc_output.', 'simple_am_proj.', 'simple_lm_proj.')
    raw_finite = {}

    def inspect_raw(module, inputs, output):
        for name, value in zip(('simple_rnnt', 'pruned_rnnt', 'ctc'), output[:3]):
            raw_finite[name] = bool(torch.isfinite(value).all().item())
        if not all(raw_finite.values()):
            raise RuntimeError('Nonfinite raw recipe loss; no silent filtering')

    hook = model.register_forward_hook(inspect_raw)
    if PINS['profiling_optimizer_steps'] != PINS['warmup_passes'] + PINS['measured_passes']:
        raise RuntimeError('Profiling step budget inconsistent')
    torch.cuda.reset_peak_memory_stats()
    for step in range(PINS['profiling_optimizer_steps']):
        params.batch_idx_train = step
        train.set_batch_count(model, step)

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss, info = train.compute_loss(params, model, tokenizer, batch, is_training=True)
            values = {key: float(info[key]) for key in ('loss', 'simple_loss', 'pruned_loss', 'ctc_loss', 'heat_loss')}
            if not torch.isfinite(loss).item() or not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError('Nonfinite profiling loss')
            loss.backward()
            active = {prefix: False for prefix in groups}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError(f'Nonfinite profiling gradient: {name}')
                for prefix in groups:
                    if name.startswith(prefix) and parameter.grad.detach().norm().item() > 0:
                        active[prefix] = True
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            if norm.item() <= 0:
                raise RuntimeError('Zero profiling gradient')
            optimizer.step()
            return dict(losses=values, gradient_norm=norm.item(), active_gradient_groups=active,
                        raw_finite=dict(raw_finite))

        result, seconds = timed(torch, update)
        steps.append(dict(profiling_step=step + 1, batch_idx_train=step,
                          measured=step >= PINS['warmup_passes'],
                          seconds=seconds, **result))
    hook.remove()
    measured = [row['seconds'] for row in steps if row['measured']]
    training = dict(protocol='Disposable in-memory FP32 profiling copy, fresh Adam; NOT resumed adaptation',
                    precision='FP32; no autocast/scaler', learning_rate=MINI_PINS['optimizer']['lr'],
                    initialized_from_1b_step=1, optimizer_steps=len(steps), steps=steps,
                    median_seconds=statistics.median(measured),
                    p90_seconds=float(np.quantile(measured, 0.9)),
                    mixed_audio_seconds_per_batch=cuts[0].duration,
                    includes='forward/backward/gradient checks/clip/Adam/CUDA synchronization',
                    excludes='data loading, feature preparation, checkpoints, validation and scheduler scaling',
                    source_audio_seconds_per_batch=sum(
                        row['duration'] for row in read(CANONICAL / 'selection.json')
                        if row['source_id'] in next(m['sources'] for m in read(CANONICAL / 'mixtures.json')
                                                   if m['id'] == cuts[0].id)),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    persistent_checkpoint_written=False)
    dump(run / 'training-resource-report.json', training)
    estimates = []
    for hours in (10, 50, 100):
        for epochs in (1, 5, 10):
            count = math.ceil(hours * 3600 / cuts[0].duration) * epochs
            estimate = count * training['median_seconds'] / 3600
            estimates.append(dict(mixed_audio_hours=hours, epochs=epochs, steps=count,
                                  idealized_gpu_hours=estimate,
                                  idealized_p90_gpu_hours=count * training['p90_seconds'] / 3600,
                                  sessions_at_user_gpu_hours=math.ceil(estimate / gpu_hours) if gpu_hours else None))
    dump(run / 'resource-estimates.json', dict(
        status='small-hot-cache linear arithmetic only; not a full-training forecast',
        assumptions='Same short mixture duration, batch=1, FP32, fresh Adam and GPU; excludes I/O/evaluation/save overhead',
        available_gpu_hours_per_session=gpu_hours or None, scenarios=estimates,
        limitations='Longer speech, token lengths, FastMSS overlap, actual hours, batching and quota may change cost substantially'))
    return dict(inference=inference, joint=joint, training=training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--available-gpu-hours', type=float, default=0)
    args = parser.parse_args()
    kaggle_only()
    if not math.isfinite(args.available_gpu_hours) or args.available_gpu_hours < 0:
        parser.error('--available-gpu-hours must be finite and nonnegative; zero means unknown')
    run = args.run
    if not (run / 'restored-inputs.json').is_file():
        raise RuntimeError('Run restore/verification cell first')
    dump(run / 'started.json', dict(argv=os.sys.argv, limits=PINS,
                                   note='A repeated run is forbidden; archive and choose a fresh ID'))
    def deadline(signum, frame):
        raise TimeoutError('Stage 1C exceeded the 900-second experiment budget')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(PINS['wall_seconds'])
    try:
        import numpy as np
        import torch
        import train
        from dependencies import audit, require

        if os.sys.version_info[:3] != (3, 12, 10) or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Use isolated Kaggle Python 3.12.10 and exactly one GPU')
        report = audit()
        require(report)
        icefall = Path(os.environ['ICEFALL'])
        revision = subprocess.check_output(['git', '-C', str(icefall), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != BASE_PINS['icefall_revision']:
            raise RuntimeError('Icefall revision changed')
        changed = set(subprocess.check_output(
            ['git', '-C', str(icefall), 'diff', 'HEAD', '--name-only'], text=True).splitlines())
        if changed != {'egs/libricss/SURT/dprnn_zipformer/' + name
                       for name in ('asr_datamodule.py', 'pretrained.py')}:
            raise RuntimeError(f'Unexpected Icefall modifications: {sorted(changed)}')
        for name, expected in BASE_PINS['patched_source_sha256'].items():
            if digest(icefall / 'egs/libricss/SURT/dprnn_zipformer' / name) != expected:
                raise RuntimeError(f'Inherited patch changed: {name}')
        for name, expected in MINI_PINS['source_sha256'].items():
            if digest(icefall / name) != expected:
                raise RuntimeError(f'Pinned source mismatch: {name}')
        before = read(run / 'restored-inputs.json')['files']
        for relative, expected in before.items():
            if digest(CANONICAL / relative) != expected:
                raise RuntimeError(f'Restored input changed after verification: {relative}')
        dump(run / 'runtime.json', dict(dependencies=report, python=os.sys.version,
             gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
             total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
             icefall_revision=revision, cuda_synchronization=True, new_downloads=[]))
        np.random.seed(PINS['seed'])
        random.seed(PINS['seed'])
        torch.manual_seed(PINS['seed'])
        torch.cuda.manual_seed_all(PINS['seed'])
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        (model, params), load_seconds = timed(torch, lambda: make_model(torch, train))
        dump(run / 'load-report.json', dict(seconds=load_seconds, strict=True,
             checkpoint_sha256=PINS['checkpoint_sha256'],
             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        audit_model(model, run)
        frontend_probe(run)
        resources(torch, train, model, params, run, args.available_gpu_hours)
        for relative, expected in before.items():
            if digest(CANONICAL / relative) != expected:
                raise RuntimeError(f'1B input modified during probe: {relative}')
        dump(run / 'result.json', dict(status='component_audit_and_resource_probe_completed',
             gate_1c='OPEN: native composite streaming has not been implemented or validated',
             accepted_1b=PINS['accepted_mini_run'], prior_inputs_unchanged=True,
             next_action='Return review archive with exact upstream source; implement audited native adapters',
             native_streaming=dict(partition_parity=None, bounded_acoustic_state=None,
                  reset_eof_parity=None, model_right_context_ms=None, acquisition_latency_ms=None,
                  context_latency_ms=None, endpoint_latency_ms=None, compute_latency_ms=None,
                  end_to_end_latency_ms=None, streaming_rtf=None),
             quality_claim=False, full_training_claim=False))
        print('Component/resource probe completed; native streaming gate remains OPEN', flush=True)
    except Exception as error:
        dump(run / 'failure.json', dict(exception=type(error).__name__, message=str(error),
             gate_1c='open', action='Preserve inputs/run, execute archive cell and return evidence'))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
