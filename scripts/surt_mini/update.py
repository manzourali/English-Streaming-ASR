"""Exactly one genuine recipe update, strict reload and bounded two-stream decode."""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

from contracts import PINS, ROOT, dump, file_hash, score
from transfer import transfer


def timed(torch, action):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return result, time.perf_counter() - begin


def assert_finite_tree(torch, value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite checkpoint/optimizer tensor')
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(torch, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite_tree(torch, child)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('GPU execution is Kaggle-only')
    import numpy as np
    import sentencepiece as spm
    import torch
    from lhotse import CutSet
    from lhotse.dataset import K2SurtDataset
    import train
    from beam_search import greedy_search_batch

    run = args.run
    result_path = run / 'result.json'
    checkpoint_path = run / 'mini-one-update.pt'
    if result_path.exists() or checkpoint_path.exists() or (run / 'update-started.json').exists():
        raise RuntimeError('This run was already attempted; archive it and choose a new RUN_ID; keep all assets')
    if not (run / 'runtime.json').is_file():
        raise RuntimeError('Runtime verification required')
    dump(run / 'update-started.json', dict(argv=os.sys.argv, authorized_steps=1))
    torch.set_num_threads(1)
    random.seed(PINS['seed'])
    np.random.seed(PINS['seed'])
    torch.manual_seed(PINS['seed'])
    torch.cuda.manual_seed_all(PINS['seed'])
    torch.backends.cudnn.benchmark = False
    baseline = Path('/kaggle/working/surt-base-1a')
    source_model = baseline / 'assets/model'
    base_pins = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())
    for name, expected in base_pins['assets'].items():
        if file_hash(source_model / name) != expected:
            raise RuntimeError('Official assets changed before update')
    old_sp = spm.SentencePieceProcessor(model_file=str(source_model / 'data/lang_bpe_500/bpe.model'))
    new_sp = spm.SentencePieceProcessor(model_file=str(run / 'tokenizer/bpe.model'))
    baseline_args = json.loads((baseline / 'results/result.json').read_text())['model_arguments']
    params = train.get_params()
    params.update(vars(train.get_parser().parse_args([])))
    architecture = argparse.ArgumentParser()
    train.add_model_arguments(architecture)
    architecture_keys = vars(architecture.parse_args([]))
    for name in architecture_keys:
        params[name] = baseline_args[name]
    params.context_size = baseline_args['context_size']
    params.blank_id, params.unk_id, params.vocab_size = 0, new_sp.unk_id(), new_sp.get_piece_size()
    params.batch_idx_train = 0
    params.ctc_loss_scale = PINS['loss_scales_at_step_zero']['ctc']
    params.heat_loss_scale = PINS['loss_scales_at_step_zero']['heat']
    params.exp_dir = run
    dump(run / 'model-arguments.json', dict(architecture={name: params[name] for name in architecture_keys},
         context_size=params.context_size, vocab_size=params.vocab_size, blank_id=params.blank_id,
         unk_id=params.unk_id, batch_idx_train=0, model_warm_step=params.model_warm_step,
         prune_range=params.prune_range, am_scale=params.am_scale, lm_scale=params.lm_scale,
         loss_scales=PINS['loss_scales_at_step_zero']))

    def load():
        model = train.get_surt_model(params)
        state = torch.load(source_model / 'exp/surt_base/pretrained.pt', map_location='cpu')['model']
        transfer(model, state, old_sp, new_sp, run / 'transfer-report.json')
        return model.to('cuda')

    torch.cuda.reset_peak_memory_stats()
    model, load_seconds = timed(torch, load)
    load_memory = dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                       reserved_bytes=torch.cuda.max_memory_reserved())
    train.set_batch_count(model, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=PINS['optimizer']['lr'])
    assert not optimizer.state
    dataset = K2SurtDataset(return_sources=True, return_cuts=True, strict=True, num_channels=2)
    training_cuts = CutSet.from_file(run / 'features/train.jsonl.gz')
    batch = dataset[CutSet.from_cuts([next(iter(training_cuts))])]
    dump(run / 'update-input.json', dict(cut_ids=[cut.id for cut in batch['cuts']],
         text=batch['text'], shape=list(batch['inputs'].shape), lengths=batch['input_lens'].tolist(),
         source_boundaries=batch['source_boundaries']))
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    raw_losses = {}

    def inspect_losses(module, inputs, output):
        for name, value in zip(('simple_rnnt', 'pruned_rnnt', 'ctc'), output[:3]):
            if not torch.isfinite(value).all():
                raise RuntimeError(f'Nonfinite {name}; no filtering allowed')
            raw_losses[name] = value.detach().cpu().tolist()

    hook = model.register_forward_hook(inspect_losses)
    torch.cuda.reset_peak_memory_stats()
    model.train()

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss, info = train.compute_loss(params, model, new_sp, batch, is_training=True)
        if not torch.isfinite(loss) or not all(np.isfinite(float(info[name])) for name in ('simple_loss', 'pruned_loss', 'ctc_loss', 'heat_loss')):
            raise RuntimeError('Nonfinite component loss')
        expected = info['simple_loss'] + 0.1 * info['pruned_loss'] + params.ctc_loss_scale * info['ctc_loss'] + params.heat_loss_scale * info['heat_loss']
        if not np.isclose(loss.detach().cpu().item(), expected, rtol=1e-5):
            raise RuntimeError('Recipe loss scale contract changed')
        loss.backward()
        gradients = {}
        missing_gradients = []
        for name, value in model.named_parameters():
            if value.grad is None:
                missing_gradients.append(name)
                continue
            if not torch.isfinite(value.grad).all():
                raise RuntimeError(f'Nonfinite gradient {name}')
            gradients[name] = value.grad.detach().norm().item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        if norm.item() <= 0:
            raise RuntimeError('Zero gradient')
        optimizer.step()
        return dict(losses={name: float(info[name]) for name in ('loss', 'simple_loss', 'pruned_loss', 'ctc_loss', 'heat_loss')},
                    raw_losses=raw_losses, gradient_norm_before_clip=norm.item(), clip_norm=5.0,
                    gradients=gradients, missing_gradients=missing_gradients,
                    optimizer_steps=1, precision='FP32; no autocast/scaler',
                    scales=PINS['loss_scales_at_step_zero'])

    update_result, update_seconds = timed(torch, update)
    hook.remove()
    update_result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    changed = []
    for name, value in model.named_parameters():
        if not torch.isfinite(value).all():
            raise RuntimeError(f'Nonfinite updated parameter {name}')
        delta = (value.detach().cpu() - before[name]).norm().item()
        if delta > 0:
            changed.append(dict(name=name, delta_l2=delta))
    if not changed:
        raise RuntimeError('Optimizer changed no weights')
    del before
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f'Nonfinite checkpoint tensor {name}')
    dump(run / 'update-report.json', dict(**update_result, changed=changed, seconds=update_seconds,
                                         model_load_seconds=load_seconds))
    checkpoint = dict(model=state, optimizer=optimizer.state_dict(), step=1,
                      tokenizer_sha256=file_hash(run / 'tokenizer/bpe.model'),
                      dataset_revision=PINS['dataset_revision'])
    assert_finite_tree(torch, checkpoint)
    if not all(value['step'].item() == 1 for value in checkpoint['optimizer']['state'].values()):
        raise RuntimeError('Unexpected optimizer step count')
    torch.save(checkpoint, checkpoint_path)
    del checkpoint
    del model, optimizer, state
    gc.collect()
    saved = torch.load(checkpoint_path, map_location='cpu')
    assert_finite_tree(torch, saved)
    assert saved['step'] == 1 and saved['tokenizer_sha256'] == file_hash(run / 'tokenizer/bpe.model')
    model = train.get_surt_model(params)
    model.load_state_dict(saved['model'], strict=True)
    assert all(torch.equal(value, saved['model'][name]) for name, value in model.state_dict().items())
    model.to('cuda').eval()
    del saved
    torch.cuda.synchronize()
    decode_results = []
    for split in ('validation', 'test'):
        decode_batch = dataset[CutSet.from_file(run / f'features/{split}.jsonl.gz')]

        def decode():
            with torch.inference_mode():
                feature = decode_batch['inputs'].to('cuda')
                lengths = decode_batch['input_lens'].to('cuda')
                size, frames, bins = feature.shape
                masked = model.mask_encoder(feature).view(size, frames, bins, 2).unbind(-1)
                inputs = torch.cat([feature * mask for mask in masked], dim=0)
                encoded, encoded_lengths = model.encoder(x=inputs, x_lens=lengths.repeat(2))
                if model.joint_encoder_layer is not None:
                    encoded = model.joint_encoder_layer(encoded)
                ids = greedy_search_batch(model=model, encoder_out=encoded, encoder_out_lens=encoded_lengths)
                hypotheses = [new_sp.decode(tokens) for tokens in ids]
                return [dict(cut_id=cut.id, split=split, references=refs,
                             hypotheses=hypotheses[index::size],
                             score=score(refs, hypotheses[index::size]))
                        for index, (cut, refs) in enumerate(zip(decode_batch['cuts'], decode_batch['text']))]

        torch.cuda.reset_peak_memory_stats()
        decoded, seconds = timed(torch, decode)
        decode_results.append(dict(split=split, seconds=seconds, utterances=decoded,
                                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    dump(result_path, dict(status='technical_execution_completed; pending Kaggle evidence review',
         gate_1b='pending human review', optimizer_steps=1, strict_reload=True,
         checkpoint_sha256=file_hash(checkpoint_path), tokenizer_sha256=file_hash(run / 'tokenizer/bpe.model'),
         load_seconds=load_seconds, update_seconds=update_seconds, decode=decode_results,
         model_load_memory=load_memory,
         memory_definition='per-stage reset peak stats; allocated/reserved include live model; reserved may retain earlier cache; no warmup',
         timing_definition='CUDA synchronized before/after; load includes read/remap/CPU construction/GPU copy (asset preflight hashing separate); update includes forward/backward/clip/Adam; features recorded separately; decode includes GPU copy/mask/encoder/greedy/scoring',
         limitations='One update is no learning/quality or incremental streaming evidence; PI-WER only; inspect tokenizer-report for unknowns and Latin coverage'))
    (run / 'transcripts.txt').write_text('\n'.join(
        json.dumps(row, ensure_ascii=False) for item in decode_results for row in item['utterances']) + '\n')


if __name__ == '__main__':
    main()
