"""Explicit semantic row transfer; never infer token compatibility from shape."""
from contracts import VOCAB_TENSORS, dump, token_mapping


def transfer(model, state, old_sp, new_sp, report_path):
    import torch

    target = model.state_dict()
    report = dict(token_rows=token_mapping(old_sp, new_sp), tensors=[], optimizer='fresh Adam; no imported state')
    if set(state) != set(target):
        raise RuntimeError(f'State key mismatch: missing={set(target)-set(state)}, unexpected={set(state)-set(target)}')
    if not VOCAB_TENSORS.issubset(target):
        raise RuntimeError('Recipe vocabulary tensor inventory changed')
    old_size, new_size = old_sp.get_piece_size(), new_sp.get_piece_size()
    for name, destination in target.items():
        source = state[name]
        if source.is_floating_point() and not torch.isfinite(source).all():
            raise RuntimeError(f'Nonfinite official tensor: {name}')
        row = dict(name=name, source_shape=list(source.shape), target_shape=list(destination.shape))
        if name in VOCAB_TENSORS:
            if source.shape[0] != old_size or destination.shape[0] != new_size or source.shape[1:] != destination.shape[1:]:
                raise RuntimeError(f'Unexpected vocabulary axis: {name}')
            for mapping in report['token_rows']:
                destination[mapping['new_id']].copy_(source[mapping['old_id']])
            row.update(action='semantic row remap + fresh initialization for remaining rows',
                       transferred_rows=len(report['token_rows']), fresh_rows=new_size - len(report['token_rows']),
                       reason='vocabulary axis 0; exact ordinary piece or matching special role')
        else:
            if source.shape != destination.shape:
                raise RuntimeError(f'Unclassified changed tensor: {name}')
            if old_size in source.shape:
                raise RuntimeError(f'Unclassified vocabulary-sized axis requires review: {name}')
            destination.copy_(source)
            row.update(action='full', reason='same pinned non-vocabulary parameter role and shape')
        report['tensors'].append(row)
    model.load_state_dict(target, strict=True)
    dump(report_path, report)
    return report
