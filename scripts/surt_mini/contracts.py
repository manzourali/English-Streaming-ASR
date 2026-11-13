"""Dependency-free policies for the real Persian mini gate."""
import hashlib
import itertools
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / 'configs/surt_mini/pins.json').read_text())
VOCAB_TENSORS = {
    'decoder.embedding.weight', 'joiner.output_linear.weight',
    'joiner.output_linear.bias', 'simple_am_proj.weight', 'simple_am_proj.bias',
    'simple_lm_proj.weight', 'simple_lm_proj.bias',
    'ctc_output.1.weight', 'ctc_output.1.bias',
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def dump(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(content)


def normalize(text):
    text = unicodedata.normalize('NFC', text).translate(str.maketrans(
        'يىك۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', 'ییک01234567890123456789')).lower()
    result = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith('M') or char == '\u0640':
            continue
        if char == '\u200c':
            result.append(char)
        elif category[0] in 'LN' or char.isspace():
            result.append(char)
        else:
            result.append(' ')
    return ' '.join(''.join(result).split()).strip('\u200c ')


def identity(row):
    speaker = row.get('client_id')
    if not isinstance(speaker, str) or not speaker.strip():
        raise ValueError('Missing client_id')
    if not isinstance(row.get('sentence'), str) or not normalize(row['sentence']):
        raise ValueError('Missing usable reference')
    if not isinstance(row.get('path'), str) or not row['path'].strip():
        raise ValueError('Missing source path')
    return digest(('cv-client:' + speaker.strip()).encode())


def assert_disjoint(records):
    for key in ('speaker_hash', 'source_id', 'original_sha256', 'waveform_sha256'):
        seen = {}
        for row in records:
            value = row[key]
            if value in seen:
                raise ValueError(f'Duplicate {key}: {value}')
            seen[value] = row['split']
    for split, count in PINS['mini_split_counts'].items():
        if sum(row['split'] == split for row in records) != count:
            raise ValueError(f'Wrong {split} count')


def token_mapping(old, new):
    mapping = []
    for new_id in range(new.get_piece_size()):
        piece = new.id_to_piece(new_id)
        old_id = old.piece_to_id(piece)
        reason = 'identical ordinary piece and identity normalization'
        if new_id == new.pad_id():
            old_id, reason = old.piece_to_id('<blk>'), 'blank role'
            if old.id_to_piece(old_id) != '<blk>':
                raise ValueError('Old blank not found')
        elif new_id == new.unk_id():
            old_id, reason = old.unk_id(), 'unknown role'
        elif (new.is_control(new_id) or new.is_unused(new_id)
              or old_id == old.unk_id() or old.id_to_piece(old_id) != piece
              or old.is_control(old_id) or old.is_unused(old_id)):
            continue
        mapping.append(dict(old_id=old_id, new_id=new_id, piece=piece, reason=reason))
    return mapping


def errors(reference, hypothesis):
    reference, hypothesis = normalize(reference).split(), normalize(hypothesis).split()
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, word in enumerate(reference, 1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, other in enumerate(hypothesis, 1):
            if word == other:
                current.append(previous[hyp_index - 1])
                continue
            cost, sub, delete, insert = previous[hyp_index - 1]
            choices = [(cost + 1, sub + 1, delete, insert)]
            cost, sub, delete, insert = previous[hyp_index]
            choices.append((cost + 1, sub, delete + 1, insert))
            cost, sub, delete, insert = current[-1]
            choices.append((cost + 1, sub, delete, insert + 1))
            current.append(min(choices, key=lambda item: item[0]))
        previous = current
    total, sub, delete, insert = previous[-1]
    return dict(errors=total, substitutions=sub, deletions=delete,
                insertions=insert, words=len(reference))


def score(references, hypotheses):
    if len(references) != 2 or len(hypotheses) != 2:
        raise ValueError('Exactly two streams required')
    choices = []
    for order in itertools.permutations(range(2)):
        branches = [errors(ref, hypotheses[order[index]])
                    for index, ref in enumerate(references)]
        total = {key: sum(branch[key] for branch in branches) for key in branches[0]}
        choices.append(dict(assignment=list(order), branches=branches, **total))
    best = min(choices, key=lambda item: item['errors'])
    return dict(metric='bounded two-stream PI-WER; not session ORC-WER',
                wer=best['errors'] / best['words'] if best['words'] else None, **best)
