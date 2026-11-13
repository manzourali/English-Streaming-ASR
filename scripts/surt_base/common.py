"""Small, dependency-free contracts and scoring for the bounded English gate."""
import hashlib
import itertools
import json
import re
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def normalize(text):
    # LibriSpeech-style uppercase, preserve internal apostrophes; no hidden LM.
    return re.findall(r"[A-Z]+(?:'[A-Z]+)*", text.upper())


def errors(reference, hypothesis):
    """Levenshtein counts, deterministic ties: substitution, deletion, insertion."""
    r, h = normalize(reference), normalize(hypothesis)
    previous = [(j, 0, 0, j) for j in range(len(h) + 1)]
    for i, word in enumerate(r, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(h, 1):
            if word == other:
                current.append(previous[j - 1])
                continue
            c, s, d, ins = previous[j - 1]
            sub = (c + 1, s + 1, d, ins)
            c, s, d, ins = previous[j]
            delete = (c + 1, s, d + 1, ins)
            c, s, d, ins = current[j - 1]
            insert = (c + 1, s, d, ins + 1)
            current.append(min((sub, delete, insert), key=lambda x: x[0]))
        previous = current
    e, s, d, ins = previous[-1]
    return dict(errors=e, substitutions=s, deletions=d, insertions=ins, words=len(r))


def score(references, hypotheses):
    """Two-stream permutation-invariant WER; unused clean stream counts insertions."""
    if len(hypotheses) != 2 or len(references) not in (1, 2):
        raise ValueError('Expected 1 or 2 references and exactly 2 hypotheses')
    refs = references + [''] * (2 - len(references))
    choices = []
    for order in itertools.permutations(range(2)):
        counts = [errors(refs[i], hypotheses[order[i]]) for i in range(2)]
        total = {k: sum(c[k] for c in counts) for k in counts[0]}
        choices.append(dict(assignment=list(order), branches=counts, **total))
    best = min(choices, key=lambda c: c['errors'])
    if best['words'] == 0:
        raise ValueError('Empty reference')
    return dict(metric='two-stream PI-WER (bounded utterances)',
                wer=best['errors'] / best['words'], **best)


def validate_records(records):
    if len(records) != 2:
        raise ValueError('Exactly two real clean utterances required')
    for row in records:
        for key in ('id', 'speaker_id', 'text', 'source', 'license', 'audio'):
            if not str(row.get(key, '')).strip():
                raise ValueError(f'Missing {key}')
        if not normalize(row['text']):
            raise ValueError('Empty English reference')
    if str(records[0]['speaker_id']) == str(records[1]['speaker_id']):
        raise ValueError('Mixture requires different speakers')
    if records[0]['id'] == records[1]['id']:
        raise ValueError('Duplicate utterance')
