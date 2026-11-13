"""Kaggle only: official assets and two real utterances; never a corpus download."""
import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from common import dump, sha256, validate_records

ROOT = Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())


def fetch(url, target, limit):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response, target.open('wb') as out:
        count = 0
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > limit:
                raise ValueError(f'Download exceeds {limit} bytes: {url}')
            out.write(block)


def get_json(url, target):
    fetch(url, target, 1024 * 1024)
    return json.loads(Path(target).read_text())


def assets(out):
    # All large downloads are guarded against accidental Mac execution.
    for rel, digest in PINS['assets'].items():
        target = out / 'model' / rel
        if not target.exists():
            fetch(f"https://huggingface.co/{PINS['model_repo']}/resolve/"
                  f"{PINS['model_revision']}/{rel}", target, 110 * 1024 * 1024)
        if sha256(target) != digest:
            raise ValueError(f'Hash mismatch: {target}')


def fallback(out):
    info = get_json(f"https://huggingface.co/api/datasets/{PINS['dataset_repo']}",
                    out / 'dataset-metadata.json')
    if info['sha'] != PINS['dataset_revision']:
        raise ValueError('Dataset viewer moved: attach the pinned clips; do not change pins silently')
    records = []
    for expected in PINS['samples']:
        query = urllib.parse.urlencode(dict(dataset=PINS['dataset_repo'], config='clean',
                                           split='test', offset=expected['row'], length=1))
        response = get_json('https://datasets-server.huggingface.co/rows?' + query,
                            out / f"row-{expected['row']}.json")
        actual = response['rows'][0]['row']
        for key in ('id', 'speaker_id', 'text'):
            if actual[key] != expected[key]:
                raise ValueError(f'Dataset row drift: {key}')
        url = actual['audio'][0]['src']
        if f"/--/{PINS['dataset_revision']}/--/" not in url:
            raise ValueError('Unpinned cached audio URL')
        audio = out / 'downloads' / f"{actual['id']}.flac"
        fetch(url, audio, 2 * 1024 * 1024)
        records.append(dict(id=actual['id'], speaker_id=actual['speaker_id'],
                            text=actual['text'], audio=str(audio),
                            source='LibriSpeech test-clean / openslr/librispeech_asr@' +
                            PINS['dataset_revision'], license='CC BY 4.0; Panayotov et al., 2015'))
    return records


def mix_waves(waves, offset=8000):
    """Retain full waveforms at fixed gains; return mixture and true overlap samples."""
    import numpy as np
    if len(waves) != 2 or offset < 0:
        raise ValueError('Expected two waves and a nonnegative offset')
    mix = np.zeros(max(len(waves[0]), offset + len(waves[1])), dtype=np.float32)
    mix[:len(waves[0])] += 0.5 * waves[0]
    mix[offset:offset + len(waves[1])] += 0.5 * waves[1]
    overlap = max(0, min(len(waves[0]), offset + len(waves[1])) - offset)
    if overlap <= 0:
        raise ValueError('No real overlap')
    return mix, overlap


def build(out, attached=None):
    import numpy as np
    import soundfile as sf

    if attached:
        attached = attached.resolve()
        if not attached.is_relative_to(Path('/kaggle/input')):
            raise ValueError('Attached manifest must be under read-only /kaggle/input')
        records = json.loads(attached.read_text())
        validate_records(records)
        for row in records:
            path = (attached.parent / row['audio']).resolve()
            if not path.is_relative_to(attached.parent):
                raise ValueError('Audio must be inside the attached dataset directory')
            row['audio'] = str(path)
        shutil.copyfile(attached, out / 'attached-input.json')
    else:
        records = fallback(out)
    validate_records(records)
    waves, items = [], []
    for i, row in enumerate(records):
        info = sf.info(row['audio'])
        if info.samplerate != 16000 or info.channels != 1 or not 1 <= info.duration <= 12:
            raise ValueError(f'Expected mono 16kHz, 1–12 seconds: {row["audio"]}')
        wave, sr = sf.read(row['audio'], dtype='float32')
        if not np.isfinite(wave).all() or not 0 < np.max(np.abs(wave)) <= 1:
            raise ValueError('Nonfinite, silent or out-of-range audio')
        dest = out / f'clean-{i}.wav'
        sf.write(dest, wave, sr, subtype='FLOAT')
        waves.append(wave)
        items.append(dict(id=row['id'], kind='clean', audio=str(dest),
                          duration=len(wave) / sr, references=[row['text']],
                          segments=[dict(speaker_id=row['speaker_id'], start=0,
                                         duration=len(wave) / sr, text=row['text'])],
                          provenance=[dict(row, original_sha256=sha256(row['audio']))]))
    offset = 8000  # Real 0.5s offset, full utterances retained, no invented word timing.
    mix, overlap_samples = mix_waves(waves, offset)
    dest = out / 'mixture.wav'
    sf.write(dest, mix, 16000, subtype='FLOAT')
    segments = [dict(items[0]['segments'][0]), dict(items[1]['segments'][0], start=0.5)]
    overlap = overlap_samples / 16000
    items.append(dict(id='two-speaker-mixture', kind='mixture', audio=str(dest),
                      duration=len(mix) / 16000, references=[r['text'] for r in records],
                      segments=segments, gains=[0.5, 0.5], overlap_seconds=overlap,
                      overlap_fraction=overlap / (len(mix) / 16000),
                      provenance=[i['provenance'][0] for i in items]))
    for item in items:
        item['sha256'] = sha256(item['audio'])
    dump(out / 'inputs.json', items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--attached', type=Path)
    args = p.parse_args()
    if not Path('/kaggle/working').is_dir() or not args.output.resolve().is_relative_to(Path('/kaggle/working')):
        p.error('This downloader is restricted to /kaggle/working')
    args.output.mkdir(parents=True, exist_ok=False)
    assets(args.output)
    build(args.output, args.attached)


if __name__ == '__main__':
    main()
