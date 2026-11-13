"""Kaggle-only individual viewer assets with hard transport and cache budgets."""
import argparse
import json
import math
import platform
import signal
import time
import urllib.parse
import urllib.request
from pathlib import Path

from contracts import PINS, assert_disjoint, digest, dump, file_hash, identity, normalize


class Budget:
    def __init__(self, cache):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.begin = time.monotonic()
        self.bytes = 0
        self.rows = 0
        self.attempts = 0

    def check(self):
        limits = PINS['limits']
        if time.monotonic() - self.begin >= limits['seconds']:
            raise RuntimeError('Download wall-time budget exhausted')
        if sum(path.stat().st_size for path in self.cache.rglob('*') if path.is_file()) > limits['cache_bytes']:
            raise RuntimeError('Persistent cache budget exhausted; preserve it for review')

    def get(self, url, limit, cache_key=None):
        self.check()
        cached = self.cache / cache_key if cache_key else None
        if cached and cached.exists():
            if cached.stat().st_size > limit:
                raise RuntimeError('Cached asset exceeds per-file budget')
            return cached.read_bytes()
        chunks = []
        size = 0
        with urllib.request.urlopen(url, timeout=min(20, max(1, PINS['limits']['seconds'] - (time.monotonic() - self.begin)))) as response:
            while True:
                self.check()
                remaining = min(limit - size, PINS['limits']['network_bytes'] - self.bytes)
                if remaining <= 0:
                    raise RuntimeError('Transport/per-file byte budget exhausted')
                block = response.read(min(65536, remaining))
                self.bytes += len(block)
                size += len(block)
                if size > limit or self.bytes > PINS['limits']['network_bytes']:
                    raise RuntimeError('Transport byte budget exhausted')
                if not block:
                    break
                chunks.append(block)
        value = b''.join(chunks)
        if cached:
            current = sum(path.stat().st_size for path in self.cache.rglob('*') if path.is_file())
            if current + len(value) > PINS['limits']['cache_bytes']:
                raise RuntimeError('Persistent cache would exceed budget')
            with cached.open('xb') as stream:
                stream.write(value)
        return value

    def report(self):
        return dict(rows_inspected=self.rows, audio_attempts=self.attempts,
                    downloaded_bytes=self.bytes, elapsed_seconds=time.monotonic() - self.begin,
                    cache_bytes=sum(path.stat().st_size for path in self.cache.rglob('*') if path.is_file()),
                    limits=PINS['limits'])


def repository(budget):
    url = 'https://huggingface.co/api/datasets/' + PINS['dataset_repo']
    meta = json.loads(budget.get(url, 1024 * 1024))
    if meta['sha'] != PINS['dataset_revision'] or meta.get('gated') or meta.get('private') or meta.get('disabled'):
        raise RuntimeError('Repository changed or is no longer public/ungated; review explicitly')
    info = meta['cardData']['dataset_info']
    schema = {feature['name']: feature['dtype'] for feature in info['features']}
    if (any(schema.get(name) != 'string' for name in ('sentence', 'client_id', 'path', 'sentence_id'))
            or schema.get('audio') != {'audio': {'sampling_rate': 16000}}
            or {split['name']: split['num_examples'] for split in info['splits']} != PINS['splits']):
        raise RuntimeError('Pinned dataset schema/split metadata mismatch')
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('cache', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
        raise RuntimeError('Real audio download is authorized only in Kaggle')
    import io
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    args.run.mkdir(parents=True, exist_ok=True)
    output = args.run / 'data'
    output.mkdir(exist_ok=False)
    budget = Budget(args.cache)
    def deadline(signum, frame):
        raise TimeoutError('Hard 300-second download deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(PINS['limits']['seconds'])
    records, rejected, inspected, attempted = [], [], [], []
    try:
        meta = repository(budget)
        readme = budget.get(f"https://huggingface.co/datasets/{PINS['dataset_repo']}/raw/{PINS['dataset_revision']}/README.md", 1024 * 1024).decode()
        if '**Public Domain, CC-0**' not in readme:
            raise RuntimeError('Declared license changed')
        dump(args.run / 'dataset-metadata.json', dict(repository=meta, readme=readme))
        candidates = []
        for original_split in PINS['splits']:
            for offset in PINS['metadata_offsets']:
                budget.check()
                query = urllib.parse.urlencode(dict(dataset=PINS['dataset_repo'], config=PINS['config'],
                                                    split=original_split, offset=offset, length=PINS['rows_per_page']))
                page = json.loads(budget.get('https://datasets-server.huggingface.co/rows?' + query, 1024 * 1024))
                if len(page['rows']) > PINS['rows_per_page']:
                    raise RuntimeError('Unexpected viewer page size')
                for item in page['rows']:
                    budget.rows += 1
                    if budget.rows > PINS['limits']['rows']:
                        raise RuntimeError('Metadata row budget exhausted')
                    if item.get('truncated_cells'):
                        continue
                    row = item['row']
                    inspection = dict(source_id=row.get('path'), original_split=original_split,
                                      row_index=item['row_idx'], speaker_hash=None)
                    inspected.append(inspection)
                    try:
                        speaker = identity(row)
                    except ValueError:
                        continue
                    inspection['speaker_hash'] = speaker
                    candidates.append((digest((str(PINS['seed']) + original_split + row['path']).encode()),
                                       speaker, original_split, item['row_idx'], row))
        used_speakers, used_ids, used_audio, used_wave = set(), set(), set(), set()
        split_slots = [split for split, count in PINS['mini_split_counts'].items() for unused in range(count)]
        for rank, speaker, original_split, row_index, row in sorted(candidates):
            budget.check()
            if len(records) == PINS['limits']['clips']:
                break
            if speaker in used_speakers or row['path'] in used_ids:
                continue
            if budget.attempts >= PINS['limits']['audio_attempts']:
                break
            audio = row.get('audio')
            if not isinstance(audio, list) or len(audio) != 1:
                continue
            url = audio[0]['src']
            parsed = urllib.parse.urlparse(url)
            expected = f"/assets/{PINS['dataset_repo']}/--/{PINS['dataset_revision']}/--/{PINS['config']}/{original_split}/{row_index}/audio/"
            if parsed.scheme != 'https' or parsed.hostname != 'datasets-server.huggingface.co' or not parsed.path.startswith(expected):
                raise RuntimeError('Viewer asset revision/path mismatch')
            budget.attempts += 1
            attempt = dict(source_id=row['path'], speaker_hash=speaker, original_split=original_split,
                           row_index=row_index, status='attempted; not selected')
            attempted.append(attempt)
            raw = budget.get(url, PINS['limits']['clip_bytes'], digest(parsed.path.encode()) + '.audio')
            raw_hash = digest(raw)
            attempt['original_sha256'] = raw_hash
            with sf.SoundFile(io.BytesIO(raw)) as stream:
                rate, channels, frames = stream.samplerate, stream.channels, len(stream)
                duration = frames / rate
                if not PINS['limits']['min_clip_seconds'] <= duration <= PINS['limits']['max_clip_seconds']:
                    rejected.append(dict(source_id=row['path'], reason='duration', seconds=duration))
                    continue
                if not 8000 <= rate <= 96000 or not 1 <= channels <= 2:
                    rejected.append(dict(source_id=row['path'], reason='rate/channels'))
                    continue
                wave = stream.read(dtype='float32', always_2d=True)
            if not np.isfinite(wave).all() or not np.any(wave):
                rejected.append(dict(source_id=row['path'], reason='nonfinite/silent'))
                continue
            wave = wave.mean(axis=1)
            divisor = math.gcd(rate, 16000)
            wave = resample_poly(wave, 16000 // divisor, rate // divisor).astype('float32') if rate != 16000 else wave
            wave_hash = digest(wave.astype('<f4').tobytes())
            attempt['waveform_sha256'] = wave_hash
            if raw_hash in used_audio or wave_hash in used_wave:
                continue
            if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.0:
                rejected.append(dict(source_id=row['path'], reason='converted nonfinite/out-of-range'))
                continue
            filename = output / f'clean-{len(records)}.wav'
            sf.write(filename, wave, 16000, subtype='FLOAT')
            records.append(dict(source_id=row['path'], sentence_id=row.get('sentence_id'),
                                original_split=original_split, row_index=row_index,
                                split=split_slots[len(records)], speaker_hash=speaker,
                                text_raw=row['sentence'], text=normalize(row['sentence']),
                                original_sha256=raw_hash, waveform_sha256=wave_hash,
                                file_sha256=file_hash(filename), audio=str(filename.resolve()),
                                duration=len(wave) / 16000, original_rate=rate, original_channels=channels,
                                original_frames=frames, conversion='channel mean; scipy resample_poly to 16kHz; float32, no x32768',
                                dataset=PINS['dataset_repo'], revision=PINS['dataset_revision']))
            used_speakers.add(speaker)
            used_ids.add(row['path'])
            used_audio.add(raw_hash)
            used_wave.add(wave_hash)
            attempt['status'] = 'selected'
        if len(records) != PINS['limits']['clips']:
            raise RuntimeError(f'Only {len(records)}/8 suitable distinct speakers within budget; no fallback')
        assert_disjoint(records)
        if sum(row['duration'] for row in records) > PINS['limits']['audio_seconds']:
            raise RuntimeError('Selected audio duration budget exceeded')
        repository(budget)
        dump(args.run / 'selection.json', records)
        dump(args.run / 'exposure-ledger.json', dict(samples=records,
             policy='All selected mini samples and conservatively all metadata/audio inspections in inspection-ledger.json are exposed to pipeline decisions; exclude colliding speakers, IDs and audio from final CV27 test, including re-encoded duplicates'))
    finally:
        signal.alarm(0)
        dump(args.run / 'selection-partial.json', records)
        dump(args.run / 'inspection-ledger.json', dict(dataset=PINS['dataset_repo'],
             revision=PINS['dataset_revision'], metadata=inspected, audio_attempts=attempted,
             migration_policy='Conservatively exclude all inspected source IDs and nonempty speaker hashes from final test, plus audio duplicates; retain this ledger even when the mini attempt fails'))
        dump(args.run / 'fetch-budget.json', dict(**budget.report(), rejected=rejected))


if __name__ == '__main__':
    main()
