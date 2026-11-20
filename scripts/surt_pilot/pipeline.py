"""Stage 2 only. Kaggle-only execution, append-only artifacts, no downloader/trainer."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import tarfile
import time

from policy import VERSION, normalize

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/surt_pilot/pilot-v1.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def safe_name(value):
    if not isinstance(value, str) or Path(value).name != value or value in ('', '.', '..'):
        raise ValueError('source path must be a single original clip basename')
    if Path(value).suffix != '.mp3':
        raise ValueError('Expected official MP3 source basename')
    return value


def walk_records(value):
    if isinstance(value, dict):
        if 'source_id' in value or 'source_id_sha256' in value:
            yield value
        for item in value.values():
            yield from walk_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_records(item)


def exposure(paths):
    keys = ('speaker_hash', 'source_id', 'source_id_sha256', 'original_sha256', 'waveform_sha256')
    sets = {key: set() for key in keys}
    missing_speakers = 0
    for path in paths:
        for row in walk_records(read(path)):
            for key in keys:
                if row.get(key):
                    sets[key].add(row[key])
            if not row.get('speaker_hash'):
                missing_speakers += 1
    return sets, missing_speakers


def split_for(speaker, config):
    bucket = int(hashed(str(config['seed']) + ':' + speaker), 16) % 100
    boundary = 0
    for split, percent in config['split_percent'].items():
        boundary += percent
        if bucket < boundary:
            return split
    raise ValueError('split percentages must sum to 100')


def init(args):
    args.run.mkdir(exist_ok=False, parents=True)
    (args.run / 'logs').mkdir()
    for name in ('private', 'review'):
        (args.run / name).mkdir()
    shutil.copyfile(CONFIG, args.run / 'private/config.json')
    dump(args.run / 'private/implementation-lock.json', {
        str(path.relative_to(ROOT)): sha(path)
        for folder in ('scripts/surt_pilot', 'configs/surt_pilot')
        for path in sorted((ROOT / folder).rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts})
    source = read(CONFIG)['source']
    dump(args.run / 'review/status.json', dict(gate_2='OPEN',
         project_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
         config_sha256=sha(CONFIG), implementation_lock_sha256=sha(args.run / 'private/implementation-lock.json'),
         source=source, blockers=['Official bounded subset access not yet verified',
         'CV27 metadata/audio, mini overlap audit and Kaggle tests pending'],
         prior_gates='1A, 1B and technical 1C accepted; quality unproven'))
    print('Run تازه ساخته شد؛ سقف پایلوت ثبت شد؛ gate داده باز است و هیچ داده‌ای دریافت نشد.')


def plan(args):
    config = read(args.run / 'private/config.json')
    limits = config['limits']
    if args.metadata.stat().st_size > limits['metadata_bytes']:
        raise ValueError('Metadata byte cap exceeded')
    receipt = read(args.receipt)
    for key in ('dataset_id', 'release', 'locale'):
        if receipt.get(key) != config['source'][key]:
            raise ValueError('Official source mismatch: ' + key)
    for key in ('terms_accepted_at', 'access_verified_at', 'official_subset_route',
                'route_evidence', 'subset_origin_checksum'):
        if not receipt.get(key):
            raise ValueError('Access blocked: private receipt missing ' + key)
    if receipt.get('metadata_sha256') != sha(args.metadata):
        raise ValueError('Receipt does not bind this exact metadata snapshot')
    ledger_pin = read(ROOT / 'configs/surt_pilot/exposure-inventory-v1.json')
    prior = Path('/kaggle/working/surt-mini-1b/runs/mini-20261119-d')
    paths = []
    for item in ledger_pin['files']:
        path = ROOT / item['path']
        if not path.exists():
            path = prior / Path(item['path']).name
        if sha(path) != item['sha256']:
            raise ValueError('Prior exposure ledger changed: ' + str(path))
        paths.append(path)
    # Include every available additional failed/successful attempt, never just selected clips.
    discovered = set()
    for base in (ROOT / 'outputs', Path('/kaggle/working/surt-mini-1b/runs'), args.ledgers):
        if base and base.is_dir():
            for pattern in ('*ledger.json', 'selection.json'):
                discovered.update(base.rglob(pattern))
    for path in sorted(discovered - set(paths)):
        paths.append(path)
    prefix_exposure_path = ROOT / 'configs/surt_pilot/user-prefix-exposure-v1.json'
    prefix_exposure = read(prefix_exposure_path)
    paths.append(prefix_exposure_path)
    exposed, missing = exposure(paths)
    snapshot = rows(args.metadata)
    if len(snapshot) > limits['metadata_rows']:
        raise ValueError('Metadata row cap exceeded')
    source_counts = Counter(r.get('path') for r in snapshot
                            if isinstance(r, dict) and isinstance(r.get('path'), str))
    # Preserve every metadata inspection for future migrations, including exclusions.
    jsonl_rows = []
    for index, row in enumerate(snapshot):
        if isinstance(row, dict):
            client = row.get('client_id')
            jsonl_rows.append(dict(source_id=row.get('path'), metadata_row=index,
                speaker_hash=hashed('cv-client:' + client.strip())
                if isinstance(client, str) and client.strip() else None,
                original_sha256=row.get('original_sha256')))
    dump(args.run / 'private/exposure-ledger.json', dict(
         dataset=config['source'], metadata_sha256=sha(args.metadata), metadata=jsonl_rows,
         policy='All metadata inspected for split decisions; all splits used for coverage are exposed. Preserve for subsequent migrations; no test-driven tuning.'))
    candidates, rejected = [], []
    seen = {key: set() for key in ('source_id', 'original_sha256', 'text_normalized')}
    for index, row in enumerate(snapshot):
        reason = None
        try:
            if not isinstance(row, dict):
                raise ValueError('metadata row must be an object')
            source = safe_name(row['path'])
            if source_counts[source] != 1:
                raise ValueError('ambiguous/repeated source in metadata; all copies excluded')
            client = row['client_id']
            if not isinstance(client, str) or not client.strip() or client != client.strip():
                raise ValueError('missing/ambiguous client_id')
            if row.get('locale') != 'fa' or row.get('validated') is not True:
                raise ValueError('not confirmed validated Persian')
            if not isinstance(row['sentence'], str):
                raise ValueError('sentence must be a string')
            text = normalize(row['sentence'])
            if not text:
                raise ValueError('empty normalized text')
            duration = row.get('duration_metadata_seconds')
            if duration is not None and (not math.isfinite(duration) or not
                    limits['clip_seconds_min'] <= duration <= limits['clip_seconds_max']):
                raise ValueError('metadata duration outside budget')
            reserve = row.get('bytes') or limits['per_clip_bytes']
            if not isinstance(reserve, int) or not 0 < reserve <= limits['per_clip_bytes']:
                raise ValueError('clip byte cap')
            item = dict(schema_version=config['schema_version'], source_id=source,
                        source_id_sha256=hashed(source),
                        speaker_hash=hashed('cv-client:' + client), text_raw=row['sentence'], text_normalized=text,
                        text_policy=VERSION, original_sha256=row.get('original_sha256'),
                        reserved_bytes=reserve, duration_metadata_seconds=duration,
                        reserved_seconds=duration if duration is not None else limits['clip_seconds_max'],
                        metadata_row=index, provenance=config['source'])
            # Conservative policy: exclude all known mini speakers/sources from EVERY split.
            for key in exposed:
                if item.get(key) in exposed[key]:
                    raise ValueError('prior exposure: ' + key)
            candidates.append(item)
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error)
        if reason:
            rejected.append(dict(metadata_row=index, reason=reason))
    # Order-independent selection; a speaker is always assigned the same split.
    candidates.sort(key=lambda r: (hashed(str(config['seed']) + r['source_id']), r['source_id']))
    selected, counts, speakers, seconds, size = [], Counter(), set(), 0, 0
    for item in candidates:
        reason = None
        if any(item.get(k) and item[k] in seen[k] for k in seen):
            reason = 'duplicate source/hash/normalized sentence in candidate pool'
        elif counts[item['speaker_hash']] >= limits['clips_per_speaker']:
            reason = 'per speaker cap'
        elif item['speaker_hash'] not in speakers and len(speakers) >= limits['speakers']:
            reason = 'speaker cap'
        elif (len(selected) >= limits['clips'] or seconds + item['reserved_seconds'] > limits['duration_seconds']
              or size + item['reserved_bytes'] > limits['audio_import_bytes']):
            reason = 'pilot clips/duration/byte cap'
        if reason:
            rejected.append(dict(metadata_row=item['metadata_row'], reason=reason))
            continue
        for key in seen:
            if item.get(key):
                seen[key].add(item[key])
        item['split'] = split_for(item['speaker_hash'], config)
        selected.append(item)
        counts[item['speaker_hash']] += 1
        speakers.add(item['speaker_hash'])
        seconds += item['reserved_seconds']
        size += item['reserved_bytes']
    if any(sum(r['split'] == split for r in selected) < config['minimum_clips_per_split']
           for split in config['split_percent']):
        raise ValueError('Insufficient independent clips in a split; do not tune seed using test')
    for source, target in ((args.metadata, 'metadata.jsonl'), (args.receipt, 'access-receipt.json')):
        with (args.run / 'private' / target).open('xb') as out:
            out.write(source.read_bytes())
    jsonl(args.run / 'private/selection.jsonl', selected)
    jsonl(args.run / 'private/exclusions-plan.jsonl', rejected)
    dump(args.run / 'private/exposure-snapshot.json', {k: sorted(v) for k, v in exposed.items()})
    dump(args.run / 'private/plan-lock.json', dict(
        config_sha256=sha(args.run / 'private/config.json'), metadata_sha256=sha(args.metadata),
        receipt_sha256=sha(args.receipt), selection_sha256=sha(args.run / 'private/selection.jsonl'),
        ledgers=[dict(path=str(p), sha256=sha(p)) for p in paths],
        exposure_sha256=sha(args.run / 'private/exposure-snapshot.json'),
        unmapped_exposure_rows=missing, speaker_mapping_evidence_present=bool(receipt.get('speaker_mapping_evidence')),
        missing_user_prefix_exposure_rows=prefix_exposure['missing_exposure_rows'],
        exposure_history_evidence_present=bool(receipt.get('exposure_history_evidence')),
        reencoded_duplicate_review=bool(receipt.get('reencoded_duplicate_review_evidence')),
        frozen_before_augmentation=True, frozen_before_audio=True))
    dump(args.run / 'review/plan-summary.json', dict(clips=len(selected), speakers=len(speakers),
        splits=dict(Counter(r['split'] for r in selected)), reserved_seconds=seconds,
        reserved_bytes=size, exclusions=len(rejected), unmapped_exposure_rows=missing,
        missing_user_prefix_exposure_rows=prefix_exposure['missing_exposure_rows'],
        selection_sha256=sha(args.run / 'private/selection.jsonl'), gate_2='OPEN'))
    print(f'انتخاب فریز شد: {len(selected)} کلیپ، {len(speakers)} گوینده؛ سقف رزرو {seconds:.0f} ثانیه و {size} بایت. صوت هنوز خوانده نشده است.')


def locked(args):
    private = args.run / 'private'
    lock = read(private / 'plan-lock.json')
    for filename, key in (('config.json', 'config_sha256'), ('metadata.jsonl', 'metadata_sha256'),
                          ('selection.jsonl', 'selection_sha256'), ('access-receipt.json', 'receipt_sha256'),
                          ('exposure-snapshot.json', 'exposure_sha256')):
        if sha(private / filename) != lock[key]:
            raise ValueError('Frozen artifact changed: ' + filename)
    return private, read(private / 'config.json'), lock


def assert_disjoint(records):
    for key in ('speaker_hash', 'source_id', 'original_sha256', 'waveform_sha256'):
        seen = {}
        for row in records:
            value = row.get(key)
            if value and value in seen and seen[value] != row['split']:
                raise ValueError('Cross-split collision: ' + key)
            if value:
                seen[value] = row['split']


def audio(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    private, config, lock = locked(args)
    limits = config['limits']
    selected = rows(private / 'selection.jsonl')
    # Check ALL input sizes before any audio read/copy. No archive extraction or network.
    total = 0
    for row in selected:
        path = args.audio_root / row['source_id']
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(args.audio_root.resolve()):
            raise ValueError('Missing/unsafe bounded private clip: ' + row['source_id'])
        size = path.stat().st_size
        if not 0 < size <= row['reserved_bytes']:
            raise ValueError('Actual input exceeds reserved clip byte budget')
        total += size
    if total > limits['audio_import_bytes']:
        raise ValueError('Total byte budget exceeded before audio processing')
    output = private / 'audio'
    output.mkdir(exist_ok=False)
    exposure_sets = read(private / 'exposure-snapshot.json')
    seen_raw, seen_wave = set(exposure_sets['original_sha256']), set(exposure_sets['waveform_sha256'])
    accepted, excluded = [], []
    measured_total = 0
    for row in selected:
        try:
            path = args.audio_root / row['source_id']
            original = sha(path)
            if row['original_sha256'] and row['original_sha256'] != original:
                raise ValueError('Metadata original hash mismatch')
            info = sf.info(path)
            actual = info.frames / info.samplerate
            if not limits['clip_seconds_min'] <= actual <= limits['clip_seconds_max']:
                raise ValueError('Measured duration outside clip budget')
            if measured_total + actual > limits['duration_seconds']:
                raise ValueError('Measured total duration cap')
            data, rate = sf.read(path, dtype='float32', always_2d=True)
            if data.shape[0] != info.frames or not np.isfinite(data).all():
                raise ValueError('Invalid/partial/nonfinite decode')
            mono = data.mean(axis=1)
            divisor = math.gcd(rate, config['audio']['rate'])
            if rate != config['audio']['rate']:
                mono = resample_poly(mono, config['audio']['rate'] // divisor, rate // divisor).astype(np.float32)
            wave = hashlib.sha256(mono.astype('<f4').tobytes()).hexdigest()
            if original in seen_raw or wave in seen_wave:
                raise ValueError('Duplicate prior/selected original or canonical waveform; no reassignment')
            seen_raw.add(original)
            seen_wave.add(wave)
            target = output / (Path(row['source_id']).stem + '.wav')
            sf.write(target, mono, config['audio']['rate'], subtype='FLOAT')
            item = dict(row, original_sha256=original, waveform_sha256=wave, audio=str(target),
                        file_sha256=sha(target), duration_measured_seconds=actual,
                        duration_canonical_seconds=len(mono) / config['audio']['rate'],
                        original_rate=rate, original_channels=info.channels,
                        sample_rate=config['audio']['rate'], channels=1,
                        conversion='channel mean; scipy resample_poly; little-endian float32; no x32768',
                        auditory_acceptance=None)
            accepted.append(item)
            measured_total += actual
        except (RuntimeError, ValueError) as error:
            excluded.append(dict(source_id=row['source_id'], split=row['split'], reason=str(error)))
    assert_disjoint(accepted)
    jsonl(private / 'manifest-v1.jsonl', accepted)
    jsonl(private / 'exclusions-audio.jsonl', excluded)
    dump(private / 'manifest-lock.json', dict(sha256=sha(private / 'manifest-v1.jsonl'),
         plan_sha256=sha(private / 'plan-lock.json')))
    summary = {}
    for split in config['split_percent']:
        subset = [r for r in accepted if r['split'] == split]
        summary[split] = dict(clips=len(subset), speakers=len({r['speaker_hash'] for r in subset}),
             measured_seconds=sum(r['duration_measured_seconds'] for r in subset),
             metadata_seconds_known=sum(r['duration_metadata_seconds'] or 0 for r in subset),
             metadata_duration_missing=sum(r['duration_metadata_seconds'] is None for r in subset),
             original_rates=dict(Counter(r['original_rate'] for r in subset)),
             original_channels=dict(Counter(r['original_channels'] for r in subset)))
    dump(args.run / 'review/audio-summary.json', dict(splits=summary, excluded=len(excluded),
         imported_bytes=total, manifest_sha256=sha(private / 'manifest-v1.jsonl'),
         exact_hash_collisions_removed=True, reencoded_duplicates_proven_absent=False,
         auditory_acceptance=None, gate_2='OPEN'))
    if any(s['clips'] < config['minimum_clips_per_split'] for s in summary.values()):
        raise ValueError('Audio exclusions left insufficient split clips; frozen splits must not be refilled using test')
    print(f'صوت بررسی شد: {len(accepted)} سالم، {len(excluded)} حذف؛ {measured_total:.1f} ثانیه واقعی. شنیدن انسانی و duplicateهای بازکدگذاری‌شده هنوز تأیید نشده‌اند.')


def tokenizer(args):
    import sentencepiece as spm

    private, config, lock = locked(args)
    manifest = private / 'manifest-v1.jsonl'
    manifest_lock = read(private / 'manifest-lock.json')
    if sha(manifest) != manifest_lock['sha256'] or sha(private / 'plan-lock.json') != manifest_lock['plan_sha256']:
        raise ValueError('Manifest or plan changed')
    records = rows(manifest)
    assert_disjoint(records)
    target = private / 'tokenizer'
    target.mkdir(exist_ok=False)
    train = [r for r in records if r['split'] == 'train']
    text = target / 'train.txt'
    with text.open('x') as stream:
        stream.write(''.join(r['text_normalized'] + '\n' for r in train))
    spm.SentencePieceTrainer.train(input=str(text), model_prefix=str(target / 'bpe'), **config['tokenizer'])
    processor = spm.SentencePieceProcessor(model_file=str(target / 'bpe.model'))
    if processor.pad_id() != 0 or processor.unk_id() != 1:
        raise ValueError('Blank/unknown ID contract changed')
    with (target / 'tokens.txt').open('x') as stream:
        stream.write(''.join(f'{processor.id_to_piece(i)} {i}\n' for i in range(processor.get_piece_size())))
    # Written BEFORE opening held-out references for coverage; no fit or automatic tuning after this point.
    dump(target / 'freeze.json', dict(model_sha256=sha(target / 'bpe.model'),
         tokens_sha256=sha(target / 'tokens.txt'), config_sha256=sha(private / 'config.json'),
         train_text_sha256=sha(text), train_sources=[r['source_id'] for r in train],
         manifest_sha256=sha(manifest), sentencepiece_version=spm.__version__,
         test_tuning_allowed=False, transfer_compatibility='UNVERIFIED; semantic ID audit required'))
    train_chars = set(''.join(r['text_normalized'] for r in train))
    report = {}
    for split in config['split_percent']:
        count = unknown = characters = covered = roundtrips = latin_words = latin_unknown = 0
        for row in records:
            if row['split'] != split:
                continue
            ref = row['text_normalized']
            ids = processor.encode(ref, out_type=int)
            count += len(ids)
            unknown += ids.count(processor.unk_id())
            characters += len(ref)
            covered += sum(c in train_chars for c in ref)
            roundtrips += processor.decode(ids) != ref
            for word in ref.split():
                if any('a' <= c.lower() <= 'z' for c in word):
                    latin_words += 1
                    latin_unknown += processor.unk_id() in processor.encode(word, out_type=int)
        report[split] = dict(tokens=count, unknown=unknown, unknown_rate=unknown / count if count else None,
             characters=characters, train_character_coverage=covered / characters if characters else None,
             round_trip_failures=roundtrips, latin_words=latin_words, latin_words_with_unknown=latin_unknown)
    dump(args.run / 'review/tokenizer-summary.json', dict(splits=report,
         actual_vocab=processor.get_piece_size(), model_sha256=sha(target / 'bpe.model'),
         test_used_for_tuning=False, checkpoint_compatible='UNVERIFIED',
         code_switching_quality='not measured; text coverage only', gate_2='OPEN'))
    if report['train']['unknown'] or report['train']['round_trip_failures']:
        raise ValueError('Train tokenizer unknown/round-trip failure; preserve this attempt')
    print('BPE فقط با train ساخته و فریز شد؛ پوشش و unknown هر split ثبت شد. نتیجهٔ test مجوز تغییر tokenizer نیست؛ انتقال checkpoint هنوز ممیزی نشده است.')


def finalize(args):
    private, config, lock = locked(args)
    blockers = []
    for name in ('audio-summary.json', 'tokenizer-summary.json'):
        if not (args.run / 'review' / name).is_file():
            blockers.append('Missing ' + name)
    if not (args.run / 'tests-passed.json').is_file():
        blockers.append('Kaggle tests have not passed')
    if lock['unmapped_exposure_rows']:
        blockers.append('Prior exposed rows have unmapped speaker identities')
    if lock['missing_user_prefix_exposure_rows']:
        blockers.append('User prefix inspection exposed 5000 rows but only 3 identifiers were provided; preserve and review the full private exposure ledger')
    if not lock['speaker_mapping_evidence_present']:
        blockers.append('Cross-release speaker ID continuity not established')
    if not lock['exposure_history_evidence_present']:
        blockers.append('Completeness of all prior inspection/exposure ledgers not established')
    if not lock['reencoded_duplicate_review']:
        blockers.append('Reencoded/trimmed duplicate exclusion lacks independent evidence')
    if list(args.run.glob('failure-*.json')):
        blockers.append('This run contains failures; preserve and review them')
    if (private / 'manifest-v1.jsonl').exists():
        if sha(private / 'manifest-v1.jsonl') != read(private / 'manifest-lock.json')['sha256']:
            raise ValueError('Manifest changed after freeze')
        assert_disjoint(rows(private / 'manifest-v1.jsonl'))
    if (private / 'tokenizer/freeze.json').exists():
        freeze = read(private / 'tokenizer/freeze.json')
        for name, key in (('bpe.model', 'model_sha256'), ('tokens.txt', 'tokens_sha256')):
            if sha(private / 'tokenizer' / name) != freeze[key]:
                raise ValueError('Frozen tokenizer changed')
    dump(args.run / 'review/result.json', dict(gate_2='OPEN' if blockers else 'REVIEW_REQUIRED',
         blockers=blockers, checkpoint_transfer='UNVERIFIED; deferred to training stage',
         augmentation_started=False, training_started=False,
         manual_acceptance_required=True, plan_sha256=sha(private / 'plan-lock.json')))
    print(f'گزارش نهایی ثبت شد؛ {len(blockers)} مانع باقی است. gate خودکار بسته نمی‌شود؛ مرحلهٔ ۳ و آموزش شروع نشده‌اند.')


def archive(args):
    # Strict public allowlist: no transcripts, IDs, private manifests, vocab, audio or restart contents.
    chosen = list((args.run / 'review').glob('*.json'))
    test_result = args.run / 'tests-passed.json'
    if test_result.exists():
        chosen.append(test_result)
    if sum(p.stat().st_size for p in chosen) > read(CONFIG)['limits']['review_bytes']:
        raise ValueError('Review size cap')
    serial = str(time.time_ns())
    target = args.run.parent / f'{args.run.name}-review-{serial}.tar.gz'
    hashes = {str(p.relative_to(args.run)): sha(p) for p in chosen}
    dump(args.run / f'review-hashes-{serial}.json', hashes)
    chosen.append(args.run / f'review-hashes-{serial}.json')
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as bundle:
        for path in chosen:
            bundle.add(path, arcname=str(path.relative_to(args.run)), recursive=False)
    print(f'آرشیو کوچکِ بدون متن و صوت ساخته شد: {target}؛ restart خصوصی جدا حفظ شود.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('init', 'plan', 'audio', 'tokenizer', 'finalize', 'archive'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--ledgers', type=Path)
    parser.add_argument('--audio-root', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
        raise RuntimeError('Execution, tests and audio work are Kaggle-only')
    if args.command != 'init':
        expected = (args.run / 'project-commit.txt').read_text().strip()
        actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError('Project SHA changed since run initialization')
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', '--quiet'], check=True)
    def expired(signum, frame):
        raise TimeoutError('Command wall-time cap exceeded; preserve partial outputs and use a fresh Run ID')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(read(CONFIG)['limits']['wall_seconds_per_command'])
    try:
        globals()[args.command](args)
    except Exception as error:
        if args.run.is_dir():
            dump(args.run / f'failure-{time.time_ns()}.json', dict(command=args.command,
                 error=type(error).__name__, reason=str(error), gate_2='OPEN'))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
