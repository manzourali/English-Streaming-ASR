"""Train-only BPE and full-duration mixtures using pinned Lhotse contracts."""
import argparse
import json
import time
from pathlib import Path

from contracts import PINS, assert_disjoint, digest, dump, file_hash


def tokenizer(run, records):
    import sentencepiece as spm

    directory = run / 'tokenizer'
    directory.mkdir(exist_ok=False)
    training = [row for row in records if row['split'] == 'train']
    text_file = directory / 'train.txt'
    text_file.write_text(''.join(row['text'] + '\n' for row in training))
    prefix = directory / 'bpe'
    spm.SentencePieceTrainer.train(
        input=str(text_file), model_prefix=str(prefix), model_type='bpe',
        vocab_size=PINS['vocab_size_requested'], hard_vocab_limit=False,
        pad_id=0, pad_piece='<blk>', unk_id=1, bos_id=-1, eos_id=-1,
        normalization_rule_name='identity', remove_extra_whitespaces=False,
        shuffle_input_sentence=False, num_threads=1, character_coverage=1.0,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + '.model')
    pieces = [processor.id_to_piece(index) for index in range(processor.get_piece_size())]
    (directory / 'tokens.txt').write_text(''.join(f'{piece} {index}\n' for index, piece in enumerate(pieces)))
    report = dict(requested_vocab=PINS['vocab_size_requested'], actual_vocab=len(pieces),
                  blank=processor.pad_id(), unknown=processor.unk_id(),
                  pieces=[dict(id=index, piece=piece) for index, piece in enumerate(pieces)],
                  training_sources=[row['source_id'] for row in training],
                  training_text_sha256=file_hash(text_file), splits={})
    for split in PINS['mini_split_counts']:
        rows = []
        for row in records:
            if row['split'] != split:
                continue
            tokens = processor.encode(row['text'], out_type=int)
            rows.append(dict(source_id=row['source_id'], ids=tokens, length=len(tokens),
                             unknown=tokens.count(processor.unk_id()),
                             round_trip=processor.decode(tokens) == row['text'],
                             decoded=processor.decode(tokens), normalized=row['text']))
        report['splits'][split] = dict(rows=rows, tokens=sum(row['length'] for row in rows),
                                      unknown=sum(row['unknown'] for row in rows),
                                      latin_references=sum(any('a' <= char <= 'z' for char in row['normalized']) for row in rows),
                                      latin_words=sum(any('a' <= char <= 'z' for char in word)
                                                      for row in rows for word in row['normalized'].split()),
                                      round_trip_failures=sum(not row['round_trip'] for row in rows))
        summary = report['splits'][split]
        summary['unknown_rate'] = summary['unknown'] / summary['tokens'] if summary['tokens'] else None
    report['code_switching'] = 'Latin counts are textual coverage only; real code-switching recognition not assessed; zero/sparse counts limit coverage conclusions'
    dump(run / 'tokenizer-report.json', report)
    if processor.pad_id() != 0 or processor.unk_id() != 1:
        raise RuntimeError('Unexpected special IDs')
    if any(row['unknown'] or not row['round_trip'] for row in report['splits']['train']['rows']):
        raise RuntimeError('Train tokenizer round-trip/unknown failure')
    return processor


def features(run, records):
    import numpy as np
    import soundfile as sf
    import torch
    from lhotse import CutSet, NumpyFilesWriter, Recording, SupervisionSegment
    from lhotse.dataset import K2SurtDataset
    from lhotse.features.kaldifeat import KaldifeatFbank, KaldifeatFbankConfig

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    directory = run / 'features'
    directory.mkdir(exist_ok=False)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device='cuda'))
    config = extractor.config.to_dict()
    frame = config['frame_opts']
    assert frame['samp_freq'] == 16000 and frame['frame_shift_ms'] == 10
    assert frame['frame_length_ms'] == 25 and not frame['snip_edges'] and frame['dither'] == 0
    assert config['mel_opts']['num_bins'] == 80 and config['mel_opts']['high_freq'] == -400
    manifests, mixtures, contracts = {}, [], []
    with NumpyFilesWriter(directory / 'source-arrays') as writer, NumpyFilesWriter(directory / 'mixtures') as mixture_writer:
        for split in PINS['mini_split_counts']:
            selected = [row for row in records if row['split'] == split]
            cuts = []
            for pair_index in range(0, len(selected), 2):
                pair = selected[pair_index:pair_index + 2]
                assert len(pair) == 2 and pair[0]['speaker_hash'] != pair[1]['speaker_hash']
                waves = [sf.read(row['audio'], dtype='float32')[0] for row in pair]
                offset = 8000
                gains = [0.5, 0.5]
                length = max(len(waves[0]), offset + len(waves[1]))
                mixed = np.zeros(length, dtype=np.float32)
                mixed[:len(waves[0])] += gains[0] * waves[0]
                mixed[offset:offset + len(waves[1])] += gains[1] * waves[1]
                overlap = min(len(waves[0]), offset + len(waves[1])) - offset
                assert overlap > 0 and np.isfinite(mixed).all() and np.max(np.abs(mixed)) <= 1
                identifier = f'{split}-{pair_index // 2}'
                audio_path = run / 'data' / (identifier + '.wav')
                sf.write(audio_path, mixed, 16000, subtype='FLOAT')
                recording = Recording.from_file(audio_path, recording_id=identifier)
                cut = recording.to_cut()
                cut.supervisions = [SupervisionSegment(
                    id=row['source_id'], recording_id=recording.id,
                    start=start, duration=len(wave) / 16000, channel=0,
                    text=row['text'], speaker=row['speaker_hash'], language='Farsi',
                ) for row, wave, start in zip(pair, waves, (0.0, 0.5))]
                cut = cut.compute_and_store_features(extractor, storage=mixture_writer)
                sources = [extractor.extract(gain * wave, 16000) for gain, wave in zip(gains, waves)]
                assert all(np.isfinite(source).all() and source.shape[1] == 80 for source in sources)
                cut.source_feats = writer.store_array(identifier, np.concatenate(sources, axis=0))
                cut.source_feat_offsets = [0, sources[0].shape[0]]
                cuts.append(cut)
                mixtures.append(dict(id=identifier, split=split, audio=str(audio_path),
                                     sha256=file_hash(audio_path), waveform_sha256=digest(mixed.astype('<f4').tobytes()),
                                     sources=[row['source_id'] for row in pair],
                                     references=[row['text'] for row in pair],
                                     gains=gains, offsets_seconds=[0, 0.5], duration=length / 16000,
                                     overlap_seconds=overlap / 16000, overlap_fraction=overlap / length,
                                     clean_audio=[row['audio'] for row in pair],
                                     source_feature_frames=[source.shape[0] for source in sources]))
            cutset = CutSet.from_cuts(cuts)
            manifest = directory / (split + '.jsonl.gz')
            cutset.to_file(manifest)
            dataset = K2SurtDataset(return_sources=True, return_cuts=True, num_channels=2, strict=True)
            batch = dataset[CutSet.from_file(manifest)]
            assert batch['inputs'].ndim == 3 and batch['inputs'].shape[-1] == 80
            assert len(batch['text']) == len(cuts) and torch.isfinite(batch['inputs']).all()
            for index, cut in enumerate(batch['cuts']):
                expected = [sup.text for sup in sorted(cut.supervisions, key=lambda sup: sup.start)]
                assert batch['text'][index] == expected
                boundaries = batch['source_boundaries'][index]
                assert boundaries[0][0] == 0 and boundaries[1][0] == 50
                assert boundaries[0][1] > boundaries[1][0]
                for source, (start, end) in zip(batch['source_feats'][index], boundaries):
                    assert source.shape == (end - start, 80) and torch.isfinite(source).all()
                    assert 0 <= start < end <= batch['input_lens'][index]
            contracts.append(dict(split=split, inputs=list(batch['inputs'].shape),
                                  input_lens=batch['input_lens'].tolist(), text=batch['text'],
                                  boundaries=batch['source_boundaries'],
                                  cut_ids=[cut.id for cut in batch['cuts']],
                                  source_shapes=[[list(source.shape) for source in sources] for sources in batch['source_feats']]))
            manifests[split] = str(manifest)
    torch.cuda.synchronize()
    dump(run / 'mixtures.json', mixtures)
    dump(run / 'feature-contract.json', dict(config=config, manifests=manifests, batches=contracts,
         seconds=time.perf_counter() - begin, waveform_scaling='float32, no x32768',
         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
         source_features='full clean source multiplied by recorded gain before fbank; concatenated by (start,speaker)',
         framing='10ms shift, 25ms window, dither=0, snip_edges=False; Lhotse may adjust <=2 boundary frames',
         heat='K2SurtDataset assigns first available channel by start time; pinned compute_heat_loss retains LOG_EPSILON addition and > length padding convention',
         storage='numpy_files replaces lilcom compression only; extractor and dataset unchanged'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Real feature work is Kaggle-only')
    records = json.loads((args.run / 'selection.json').read_text())
    assert_disjoint(records)
    for row in records:
        if file_hash(row['audio']) != row['file_sha256']:
            raise RuntimeError('Selected audio changed')
    tokenizer(args.run, records)
    features(args.run, records)


if __name__ == '__main__':
    main()
