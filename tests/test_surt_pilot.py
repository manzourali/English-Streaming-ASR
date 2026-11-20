"""Stage-2 contract tests. Execute only on Kaggle; synthetic unit fixtures are not ASR evidence."""
import argparse
import copy
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch

if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
    raise RuntimeError('These tests are authorized only on Kaggle')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/surt_pilot'))
import pipeline as pilot
from access import Probe, clean_key, identity_matches, safe_details
from policy import normalize


class TextPolicyTests(unittest.TestCase):
    def test_edges_and_idempotence(self):
        cases = {
            '  كِي‌بُرد، يک!  ': 'کی‌برد یک',
            'API GPU OpenAI X123': 'API GPU OpenAI X123',
            '۱۲/۰۳/۱۴۰۳ و 12.5%': '12 03 1403 و 12 5',
            '\u200cالف\u200c \u200cب\u200c': 'الف ب',
            'می\u200c\u200cروم\tخانه': 'می\u200cروم خانه',
            '۱۲٣ ٤٥6': '123 456',
            'الف\u200fب\x00ج': 'الف ب ج',
            '--- ، ؟': '',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                actual = normalize(raw)
                self.assertEqual(actual, expected)
                self.assertEqual(normalize(actual), actual)

    def test_numeric_ambiguity_not_verbalized(self):
        self.assertEqual(normalize('۱۴۰۳'), '1403')
        self.assertEqual(normalize('C++ foo@example.com'), 'C foo example com')


class AccessProbeTests(unittest.TestCase):
    class Raw(io.BytesIO):
        def read(self, size=-1, decode_content=False):
            return super().read(size)

    class Response:
        def __init__(self, status, body, kind='application/json'):
            self.status_code = status
            self.headers = {'Content-Type': kind}
            self.raw = AccessProbeTests.Raw(body)
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    class Session:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def post(self, endpoint, **kwargs):
            self.calls.append((endpoint, kwargs))
            return self.response

        def close(self):
            pass

    def probe(self, response):
        session = self.Session(response)
        return Probe('unit-key', dict(requests=2, response_bytes=1024,
                     total_bytes=2048, timeout_seconds=30), session=session), session

    def test_matches_successful_empty_post_and_never_follows_redirects(self):
        response = self.Response(200, b'{"downloadUrl":"https://example.invalid/signed"}')
        probe, session = self.probe(response)
        result = probe.call('https://example.invalid/download', authenticated=True)
        kwargs = session.calls[0][1]
        self.assertNotIn('data', kwargs)
        self.assertNotIn('json', kwargs)
        self.assertNotIn('Content-Type', kwargs['headers'])
        self.assertEqual(kwargs['headers']['Accept'], 'application/json')
        self.assertTrue(kwargs['stream'])
        self.assertFalse(kwargs['allow_redirects'])
        self.assertTrue(result['presigned_url_returned'])
        self.assertTrue(response.closed)

    def test_403_html_is_reported_without_reading_error_body(self):
        response = self.Response(403, b'private server response', 'text/html')
        probe, session = self.probe(response)
        result = probe.call('https://example.invalid/download', authenticated=True)
        self.assertEqual(result['http_status'], 403)
        self.assertEqual(result['response_kind'], 'html')
        self.assertEqual(response.raw.tell(), 0)
        self.assertTrue(response.closed)
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn('private server response', json.dumps(result))

    def test_observed_zero_width_joiner_artifact_is_removed(self):
        self.assertEqual(clean_key(' \u200dunit\u200d-key\n'), 'unit-key')
        for value in ('', 'unit\nkey', 'unit\u200bkey', 'unit key'):
            with self.subTest(kind='invalid formatting'), self.assertRaises(ValueError):
                clean_key(value)

    def test_download_identity_is_not_assumed_from_success_status(self):
        evidence = dict(filename='corpus.tar.gz', archive_size_bytes=123,
                        archive_sha256_reported_by_api='a' * 64)
        details = dict(filename='corpus.tar.gz', size_bytes=123, checksum_sha256='a' * 64)
        self.assertTrue(identity_matches(details, evidence))
        self.assertFalse(identity_matches(dict(details, size_bytes=124), evidence))
        self.assertFalse(identity_matches(dict(filename='corpus.tar.gz'), evidence))

    def test_report_excludes_credentials_and_signed_urls(self):
        payload = dict(sizeBytes='1048576', filename='sample.tar.gz', checksum='a' * 64,
                       downloadUrl='https://example.invalid/private?signature=unit-test',
                       downloadToken='unit-secret', accessRecordId='unit-private',
                       apiKey='unit-secret', unrelated='must not persist')
        result = safe_details(payload)
        self.assertEqual(result['size_bytes'], 1048576)
        self.assertTrue(result['presigned_url_returned'])
        serialized = json.dumps(result)
        for forbidden in ('unit-secret', 'signature', 'accessRecordId', 'unrelated'):
            self.assertNotIn(forbidden, serialized)

    def test_untrusted_filename_and_size_are_omitted(self):
        result = safe_details(dict(sizeBytes='unknown', filename='../unsafe?secret=unit', checksum='bad'))
        self.assertNotIn('filename', result)
        self.assertNotIn('size_bytes', result)
        self.assertNotIn('checksum_sha256', result)

class SplitTests(unittest.TestCase):
    def test_seeded_assignment_and_speaker_group(self):
        config = pilot.read(pilot.CONFIG)
        assignments = {str(i): pilot.split_for(str(i), config) for i in range(100)}
        self.assertEqual(assignments, {str(i): pilot.split_for(str(i), config) for i in reversed(range(100))})
        self.assertEqual(set(assignments.values()), {'train', 'validation', 'test'})

    def test_cross_split_collision_rejected(self):
        for key in ('speaker_hash', 'source_id', 'original_sha256', 'waveform_sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.assert_disjoint([{key: 'same', 'split': 'train'}, {key: 'same', 'split': 'test'}])

    def test_prior_inspections_include_unselected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'inspection-ledger.json'
            path.write_text(json.dumps({'metadata': [{'source_id': 'inspected.mp3', 'speaker_hash': 'exposed'}],
                                         'attempts': [{'source_id': 'bad.mp3', 'original_sha256': 'h'}]}))
            exposed, missing = pilot.exposure([path])
            self.assertEqual(exposed['source_id'], {'inspected.mp3', 'bad.mp3'})
            self.assertEqual(exposed['original_sha256'], {'h'})
            self.assertEqual(missing, 1)

    def test_paths_cannot_escape_root(self):
        for value in ('../bad.mp3', '/tmp/bad.mp3', '.', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.safe_name(value)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        (self.run / 'private').mkdir(parents=True)
        (self.run / 'review').mkdir()
        self.config = pilot.read(pilot.CONFIG)
        self.config['limits']['clips'] = 12
        self.config['limits']['speakers'] = 12
        self.config['limits']['clips_per_speaker'] = 2
        self.config['limits']['duration_seconds'] = 100
        self.config['limits']['audio_import_bytes'] = 1200
        (self.run / 'private/config.json').write_text(json.dumps(self.config))
        inventory = self.root / 'configs/surt_pilot'
        inventory.mkdir(parents=True)
        # Empty pinned inventory in synthetic fixture; actual production pins remain unchanged.
        (inventory / 'exposure-inventory-v1.json').write_text(json.dumps({'files': []}))
        (inventory / 'user-prefix-exposure-v1.json').write_text(json.dumps(
            {'metadata': [], 'missing_exposure_rows': 0}))
        self.ledgers = self.root / 'extra-ledgers'
        self.ledgers.mkdir()
        self.forbidden_client = 'fixture-exposed'
        (self.ledgers / 'inspection-ledger.json').write_text(json.dumps({'metadata': [{
            'source_id': 'exposed.mp3', 'speaker_hash': pilot.hashed('cv-client:' + self.forbidden_client)}]}))
        self.metadata = self.root / 'metadata.jsonl'
        fixture = []
        for split in ('train', 'validation', 'test'):
            matching = [str(i) for i in range(1000) if pilot.split_for(pilot.hashed('cv-client:' + str(i)), self.config) == split][:4]
            for client in matching:
                fixture.append(dict(path=f'{client}.mp3', client_id=client, sentence=f'واژه {client}',
                                    locale='fa', validated=True, bytes=100, duration_metadata_seconds=2))
        fixture.append(dict(path='exposed.mp3', client_id=self.forbidden_client, sentence='مورد بررسی شده',
                            locale='fa', validated=True, bytes=100, duration_metadata_seconds=2))
        pilot.jsonl(self.metadata, fixture)
        self.receipt = self.root / 'receipt.json'
        self.receipt.write_text(json.dumps(dict(dataset_id=self.config['source']['dataset_id'],
            release=self.config['source']['release'], locale='fa', terms_accepted_at='unit-fixture',
            access_verified_at='unit-fixture', official_subset_route='unit-fixture', route_evidence='unit-fixture',
            subset_origin_checksum='unit-fixture', metadata_sha256=pilot.sha(self.metadata))))
        self.args = argparse.Namespace(run=self.run, metadata=self.metadata, receipt=self.receipt, ledgers=self.ledgers)

    def test_caps_exposure_and_frozen_inputs(self):
        with patch.object(pilot, 'ROOT', self.root):
            pilot.plan(self.args)
        selected = pilot.rows(self.run / 'private/selection.jsonl')
        self.assertLessEqual(len(selected), 12)
        self.assertLessEqual(sum(r['reserved_bytes'] for r in selected), 1200)
        self.assertNotIn('exposed.mp3', [r['source_id'] for r in selected])
        pilot.assert_disjoint(selected)
        pilot.locked(self.args)
        with (self.run / 'private/selection.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
            pilot.locked(self.args)

    def test_user_prefix_source_hash_is_excluded_without_plaintext_identifier(self):
        first = pilot.rows(self.metadata)[0]
        target = self.root / 'configs/surt_pilot/user-prefix-exposure-v1.json'
        target.write_text(json.dumps(dict(metadata=[dict(
            source_id_sha256=pilot.hashed(first['path']), speaker_hash='other-speaker')],
            missing_exposure_rows=4997)))
        with patch.object(pilot, 'ROOT', self.root):
            pilot.plan(self.args)
        selected = pilot.rows(self.run / 'private/selection.jsonl')
        self.assertNotIn(first['path'], {row['source_id'] for row in selected})
        lock = pilot.read(self.run / 'private/plan-lock.json')
        self.assertEqual(lock['missing_user_prefix_exposure_rows'], 4997)

    def test_missing_access_stops_before_selection(self):
        receipt = pilot.read(self.receipt)
        receipt['official_subset_route'] = None
        self.receipt.write_text(json.dumps(receipt))
        with patch.object(pilot, 'ROOT', self.root), self.assertRaisesRegex(ValueError, 'Access blocked'):
            pilot.plan(self.args)
        self.assertFalse((self.run / 'private/selection.jsonl').exists())

    def test_metadata_cap_before_parse(self):
        config = copy.deepcopy(self.config)
        config['limits']['metadata_bytes'] = 1
        (self.run / 'private/config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'Metadata byte cap'):
            pilot.plan(self.args)

    def test_tokenizer_never_fits_held_out_text(self):
        with patch.object(pilot, 'ROOT', self.root):
            pilot.plan(self.args)
        selected = pilot.rows(self.run / 'private/selection.jsonl')
        for row in selected:
            if row['split'] == 'test':
                row['text_normalized'] = 'ZXQHOLDOUT'
        manifest = self.run / 'private/manifest-v1.jsonl'
        pilot.jsonl(manifest, selected)
        pilot.dump(self.run / 'private/manifest-lock.json', dict(sha256=pilot.sha(manifest),
                   plan_sha256=pilot.sha(self.run / 'private/plan-lock.json')))
        pilot.tokenizer(self.args)
        tokenizer = self.run / 'private/tokenizer'
        self.assertNotIn('ZXQHOLDOUT', (tokenizer / 'train.txt').read_text())
        freeze = pilot.read(tokenizer / 'freeze.json')
        self.assertEqual(freeze['model_sha256'], pilot.sha(tokenizer / 'bpe.model'))
        self.assertEqual(set(freeze['train_sources']),
                         {r['source_id'] for r in selected if r['split'] == 'train'})
        report = pilot.read(self.run / 'review/tokenizer-summary.json')
        self.assertGreater(report['splits']['test']['unknown'], 0)
        self.assertFalse(report['test_used_for_tuning'])


if __name__ == '__main__':
    unittest.main()
