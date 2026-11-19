"""Offline mini-gate policy tests: no audio, models, installations or GPU."""
import ast
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('mini_' + name, ROOT / 'scripts/surt_mini' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contracts = load('contracts')


@pytest.fixture
def mini_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, 'contracts', contracts)
    return types.SimpleNamespace(fetch=load('fetch'), prepare=load('prepare'),
                                 transfer=load('transfer'), archive=load('archive'))


def test_normalization_numbers_latin_zwnj():
    text = '  كِتاب يِك می\u200cرود؛ OpenAI! ۱۲.۵ / ٣٠  '
    normalized = contracts.normalize(text)
    assert normalized == 'کتاب یک می\u200cرود openai 12 5 30'
    assert contracts.normalize(normalized) == normalized
    assert contracts.normalize('\u202eسلام\u202c') == 'سلام'
    assert contracts.normalize('!،') == ''


def records():
    return [dict(split=split, speaker_hash=f'speaker-{index}', source_id=f'clip-{index}',
                 original_sha256=f'original-{index}', waveform_sha256=f'wave-{index}')
            for index, split in enumerate(['train'] * 4 + ['validation'] * 2 + ['test'] * 2)]


@pytest.mark.parametrize('key', ['speaker_hash', 'source_id', 'original_sha256', 'waveform_sha256'])
def test_cross_split_duplicates_fail(key):
    rows = records()
    contracts.assert_disjoint(rows)
    rows[-1][key] = rows[0][key]
    with pytest.raises(ValueError, match=key):
        contracts.assert_disjoint(rows)


@pytest.mark.parametrize('speaker', [None, '', ' ', 42])
def test_missing_speaker_fails(speaker):
    with pytest.raises(ValueError, match='client_id'):
        contracts.identity(dict(client_id=speaker, sentence='سلام', path='clip'))


def test_persian_scoring_swapped_streams_and_insertions():
    result = contracts.score(['سلام دنیا', 'يك کتاب'], ['یک كتاب', 'سلام دنیا اضافه'])
    assert result['assignment'] == [1, 0]
    assert result['insertions'] == 1 and result['errors'] == 1 and result['words'] == 4
    assert contracts.score(['سلام', 'دنیا'], ['', ''])['deletions'] == 2


def test_evidence_writes_never_overwrite(tmp_path):
    path = tmp_path / 'evidence.json'
    contracts.dump(path, dict(status='original'))
    with pytest.raises(FileExistsError):
        contracts.dump(path, dict(status='replacement'))
    assert json.loads(path.read_text()) == dict(status='original')


class Pieces:
    def __init__(self, pieces, unknown, pad=-1, controls=()):
        self.pieces, self.unknown, self.pad, self.controls = pieces, unknown, pad, controls

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, index):
        return self.pieces[index]

    def piece_to_id(self, piece):
        return self.pieces.index(piece) if piece in self.pieces else self.unknown

    def pad_id(self):
        return self.pad

    def unk_id(self):
        return self.unknown

    def is_control(self, index):
        return index in self.controls

    def is_unused(self, index):
        return False


def test_semantic_mapping_equal_sizes_different_meanings():
    old = Pieces(['<blk>', '▁A', '<unk>', 'x', '▁', '<s>'], 2, controls=(5,))
    new = Pieces(['<blk>', '<unk>', 'س', '▁', 'x', '<s>'], 1, 0, controls=(5,))
    mapping = contracts.token_mapping(old, new)
    assert [(row['old_id'], row['new_id']) for row in mapping] == [(0, 0), (2, 1), (4, 3), (3, 4)]
    assert all(row['new_id'] != 2 for row in mapping)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    @property
    def shape(self):
        return self.value.shape

    def is_floating_point(self):
        return True

    def __getitem__(self, index):
        return Tensor(self.value[index:index + 1])

    def copy_(self, other):
        self.value[...] = other.value


def test_transfer_remaps_every_vocab_projection_and_keeps_fresh_rows(mini_modules, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(isfinite=lambda tensor: np.isfinite(tensor.value)))
    old = Pieces(['<blk>', '▁A', '<unk>', 'x', '▁', 'y'], 2)
    new = Pieces(['<blk>', '<unk>', 'س', '▁', 'x', 'a'], 1, 0)
    source = {name: Tensor(np.arange(12).reshape(6, 2)) for name in contracts.VOCAB_TENSORS}
    destination = {name: Tensor(np.full((6, 2), -9)) for name in source}
    source['encoder.weight'] = Tensor([[17, 18], [19, 20]])
    destination['encoder.weight'] = Tensor(np.zeros((2, 2)))

    class Model:
        def state_dict(self):
            return destination

        def load_state_dict(self, state, strict):
            assert strict and state is destination

    report = mini_modules.transfer.transfer(Model(), source, old, new, tmp_path / 'transfer.json')
    for name in contracts.VOCAB_TENSORS:
        assert destination[name].value[1].tolist() == [4, 5]
        assert destination[name].value[2].tolist() == [-9, -9]
        assert destination[name].value[3].tolist() == [8, 9]
    assert destination['encoder.weight'].value.tolist() == [[17, 18], [19, 20]]
    assert len(report['tensors']) == len(source)
    source['encoder.weight'] = Tensor(np.zeros((6, 2)))
    destination['encoder.weight'] = Tensor(np.zeros((6, 2)))
    with pytest.raises(RuntimeError, match='Unclassified'):
        mini_modules.transfer.transfer(Model(), source, old, new, tmp_path / 'bad.json')


def test_download_budget_and_cache_are_bounded(mini_modules, monkeypatch, tmp_path):
    fetch = mini_modules.fetch
    budget = fetch.Budget(tmp_path / 'cache')
    requests = []

    def open_url(url, timeout):
        requests.append(url)
        return io.BytesIO(b'12345')

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', open_url)
    assert budget.get('https://example/asset', 6, 'one') == b'12345'
    assert budget.get('https://example/asset?new_signature', 6, 'one') == b'12345'
    assert len(requests) == 1 and budget.bytes == 5
    with pytest.raises(RuntimeError, match='byte budget'):
        budget.get('https://example/too-large', 4, 'two')
    assert not (tmp_path / 'cache/two').exists()
    budget.begin -= contracts.PINS['limits']['seconds']
    with pytest.raises(RuntimeError, match='wall-time'):
        budget.get('https://example/deadline', 10)


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_metadata_transient_http_retries_are_bounded(mini_modules, monkeypatch, tmp_path, status):
    fetch = mini_modules.fetch
    budget = fetch.Budget(tmp_path / 'cache')
    responses = [status, status, b'valid metadata']
    waits, bodies = [], []

    def open_url(url, timeout):
        response = responses.pop(0)
        if isinstance(response, int):
            body = io.BytesIO(b'error body is not downloaded')
            bodies.append(body)
            raise fetch.urllib.error.HTTPError(url, response, 'upstream failure', {}, body)
        return io.BytesIO(response)

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(fetch.time, 'sleep', waits.append)
    assert budget.get('https://example/rows', 100) == b'valid metadata'
    assert waits == [1, 2] and budget.http_requests == 3
    assert budget.bytes == len(b'valid metadata') and all(body.closed for body in bodies)
    assert len(budget.report()['http_failures']) == 2


@pytest.mark.parametrize('status, cache_key, expected', [(502, None, 4), (429, None, 4),
                                                       (400, None, 1), (403, None, 1),
                                                       (404, None, 1), (502, 'audio', 4)])
def test_http_failure_stops_without_fallback(mini_modules, monkeypatch, tmp_path, status, cache_key, expected):
    fetch = mini_modules.fetch
    budget = fetch.Budget(tmp_path / 'cache')
    urls, waits = [], []

    def open_url(url, timeout):
        urls.append(url)
        raise fetch.urllib.error.HTTPError(url, status, 'failure', {}, io.BytesIO(b'error'))

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(fetch.time, 'sleep', waits.append)
    with pytest.raises(RuntimeError, match=f'HTTP {status}.*new RUN_ID'):
        budget.get('https://example/resource', 100, cache_key)
    assert urls == ['https://example/resource'] * expected
    assert budget.http_requests == expected and budget.bytes == 0
    assert waits == ([1, 2, 4] if expected == 4 else [])
    assert not list((tmp_path / 'cache').iterdir())


@pytest.mark.parametrize('kind', ['timeout', 'connection', 'urlerror'])
def test_network_error_then_success(mini_modules, monkeypatch, tmp_path, capsys, kind):
    fetch = mini_modules.fetch
    budget = fetch.Budget(tmp_path / 'cache')
    errors = {'timeout': TimeoutError('read timed out'),
              'connection': ConnectionResetError('connection reset'),
              'urlerror': fetch.urllib.error.URLError(TimeoutError('read timed out'))}
    waits, timeouts = [], []

    def open_url(url, timeout):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            raise errors[kind]
        return io.BytesIO(b'complete')

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(fetch.time, 'sleep', waits.append)
    assert budget.get('https://example/rows', 100) == b'complete'
    assert waits == [1] and budget.http_requests == 2
    assert all(0 < timeout <= 60 for timeout in timeouts)
    failure = budget.report()['http_failures'][0]
    assert failure['url'] == 'https://example/rows'
    assert failure['exception'] == type(errors[kind]).__name__
    assert failure['attempt'] == 1 and failure['retry']
    assert failure['duration_seconds'] >= 0
    assert json.dumps(failure) in capsys.readouterr().out


@pytest.mark.parametrize('failure', ['timeout', 'short_response'])
def test_partial_response_is_not_cached(mini_modules, monkeypatch, tmp_path, failure):
    fetch = mini_modules.fetch
    budget = fetch.Budget(tmp_path / 'cache')
    calls = []

    class Partial(io.BytesIO):
        headers = {'Content-Length': '100'}

        def read(self, size):
            if failure == 'timeout' and self.tell():
                raise TimeoutError('read timed out after partial data')
            return super().read(size)

    def open_url(url, timeout):
        calls.append(url)
        assert not (budget.cache / 'audio').exists()
        return Partial(b'partial') if len(calls) == 1 else io.BytesIO(b'complete')

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(fetch.time, 'sleep', lambda seconds: None)
    assert budget.get('https://example/audio', 1000, 'audio') == b'complete'
    assert len(calls) == 2
    assert (budget.cache / 'audio').read_bytes() == b'complete'
    assert budget.bytes == len(b'partialcomplete')


def test_request_timeout_is_capped_by_remaining_budget(mini_modules, monkeypatch, tmp_path):
    fetch = mini_modules.fetch
    now = [100.0]
    monkeypatch.setattr(fetch.time, 'monotonic', lambda: now[0])
    budget = fetch.Budget(tmp_path / 'cache')
    now[0] += contracts.PINS['limits']['seconds'] - 0.5
    timeouts, waits = [], []

    def open_url(url, timeout):
        timeouts.append(timeout)
        now[0] += timeout
        raise TimeoutError('remaining budget elapsed')

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(fetch.time, 'sleep', waits.append)
    with pytest.raises(RuntimeError, match='wall-time budget'):
        budget.get('https://example/rows', 100)
    assert timeouts == [0.5] and waits == []
    assert budget.http_requests == 1
    assert not budget.http_failures[0]['retry']


def test_retry_preserves_original_deadline(mini_modules, monkeypatch, tmp_path):
    fetch = mini_modules.fetch
    budget = fetch.Budget(tmp_path / 'cache')

    def fail(url, timeout):
        raise fetch.urllib.error.HTTPError(url, 502, 'failure', {}, io.BytesIO())

    def advance(seconds):
        budget.begin -= contracts.PINS['limits']['seconds']

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(fetch.time, 'sleep', advance)
    with pytest.raises(RuntimeError, match='wall-time budget'):
        budget.get('https://example/rows', 100)
    assert budget.http_requests == 1


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('root', ['assets', 'cached-assets'])
def test_viewer_asset_preserves_pinned_path(mini_modules, split, root):
    fetch = mini_modules.fetch
    path = (f"/{root}/{contracts.PINS['dataset_repo']}/--/{contracts.PINS['dataset_revision']}"
            f"/--/{contracts.PINS['config']}/{split}/100/audio/clip.mp3")
    parsed = fetch.viewer_asset('https://datasets-server.huggingface.co' + path + '?Signature=secret', split, 100)
    assert parsed.path == path
    assert parsed.query == 'Signature=secret'


@pytest.mark.parametrize('field', ['scheme', 'hostname', 'revision', 'config', 'split', 'row'])
@pytest.mark.parametrize('root', ['assets', 'cached-assets'])
def test_viewer_asset_mismatch_diagnostics(mini_modules, field, root):
    fetch = mini_modules.fetch
    scheme = 'http' if field == 'scheme' else 'https'
    hostname = 'other.example' if field == 'hostname' else 'datasets-server.huggingface.co'
    revision = 'different-revision' if field == 'revision' else contracts.PINS['dataset_revision']
    config = 'other' if field == 'config' else contracts.PINS['config']
    split = 'test' if field == 'split' else 'train'
    row = 101 if field == 'row' else 100
    path = f"/{root}/{contracts.PINS['dataset_repo']}/--/{revision}/--/{config}/{split}/{row}/audio/clip.mp3"
    url = f'{scheme}://{hostname}{path}?Signature=secret#private-fragment'
    with pytest.raises(RuntimeError, match='Viewer asset revision/path mismatch: ') as caught:
        fetch.viewer_asset(url, 'train', 100)
    message = str(caught.value)
    details = json.loads(message.split(': ', 1)[1])
    assert details['mismatches'] == [field if field in ('scheme', 'hostname') else 'path']
    assert details['actual_path'] == path
    assert details['actual_hostname'] == hostname
    assert details['original_split'] == 'train' and details['row_index'] == 100
    assert contracts.PINS['dataset_revision'] in details['expected_path_prefix']
    assert len(details['expected_path_prefixes']) == 2
    assert 'Signature' not in message and 'secret' not in message and 'private-fragment' not in message


def test_viewer_asset_accepts_kaggle_cached_asset_regression(mini_modules):
    # Exact routing fields from the failed mini-20261119-c run, not a live request.
    url = ('https://datasets-server.huggingface.co/cached-assets/'
           'aliyzd95/common_voice_22_0_fa/--/5dc7ac366099292d4fbf079c99d58c1b2fba602d/'
           '--/default/train/14/audio/audio.mp3')
    assert mini_modules.fetch.viewer_asset(url, 'train', 14).geturl() == url


@pytest.mark.parametrize('root', ['other-assets', 'cached-assets-extra', 'assets-extra'])
def test_viewer_asset_rejects_unknown_root(mini_modules, root):
    path = (f"/{root}/{contracts.PINS['dataset_repo']}/--/{contracts.PINS['dataset_revision']}"
            f"/--/{contracts.PINS['config']}/train/14/audio/audio.mp3")
    with pytest.raises(RuntimeError, match='Viewer asset revision/path mismatch'):
        mini_modules.fetch.viewer_asset('https://datasets-server.huggingface.co' + path, 'train', 14)


def test_repository_change_stops_before_audio(mini_modules):
    budget = types.SimpleNamespace(get=lambda url, limit: json.dumps(dict(sha='other', gated=False)).encode())
    with pytest.raises(RuntimeError, match='Repository changed'):
        mini_modules.fetch.repository(budget)


def test_total_network_and_persistent_cache_limits(mini_modules, monkeypatch, tmp_path):
    fetch = mini_modules.fetch
    limits = dict(contracts.PINS['limits'], network_bytes=3, cache_bytes=4)
    monkeypatch.setattr(fetch, 'PINS', dict(contracts.PINS, limits=limits))
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', lambda url, timeout: io.BytesIO(b'12345'))
    budget = fetch.Budget(tmp_path / 'cache')
    with pytest.raises(RuntimeError, match='byte budget'):
        budget.get('https://example/large', 10)
    assert budget.bytes == 3
    (tmp_path / 'cache/existing').write_bytes(b'12345')
    with pytest.raises(RuntimeError, match='cache budget'):
        budget.get('https://example/another', 10)


def test_fetch_refuses_mac_without_importing_audio_stack(mini_modules, monkeypatch, tmp_path):
    monkeypatch.setattr(mini_modules.fetch.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(sys, 'argv', ['fetch.py', str(tmp_path / 'run'), str(tmp_path / 'cache')])
    with pytest.raises(RuntimeError, match='only in Kaggle'):
        mini_modules.fetch.main()
    assert not (tmp_path / 'run').exists()


def test_tokenizer_receives_train_only(mini_modules, monkeypatch, tmp_path):
    selected = [dict(split=split, source_id=str(index), text=text) for index, (split, text) in enumerate([
        ('train', 'سلام'), ('train', 'کتاب'), ('validation', 'محرمانه'), ('test', 'آزمون')])]
    received = {}

    def train(**kwargs):
        received.update(kwargs)
        received['text'] = Path(kwargs['input']).read_text()

    class Processor(Pieces):
        def __init__(self, model_file):
            super().__init__(['<blk>', '<unk>', 'سلام', 'کتاب'], 1, 0)

        def encode(self, text, out_type):
            return [self.piece_to_id(text)]

        def decode(self, tokens):
            return self.id_to_piece(tokens[0])

    fake = types.SimpleNamespace(SentencePieceTrainer=types.SimpleNamespace(train=train),
                                 SentencePieceProcessor=Processor)
    monkeypatch.setitem(sys.modules, 'sentencepiece', fake)
    mini_modules.prepare.tokenizer(tmp_path, selected)
    assert received['text'] == 'سلام\nکتاب\n'
    assert received['normalization_rule_name'] == 'identity'
    assert not received['hard_vocab_limit']
    report = json.loads((tmp_path / 'tokenizer-report.json').read_text())
    assert report['actual_vocab'] == 4
    assert report['splits']['validation']['unknown'] == 1
    assert not report['splits']['test']['rows'][0]['round_trip']


def test_archive_excludes_audio_weights_cache_secrets_and_preserves_previous(mini_modules, tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    for name, content in [('selection-partial.json', '[]'), ('secret.json', '{"token":"PRIVATE"}'),
                          ('private.wav', 'audio'), ('mini-one-update.pt', 'weights'), ('.env', 'PRIVATE')]:
        (run / name).write_text(content)
    (run / 'logs').mkdir()
    (run / 'logs/failure.log').write_text('Full diagnostic')
    first = mini_modules.archive.archive_run(run, tmp_path / 'missing-icefall')
    previous = first.read_bytes()
    second = mini_modules.archive.archive_run(run, tmp_path / 'missing-icefall')
    assert first != second and first.read_bytes() == previous
    with tarfile.open(second) as archive:
        names = archive.getnames()
        assert 'run/selection-partial.json' in names and 'run/logs/failure.log' in names
        assert not any(name.endswith(('.pt', '.wav', '.env', 'secret.json')) for name in names)
        assert json.load(archive.extractfile('review-status.json'))['gate_1b'].startswith('pending')


def test_runbook_cells_parse_and_source_update_precedes_download(tmp_path):
    text = (ROOT / 'docs/KAGGLE_SURT_MINI_PERSIAN_RUNBOOK_FA.md').read_text()
    cells = re.findall(r'```bash\n%%bash\n(.*?)```', text, re.S)
    assert len(cells) == 6
    for index, cell in enumerate(cells):
        script = tmp_path / f'cell-{index}.sh'
        script.write_text(cell)
        subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True)
    assert 'fetch --no-tags origin "$PROJECT_COMMIT"' in cells[0]
    assert '--no-overwrite-ignore --detach' in cells[0]
    assert 'fetch.py' not in cells[0] and 'fetch.py' in cells[2]
    assert not any('pip install' in cell for cell in cells)
    for path in (ROOT / 'scripts/surt_mini').glob('*.py'):
        ast.parse(path.read_text())
