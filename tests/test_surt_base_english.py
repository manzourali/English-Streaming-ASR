"""CPU contracts only: these never execute a real model or claim English quality."""
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/surt_base'))
from common import errors, normalize, score, sha256, validate_records
import prepare


def test_error_counts_and_normalization():
    assert normalize("Don't, stop!") == ["DON'T", 'STOP']
    assert errors('A B C', 'A X C D') == dict(errors=2, substitutions=1,
                                           deletions=0, insertions=1, words=3)
    assert errors('A B', '')['deletions'] == 2
    assert errors('', 'EXTRA')['insertions'] == 1


def test_swapped_streams_and_clean_insertions():
    assert score(['ONE TWO', 'THREE'], ['THREE', 'ONE TWO'])['errors'] == 0
    result = score(['ONE TWO'], ['EXTRA', 'ONE TWO'])
    assert result['assignment'] == [1, 0]
    assert result['insertions'] == 1 and result['words'] == 2 and result['wer'] == 0.5
    with pytest.raises(ValueError):
        score([''], ['', ''])


def test_different_speakers_required():
    records = [dict(id=str(i), speaker_id='same', text='REAL TEXT', source='source',
                    license='CC BY 4.0', audio='a.wav') for i in range(2)]
    with pytest.raises(ValueError, match='different speakers'):
        validate_records(records)
    records[1]['speaker_id'] = 'different'
    validate_records(records)


def patched_fixture(tmp_path):
    recipe = tmp_path / 'egs/libricss/SURT/dprnn_zipformer'
    recipe.mkdir(parents=True)
    pins = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())
    for name in ('pretrained.py', 'asr_datamodule.py'):
        path = ROOT / 'tests/fixtures/surt_base' / name
        assert sha256(path) == pins['source_sha256'][name]
        shutil.copyfile(path, recipe / name)
    for name in ('surt-simple-sampler.patch', 'surt-base-pretrained.patch'):
        assert sha256(ROOT / 'docs/patches' / name) == pins['patch_sha256'][name]
        subprocess.run(['patch', '-p1', '-i', str(ROOT / 'docs/patches' / name)],
                       cwd=tmp_path, check=True, capture_output=True)
    for name in ('pretrained.py', 'asr_datamodule.py'):
        assert sha256(recipe / name) == pins['patched_source_sha256'][name]
    return ast.parse((recipe / 'pretrained.py').read_text()), recipe


def test_pinned_patch_and_channel_grouping(tmp_path):
    tree, recipe = patched_fixture(tmp_path)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    assert sum(ast.unparse(n.func) == 'torch.load' for n in calls) == 1
    assert sum(ast.unparse(n.func) == 'get_surt_model' for n in calls) == 1
    assert sum(ast.unparse(n.func) == 'decode_features' for n in calls) == 2
    assert ast.unparse(main.decorator_list[0]) == 'torch.inference_mode()'
    load = next(n for n in calls if ast.unparse(n.func) == 'model.load_state_dict')
    assert next(k.value.value for k in load.keywords if k.arg == 'strict') is True
    group = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_group_channels')
    group = copy.deepcopy(group)
    group.returns = None
    group.args.args[0].annotation = None
    namespace = dict(B=3, params=type('Params', (), {'num_channels': 2})())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[group], type_ignores=[])),
                 '<channel grouping>', 'exec'), namespace)
    assert namespace['_group_channels'](['a0', 'b0', 'm0', 'a1', 'b1', 'm1']) == [
        ['a0', 'a1'], ['b0', 'b1'], ['m0', 'm1']]
    source = (recipe / 'asr_datamodule.py').read_text()
    for method in ('valid_dataloaders', 'test_dataloaders'):
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == method)
        assert any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'SimpleCutSampler' for n in ast.walk(node))


def test_mixture_retains_waveforms_and_true_overlap():
    np = pytest.importorskip('numpy')
    # Arithmetic fixture only; not the runbook's speech input.
    waves = [np.full(16000 * seconds, 0.2 + i * 0.1, dtype=np.float32)
             for i, seconds in enumerate((1, 2))]
    wave, overlap = prepare.mix_waves(waves)
    assert len(wave) / 16000 == 2.5 and overlap / 16000 == 0.5
    assert wave[100] == pytest.approx(0.1) and wave[9000] == pytest.approx(0.25)
    assert wave[-100] == pytest.approx(0.15)
    with pytest.raises(ValueError, match='No real overlap'):
        prepare.mix_waves(waves, offset=40000)


def test_downloader_refuses_mac_before_network(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/surt_base/prepare.py'), str(tmp_path / 'out')],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and 'restricted to /kaggle/working' in proc.stderr


def test_failed_run_archive_retains_evidence_without_audio_or_weights(tmp_path):
    import tarfile
    base = tmp_path / 'run'
    (base / 'logs').mkdir(parents=True)
    (base / 'logs/failure.log').write_text('Traceback: test failure\n')
    (base / 'assets').mkdir()
    (base / 'assets/private.wav').write_bytes(b'private-audio-test')
    (base / 'assets/weight.pt').write_bytes(b'weight-test')
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/surt_base/archive.py'),
                           str(base), str(tmp_path / 'missing-icefall')],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    with tarfile.open(tmp_path / 'surt-base-1a-review.tar.gz') as archive:
        assert not any(n.endswith(('.wav', '.pt')) for n in archive.getnames())
        prefix = 'surt-base-1a-review/'
        status = json.load(archive.extractfile(prefix + 'review-status.json'))
        assert status['status'] == 'incomplete/failed'
        inventory = json.load(archive.extractfile(prefix + 'actual-files.json'))
        assert inventory[str(base / 'assets/weight.pt')]['sha256'] == sha256(base / 'assets/weight.pt')
        assert archive.extractfile(prefix + 'logs/failure.log').read() == b'Traceback: test failure\n'


@pytest.mark.parametrize('existing', [False, True])
def test_git_update_preserves_downloads_and_refuses_local_edits(tmp_path, existing):
    """Exercise cell 1 against a tiny local remote; no network or real assets."""
    import os
    import re
    guide = (ROOT / 'docs/KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md').read_text()
    block = guide.split('# After committing and pushing locally,')[1].split('ICEFALL_COMMIT=')[0]
    block = '# After committing and pushing locally,' + block
    remote = tmp_path / 'remote'
    remote.mkdir()

    def git(*args, cwd=remote):
        return subprocess.check_output(['git', '-c', 'user.name=Fixture', '-c',
                                       'user.email=fixture@example.invalid', *args],
                                      cwd=cwd, text=True, stderr=subprocess.PIPE).strip()

    git('init')
    (remote / '.gitignore').write_text('checkpoints/\n')
    (remote / 'version.txt').write_text('old')
    git('add', '.')
    git('commit', '-m', 'fixture one')
    project = tmp_path / 'project'
    if existing:
        git('clone', str(remote), str(project), cwd=tmp_path)
        (project / 'checkpoints').mkdir()
        (project / 'checkpoints/heavy.pt').write_bytes(b'existing downloaded weight')
    (remote / 'scripts/surt_base').mkdir(parents=True)
    (remote / 'scripts/surt_base/env.sh').write_text('# fixture\n')
    (remote / 'version.txt').write_text('new')
    git('add', '.')
    git('commit', '-m', 'fixture two')
    revision = git('rev-parse', 'HEAD')
    block = block.replace('PASTE_FULL_PUSHED_COMMIT_SHA_HERE', revision)
    block = block.replace('https://github.com/manzourali/English-Streaming-ASR.git', str(remote))
    assert 'checkout --no-overwrite-ignore --detach' in block
    assert not re.search(r'git clean|reset --hard|rm -rf', block)
    logs = tmp_path / 'logs'
    logs.mkdir()
    env = dict(os.environ, PROJECT=str(project), LOG_DIR=str(logs))

    def update():
        return subprocess.run(['bash', '-euo', 'pipefail', '-c', block],
                              env=env, capture_output=True, text=True)

    proc = update()
    assert proc.returncode == 0, proc.stderr
    assert (project / 'version.txt').read_text() == 'new'
    assert (logs / 'project-commit.txt').read_text().strip() == revision
    if existing:
        assert (project / 'checkpoints/heavy.pt').read_bytes() == b'existing downloaded weight'
    assert update().returncode == 0  # source update is repeatable
    (project / 'version.txt').write_text('my uncommitted edit')
    proc = update()
    assert proc.returncode != 0 and 'Tracked local edits' in proc.stdout
    assert (project / 'version.txt').read_text() == 'my uncommitted edit'


def test_numpy_scipy_repair_matches_fresh_environment_and_preflight():
    """Guard the observed pip-check regression in both environment paths."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    def requirements(name):
        return {r.name: str(r.specifier) for line in
                (ROOT / 'configs/surt_base' / name).read_text().splitlines()
                if line and not line.startswith('#') for r in [Requirement(line)]}

    fresh = requirements('requirements-kaggle.txt')
    repair = requirements('requirements-compat.txt')
    assert repair == {'numpy': '==1.26.4', 'scipy': '==1.13.1'}
    assert {name: fresh[name] for name in repair} == repair
    # SciPy 1.13 support table: NumPy >=1.22.4,<2.3; Python >=3.9,<3.13.
    assert '1.26.4' in SpecifierSet('>=1.22.4,<2.3')
    assert '3.12.10' in SpecifierSet('>=3.9,<3.13')
    # The observed SciPy 1.18.1 requirement excludes the chosen NumPy.
    assert '1.26.4' not in SpecifierSet('>=2.0.0,<2.8')
    observed = requirements('smoke-observed-freeze.txt')
    assert observed['scipy'] == '==1.18.1'  # historical evidence stays intact
    tree = ast.parse((ROOT / 'scripts/surt_base/runtime.py').read_text())
    preflight = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    expected = next(ast.literal_eval(n.value) for n in ast.walk(preflight)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                                                        t.id == 'expected' for t in n.targets))
    assert all(repair[name] == '==' + expected[name] for name in repair)
    guide = (ROOT / 'docs/KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md').read_text()
    existing = guide.split('### سلول 2')[1].split('### سلول 3')[0].split('else\n')[1].split('\nfi')[0]
    assert '--no-deps --only-binary=:all:' in existing
    assert 'requirements-compat.txt' in existing
    assert 'requirements-kaggle.txt' not in existing


def test_existing_decode_result_preserves_assets_and_directs_to_archive(tmp_path):
    import os
    base = tmp_path / 'run'
    assets = base / 'assets'
    assets.mkdir(parents=True)
    (assets / 'inputs.json').write_text(json.dumps([{'duration': 1}] * 3))
    (assets / 'fixture-weight.pt').write_bytes(b'existing fixture weight')
    output = base / 'results'
    output.mkdir()
    # Even incomplete prior evidence must survive; existence is not a success claim.
    (output / 'result.json').write_text('{"status": "incomplete fixture"}\n')
    (output / 'process.json').write_text('{"exit_code": 1}\n')
    before = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob('*') if p.is_file()}
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/surt_base/run.py')],
                          env=dict(os.environ, BASE=str(base)), capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'Decode was not started' in proc.stderr
    assert 'Run cell 5 (05-archive)' in proc.stderr
    assert 'Keep this session and all assets' in proc.stderr
    assert 'does not prove gate 1A passed' in proc.stderr
    assert 'fresh run directory/session' not in proc.stderr
    after = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob('*') if p.is_file()}
    assert after == before
