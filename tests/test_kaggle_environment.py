"""Environment guard regressions; no downloads, GPU allocation, or pip installs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binary_import_failure_is_reported(monkeypatch):
    module = script('check_environment')
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda _: '2.1.3')
    def broken(_):
        raise OSError('binary ABI mismatch')
    monkeypatch.setattr(module.importlib, 'import_module', broken)
    result = module._package('numpy', check_import=True)
    assert result['status'] == 'broken'
    assert 'ABI mismatch' in result['error']


def test_all_visible_gpus_are_recorded(monkeypatch):
    module = script('check_environment')
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2,
        get_device_properties=lambda i: SimpleNamespace(name=f'T4-{i}', total_memory=15360 * 1024**2, major=7),
        get_device_capability=lambda i: (7, 5))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, __version__='2.11.0+cu128',
                                                           version=SimpleNamespace(cuda='12.8')))
    result = module._gpu_report()
    assert result['count'] == 2
    assert [d['index'] for d in result['devices']] == [0, 1]
    assert all(d['memory_mib'] == 15360 and not d['native_bfloat16'] for d in result['devices'])


def test_broken_torch_report_does_not_crash(monkeypatch):
    module = script('check_environment')
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert not module._gpu_report()['available']


def test_require_gpu_is_an_error_and_missing_output_dirs_are_usable(monkeypatch, tmp_path):
    module = script('check_environment')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, '_gpu_report', lambda: {'available': False, 'count': 0})
    monkeypatch.setattr(module, '_package', lambda *a, **kw: {'status': 'installed', 'version': '1'})
    report, ok = module.environment_report(require_gpu=True)
    assert not ok and 'CUDA GPU required but unavailable' in report['errors']
    assert all(report['writable'].values())
    _, ok = module.environment_report()
    assert ok


def test_setup_passes_existing_versions_as_constraints(monkeypatch, tmp_path):
    module = script('setup_kaggle')
    versions = {'torch': '2.11.0+cu128', 'numpy': '2.1.3'}
    monkeypatch.setattr(module, 'installed_constraints', lambda: versions.copy())
    monkeypatch.setattr(sys, 'argv', ['setup_kaggle', '--artifacts', str(tmp_path)])
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kwargs: commands.append((cmd, kwargs)))
    assert module.main() == 0
    assert (tmp_path / 'installed-stack-constraints.txt').read_text() == 'torch==2.11.0+cu128\nnumpy==2.1.3\n'
    cmd, options = commands[0]
    assert cmd[:3] == [sys.executable, '-m', 'pip']
    assert '--constraint' in cmd and '--only-binary=:all:' in cmd
    assert options['check'] is True


def test_setup_detects_unexpected_stack_changes(monkeypatch, tmp_path):
    module = script('setup_kaggle')
    versions = iter([{'numpy': '2.1.3'}, {'numpy': '1.26.4'}])
    monkeypatch.setattr(module, 'installed_constraints', lambda: next(versions))
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', ['setup_kaggle', '--artifacts', str(tmp_path)])
    with pytest.raises(RuntimeError, match='Protected packages changed'):
        module.main()


def test_all_notebooks_save_environment_and_use_guarded_setup():
    for path in [ROOT / 'english-streaming-asr.ipynb', *sorted((ROOT / 'notebooks').glob('*.ipynb'))]:
        source = ''.join(json.loads(path.read_text())['cells'][1]['source'])
        assert 'scripts/setup_kaggle.py' in source
        assert 'scripts/check_environment.py' in source
        assert 'environment.json' in source and 'pip-freeze.txt' in source
