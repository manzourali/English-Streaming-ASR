"""No installs, GPU imports or network: dependency policy and repair orchestration."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('surt_dependencies', ROOT / 'scripts/surt_base/dependencies.py')
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)


def test_grpcio_pin_and_smoke_setup_agree():
    assert deps.pins()['grpcio'] == '1.85.1'
    smoke = (ROOT / 'docs/KAGGLE_SURT_RUNBOOK.md').read_text()
    assert "'tensorboard==2.21.0' 'grpcio==1.85.1'" in smoke


def test_drift_diagnostics_remain_strict(monkeypatch):
    monkeypatch.setattr(deps, 'details', lambda name, expected: dict(
        expected=expected, actual='1.85.3', critical=False, source='installer=pip; index unknown'))
    report = deps.audit({'grpcio': '1.85.1'})
    with pytest.raises(RuntimeError, match='expected 1.85.1, actual 1.85.3; critical=False; source=.*repair-grpcio'):
        deps.require(report)
    report['packages']['grpcio']['actual'] = '1.85.1'
    report['drift'] = []
    deps.require(report)


def test_missing_distribution_is_reported(monkeypatch):
    def missing(name):
        raise deps.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(deps.metadata, 'distribution', missing)
    row = deps.details('torch', '2.3.1+cu121')
    assert row['actual'] is None and row['critical'] is True


def test_installer_and_reverse_dependency_evidence(monkeypatch):
    class Dist:
        version = '1.85.3'
        metadata = {'Name': 'tensorboard'}
        requires = ['grpcio>=1.74.0,<2.0']

        def read_text(self, name):
            return 'pip\n' if name == 'INSTALLER' else None

        def locate_file(self, path):
            return '/isolated/site-packages'

    monkeypatch.setattr(deps.metadata, 'distribution', lambda name: Dist())
    monkeypatch.setattr(deps.metadata, 'distributions', lambda: [Dist()])
    row = deps.details('grpcio', '1.85.1')
    assert row['source']['installer'] == 'pip'
    assert 'not identifiable' in row['source']['index']
    assert row['dependents'][0]['package'] == 'tensorboard'


@pytest.mark.parametrize('name, version', [('torch', '2.3.1'), ('grpcio', '1.85.3')])
def test_plan_rejects_unrelated_or_wrong_version(name, version):
    with pytest.raises(RuntimeError, match='Refusing resolver plan'):
        deps.check_plan({'install': [{'metadata': {'name': name, 'version': version}}]}, '1.85.1')


def test_repair_resolution_before_no_deps_and_both_pip_checks(monkeypatch):
    monkeypatch.setattr(deps.sys, 'prefix', '/isolated')
    monkeypatch.setattr(deps.sys, 'base_prefix', '/base')
    monkeypatch.setattr(deps.sys, 'version_info', (3, 12, 10))
    snapshots = iter([{'grpcio': '1.85.3', 'torch': '2.3.1+cu121'},
                      {'grpcio': '1.85.1', 'torch': '2.3.1+cu121'}])
    monkeypatch.setattr(deps, 'snapshot', lambda: next(snapshots))
    monkeypatch.setattr(deps.importlib, 'import_module', lambda name: types.SimpleNamespace(
        __version__='1.85.1', __file__='/isolated/grpc/__init__.py'))
    commands = []

    def run(command, check):
        commands.append(command)
        if '--report' in command:
            Path(command[command.index('--report') + 1]).write_text(json.dumps({
                'install': [{'metadata': {'name': 'grpcio', 'version': '1.85.1'}}]}))

    monkeypatch.setattr(deps.subprocess, 'run', run)
    report = {'before': {'packages': {'grpcio': {'dependents': [
        {'package': 'tensorboard', 'requirement': 'grpcio>=1.74,<2'}]}}}}
    deps.repair_grpcio(report)
    assert commands[0][-2:] == ['pip', 'check']
    assert '--dry-run' in commands[1] and '--constraint' in commands[1]
    assert '--no-deps' not in commands[1] and '--no-deps' in commands[2]
    assert commands[-1][-2:] == ['pip', 'check']
    assert report['grpc_import']['version'] == '1.85.1'


def test_repair_blocks_reverse_dependency_conflict(monkeypatch):
    monkeypatch.setattr(deps.sys, 'prefix', '/isolated')
    monkeypatch.setattr(deps.sys, 'base_prefix', '/base')
    monkeypatch.setattr(deps.sys, 'version_info', (3, 12, 10))
    monkeypatch.setattr(deps, 'snapshot', lambda: {})
    commands = []
    monkeypatch.setattr(deps.subprocess, 'run', lambda command, check: commands.append(command))
    report = {'before': {'packages': {'grpcio': {'dependents': [
        {'package': 'other', 'requirement': 'grpcio>=1.85.3'}]}}}}
    with pytest.raises(RuntimeError, match='conflicts with installed dependency'):
        deps.repair_grpcio(report)
    assert len(commands) == 1 and commands[0][-1] == 'check'


def test_failed_audit_persists_report_without_install(monkeypatch, tmp_path):
    report = {'drift': ['grpcio'], 'packages': {'grpcio': dict(
        expected='1.85.1', actual='1.85.3', critical=False, source='unknown')}}
    monkeypatch.setattr(deps, 'audit', lambda: report)
    path = tmp_path / 'diagnostics.txt'
    monkeypatch.setattr(sys, 'argv', ['dependencies.py', '--report', str(path)])
    with pytest.raises(RuntimeError, match='Environment drift'):
        deps.main()
    assert json.loads(path.read_text())['before']['packages']['grpcio']['actual'] == '1.85.3'
    with pytest.raises(FileExistsError):
        deps.main()
