"""Strict freeze audit and explicitly requested, resolver-checked grpcio repair.

No imports of Torch/CUDA and no installation during normal validation.
"""
import argparse
import importlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / 'configs/surt_base/requirements-kaggle.txt'


def pins():
    return dict(line.split('==', 1) for line in REQUIREMENTS.read_text().splitlines()
                if line.strip() and not line.startswith('#'))


def details(name, expected):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return dict(expected=expected, actual=None, critical=name != 'grpcio',
                    source='not installed', dependents=[])
    # Ordinary index installs do not preserve the index URL. Do not expose direct
    # URLs, which may contain credentials; retain only whether one was recorded.
    source = dict(installer=(dist.read_text('INSTALLER') or 'unknown').strip(),
                  location=str(dist.locate_file('')),
                  direct_url_recorded=dist.read_text('direct_url.json') is not None,
                  index='not identifiable from standard installed metadata')
    from packaging.requirements import Requirement
    dependents = []
    for parent in metadata.distributions():
        for requirement in parent.requires or []:
            req = Requirement(requirement)
            if req.name.lower().replace('_', '-') == name.lower().replace('_', '-'):
                dependents.append(dict(package=parent.metadata['Name'], requirement=requirement))
    return dict(expected=expected, actual=dist.version, source=source,
                critical=name != 'grpcio', dependents=dependents,
                role='TensorBoard transitive dependency; no direct SURT math role' if name == 'grpcio'
                else 'protected execution dependency or exact experiment freeze')


def audit(expected=None):
    expected = pins() if expected is None else expected
    packages = {name: details(name, version) for name, version in expected.items()}
    drift = [name for name, row in packages.items() if row['actual'] != row['expected']]
    return dict(python=sys.version, executable=sys.executable, packages=packages, drift=drift,
                policy='exact versions for every pinned distribution; no automatic upgrades')


def require(report):
    if report['drift']:
        lines = []
        for name in report['drift']:
            row = report['packages'][name]
            remediation = ('Run dependencies.py --repair-grpcio --report <new-log-path>; '
                           'resolver must change only grpcio, then pip check and both runtimes'
                           if name == 'grpcio' else
                           'Preserve freeze/install logs; restore this exact pin through a reviewed constrained resolver plan')
            lines.append(f'{name}: expected {row["expected"]}, actual {row["actual"]}; '
                         f'critical={row["critical"]}; source={row["source"]}; remediation={remediation}')
        raise RuntimeError('Environment drift (strict pinning retained):\n' + '\n'.join(lines))


def snapshot():
    return {dist.metadata['Name'].lower().replace('_', '-'): dist.version
            for dist in metadata.distributions()}


def check_plan(plan, expected):
    installs = plan.get('install', [])
    for item in installs:
        package = item['metadata']
        if package['name'].lower().replace('_', '-') != 'grpcio' or package['version'] != expected:
            raise RuntimeError('Refusing resolver plan: changes beyond pinned grpcio')


def repair_grpcio(report):
    if sys.prefix == sys.base_prefix or sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError('Repair requires the isolated Python 3.12.10 venv')
    # Resolve normally with full experiment constraints; --no-deps is used only
    # after verifying this exact plan and checking the installed dependency graph.
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    before = snapshot()
    expected = pins()['grpcio']
    from packaging.requirements import Requirement
    for dependent in report['before']['packages']['grpcio']['dependents']:
        requirement = Requirement(dependent['requirement'])
        if (requirement.marker is None or requirement.marker.evaluate()) and not requirement.specifier.contains(expected):
            raise RuntimeError(f'Pinned grpcio conflicts with installed dependency: {dependent}')
    with tempfile.TemporaryDirectory(prefix='surt-grpcio-') as temporary:
        plan = Path(temporary) / 'plan.json'
        command = [sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                   '--constraint', str(REQUIREMENTS), 'grpcio==' + expected]
        subprocess.run([*command, '--dry-run', '--report', str(plan)], check=True)
        resolved = json.loads(plan.read_text())
        check_plan(resolved, expected)
        report['resolver_plan'] = resolved
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                        '--no-deps', 'grpcio==' + expected, '--report', str(plan)], check=True)
        report['installation_report'] = json.loads(plan.read_text())
    after = snapshot()
    report['before_versions'], report['after_versions'] = before, after
    changes = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if changes - {'grpcio'}:
        raise RuntimeError(f'Unexpected package changes: {changes}; preserve logs and review')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    grpc = importlib.import_module('grpc')
    if grpc.__version__ != expected:
        raise RuntimeError('grpc import/version mismatch after repair')
    report['grpc_import'] = dict(version=grpc.__version__, file=grpc.__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repair-grpcio', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    # Reserve append-only evidence before any mutation.
    with args.report.open('x') as stream:
        report = {}
        try:
            report['before'] = audit()
            if args.repair_grpcio:
                repair_grpcio(report)
            report['after'] = audit()
            require(report['after'])
        except Exception as error:
            report['failure'] = str(error)
            raise
        finally:
            json.dump(report, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
