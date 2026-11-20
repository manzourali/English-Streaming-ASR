"""Kaggle-only MDC access probe. Requests links; NEVER follows asset URLs."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

from pipeline import CONFIG, dump, read


def clean_key(value):
    # The user's successful Kaggle probe removed a pasted zero-width joiner.
    # Strip ONLY that observed artifact; reject other non-ASCII/control characters.
    key = value.replace('\u200d', '').strip()
    if not key or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError('Invalid credential formatting; check the Kaggle Secret without printing it')
    return key


def safe_details(value):
    """An allowlist, not redaction of a raw API response containing credentials."""
    result = {}
    size = value.get('sizeBytes')
    if isinstance(size, (int, str)) and not isinstance(size, bool):
        try:
            size = int(size)
            if size >= 0:
                result['size_bytes'] = size
        except ValueError:
            pass
    checksum = value.get('checksum')
    if isinstance(checksum, str) and re.fullmatch('[0-9a-fA-F]{64}', checksum):
        result['checksum_sha256'] = checksum.lower()
    filename = value.get('filename')
    if isinstance(filename, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', filename):
        result['filename'] = filename
    if value.get('contentType') in ('application/gzip', 'application/octet-stream', 'application/x-gzip'):
        result['content_type'] = value['contentType']
    result['presigned_url_returned'] = isinstance(value.get('downloadUrl'), str) and value['downloadUrl'].startswith('https://')
    return result


def identity_matches(details, evidence):
    # Compare API metadata with reviewed evidence, NOT a computed archive hash.
    return all(details.get(key) == evidence[expected] for key, expected in (
        ('filename', 'filename'), ('size_bytes', 'archive_size_bytes'),
        ('checksum_sha256', 'archive_sha256_reported_by_api')))


class Probe:
    def __init__(self, key, limits, session=None):
        self.key = key
        self.limits = limits
        self.received = 0
        self.requests = 0
        if session is None:
            # Control-plane requests run in the Kaggle notebook's host Python,
            # matching the user's successful requests.post, without changing the
            # pinned numerical environment or installing another dependency.
            import requests
            self.session = requests.Session()
            self.transport_version = requests.__version__
        else:
            self.session = session
            self.transport_version = 'injected-unit-fixture'

    def close(self):
        self.session.close()

    def call(self, endpoint, authenticated=False):
        if self.requests >= self.limits['requests']:
            raise ValueError('API request cap exceeded')
        remaining = min(self.limits['response_bytes'], self.limits['total_bytes'] - self.received)
        if remaining <= 0:
            raise ValueError('API total byte cap exhausted before request')
        self.requests += 1
        headers = {'Accept': 'application/json'}
        if authenticated:
            headers['Authorization'] = 'Bearer ' + self.key
        try:
            method = self.session.post if authenticated else self.session.get
            # Deliberately no data=, json= or Content-Type: same empty POST as
            # the user's successful Kaggle code. Redirects never forward a key.
            with method(endpoint, headers=headers, stream=True, allow_redirects=False,
                        timeout=self.limits['timeout_seconds']) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                info = dict(http_status=response.status_code,
                            response_kind=('json' if 'json' in content_type else
                                           'html' if 'html' in content_type else 'other'),
                            web_challenge=response.headers.get('cf-mitigated', '').lower() == 'challenge')
                # Do not consume or expose arbitrary error bodies.
                if response.status_code != 200:
                    return info
                blocks = []
                while remaining > 0:
                    block = response.raw.read(min(65536, remaining), decode_content=True)
                    if not block:
                        payload = json.loads(b''.join(blocks))
                        if not isinstance(payload, dict):
                            raise ValueError('Expected JSON object')
                        return dict(info, **safe_details(payload))
                    blocks.append(block)
                    remaining -= len(block)
                    self.received += len(block)
                raise ValueError('API response/total byte cap reached')
        except (ValueError, TimeoutError):
            raise
        except Exception as error:
            # Exception text can include request URLs/headers: retain only type.
            return dict(http_status=None, error='network_or_timeout', error_type=type(error).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--check-sample', action='store_true', help='Optional; distinct from dataset authentication')
    args = parser.parse_args()
    if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
        raise RuntimeError('Authenticated access probing is Kaggle-only')
    config = read(CONFIG)
    settings = config['api_probe']
    actual = subprocess.check_output(['git', '-C', str(CONFIG.parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != (args.run / 'project-commit.txt').read_text().strip():
        raise RuntimeError('Project SHA changed since run initialization')
    key = clean_key(os.environ.pop(settings['environment_variable'], ''))
    if not (args.run / 'review').is_dir() or not (args.run / 'private').is_dir():
        raise RuntimeError('Initialize a fresh pilot run first')
    # Reserve unique outputs before making quota-consuming requests.
    serial = str(time.time_ns())
    report_path = args.run / 'review' / f'access-probe-{serial}.json'
    def expired(signum, frame):
        raise TimeoutError('API probe time cap')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(settings['wall_seconds'])
    probe = Probe(key, settings)
    report = dict(dataset_id=config['source']['dataset_id'],
                  checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  user_reported_account_access=True, authenticated_access_confirmed=False,
                  audio_downloaded=False, asset_requests=0, gate_2='OPEN', endpoints={},
                  transport=dict(client='requests', version=probe.transport_version,
                                 python=sys.executable, python_version=platform.python_version(),
                                 authenticated_method='POST', request_body='none'),
                  sample_check_requested=args.check_sample)
    base = settings['base_url'] + '/datasets/' + config['source']['dataset_id']
    try:
        full = probe.call(base + '/download', authenticated=True)
        report['endpoints']['download'] = full
        report['authenticated_access_confirmed'] = full.get('http_status') == 200 and full.get('presigned_url_returned') is True
        expected = read(CONFIG.parent / 'user-access-evidence-v1.json')
        report['archive_identity_matches_reviewed_api_metadata'] = identity_matches(full, expected)
        report['whole_archive_checksum_verified'] = False
        if report['authenticated_access_confirmed'] and not report['archive_identity_matches_reviewed_api_metadata']:
            report['error'] = 'archive_identity_changed_or_incomplete; preserve evidence and review source version'
        # No retry, no asset GET. 401/403/429 must not trigger another auth request.
        if args.check_sample and report['authenticated_access_confirmed'] and 'error' not in report:
            sample = probe.call(base + '/sample-download', authenticated=True)
            report['endpoints']['sample_download'] = sample
            report['sample_link_available'] = sample.get('http_status') == 200 and sample.get('presigned_url_returned') is True
            report['sample_declared_within_import_cap'] = (
                sample.get('size_bytes') is not None and sample['size_bytes'] <= config['limits']['audio_import_bytes'])
        report['remaining_blocker'] = 'A download link does not establish a bounded usable subset with client_id, references and disjoint test evidence'
    except (ValueError, TimeoutError):
        report['error'] = 'response_schema_size_or_time_limit; raw response omitted'
    finally:
        signal.alarm(0)
        probe.close()
        report.update(control_plane_bytes=probe.received, api_requests=probe.requests)
        dump(report_path, report)
    full_status = report['endpoints'].get('download', {}).get('http_status')
    sample_status = report['endpoints'].get('sample_download', {}).get('http_status')
    print(f'نتیجه: download HTTP={full_status}؛ دسترسی دیتاست ' +
          ('تأیید شد.' if report['authenticated_access_confirmed'] else 'در این درخواست تأیید نشد؛ علت دقیق از status تنها مشخص نیست.'))
    if args.check_sample:
        print(f'sample HTTP={sample_status}؛ نتیجهٔ sample از تأیید دسترسی دیتاست جداست.')
    else:
        print('sample بررسی نشد؛ بررسی آن اختیاری است و این اجرا فقط دسترسی دیتاست را می‌سنجد.')
    if full_status == 403:
        full = report['endpoints']['download']
        print(f"تشخیص امن: نوع پاسخ={full.get('response_kind')}؛ web challenge={full.get('web_challenge')}؛ بدنه و header محرمانه چاپ نشدند.")
    if report.get('error'):
        print('مانع مستقل: ' + report['error'])
    print(f'صوت دریافت نشد؛ gate داده باز است. گزارش: {report_path}')
    if not report['authenticated_access_confirmed'] or 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
