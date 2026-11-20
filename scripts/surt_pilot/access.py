"""Kaggle-only MDC access probe. Requests links; NEVER follows asset URLs."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

from pipeline import CONFIG, dump, read


def clean_key(value):
    # The user's successful Kaggle probe removed a pasted zero-width joiner.
    # Strip ONLY that observed artifact; reject other non-ASCII/control characters.
    key = value.replace('\u200d', '').strip()
    if not key or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError('Invalid credential formatting; check the Kaggle Secret without printing it')
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an Authorization header to another location.
        return None


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
    def __init__(self, key, limits):
        self.key = key
        self.limits = limits
        self.received = 0
        self.requests = 0
        self.opener = urllib.request.build_opener(NoRedirect())

    def call(self, endpoint, authenticated=False):
        if self.requests >= self.limits['requests']:
            raise ValueError('API request cap exceeded')
        self.requests += 1
        headers = {'Accept': 'application/json'}
        if authenticated:
            headers['Authorization'] = 'Bearer ' + self.key
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(endpoint, data=b'{}' if authenticated else None,
                                         headers=headers, method='POST' if authenticated else 'GET')
        try:
            with self.opener.open(request, timeout=self.limits['timeout_seconds']) as response:
                remaining = min(self.limits['response_bytes'], self.limits['total_bytes'] - self.received)
                blocks = []
                while remaining > 0:
                    block = response.read(min(65536, remaining))
                    if not block:
                        payload = json.loads(b''.join(blocks))
                        if not isinstance(payload, dict):
                            raise ValueError('Expected JSON object')
                        return dict(http_status=response.status, **safe_details(payload))
                    blocks.append(block)
                    remaining -= len(block)
                    self.received += len(block)
                # Do not read even one byte beyond the control-plane ceiling.
                raise ValueError('API response/total byte cap reached')
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()  # Do not print or persist server error bodies.
            return dict(http_status=status)
        except (urllib.error.URLError, TimeoutError, OSError):
            return dict(http_status=None, error='network_or_timeout; details omitted to protect secrets')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
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
                  audio_downloaded=False, asset_requests=0, gate_2='OPEN', endpoints={})
    base = settings['base_url'] + '/datasets/' + config['source']['dataset_id']
    try:
        report['endpoints']['details'] = probe.call(base)
        if report['endpoints']['details'].get('http_status') == 429:
            raise ValueError('Public API rate limited; no further requests')
        full = probe.call(base + '/download', authenticated=True)
        report['endpoints']['download'] = full
        report['authenticated_access_confirmed'] = full.get('http_status') == 200 and full.get('presigned_url_returned') is True
        expected = read(CONFIG.parent / 'user-access-evidence-v1.json')
        report['archive_identity_matches_reviewed_api_metadata'] = identity_matches(full, expected)
        report['whole_archive_checksum_verified'] = False
        if report['authenticated_access_confirmed'] and not report['archive_identity_matches_reviewed_api_metadata']:
            report['error'] = 'archive_identity_changed_or_incomplete; preserve evidence and review source version'
        # No retry, no asset GET. 401/403/429 must not trigger another auth request.
        if report['authenticated_access_confirmed'] and 'error' not in report:
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
        report.update(control_plane_bytes=probe.received, api_requests=probe.requests)
        dump(report_path, report)
    full_status = report['endpoints'].get('download', {}).get('http_status')
    sample_status = report['endpoints'].get('sample_download', {}).get('http_status')
    print(f'بررسی API ثبت شد: download HTTP={full_status}، sample HTTP={sample_status}؛ صوت دریافت نشد، کلید و URL موقت ذخیره نشدند؛ gate داده باز است.')
    if not report['authenticated_access_confirmed'] or 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
