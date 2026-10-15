#!/usr/bin/env python3
"""Install isolated Playwright/Chromium without altering the original venv.

Needs outbound HTTPS to PyPI and Playwright's browser CDN. Does not run sudo,
install system packages, disable TLS, or use the operator's browser profile.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.target).resolve()
    vendor = root / '_webwalker_browser_vendor'
    env = dict(os.environ)
    env['PLAYWRIGHT_BROWSERS_PATH'] = str(root / '_webwalker_browsers')
    env['PYTHONPATH'] = str(vendor) + os.pathsep + env.get('PYTHONPATH', '')
    with tempfile.TemporaryDirectory(prefix='dagbt-web-pip-') as temp:
        if not (vendor / 'playwright').is_dir():
            # A temporary pip wheel supplies pip even in uv-created venvs.
            with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=30) as response:
                metadata = json.load(response)
            wheel = next(r for r in metadata['urls'] if r['filename'].endswith('py3-none-any.whl'))
            with urllib.request.urlopen(wheel['url'], timeout=60) as response:
                raw = response.read()
            import hashlib
            if hashlib.sha256(raw).hexdigest() != wheel['digests']['sha256']:
                raise ValueError('Pip wheel hash mismatch')
            path = Path(temp) / wheel['filename']
            path.write_bytes(raw)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(temp)
            pip_env = {**env, 'PYTHONPATH': temp + os.pathsep + env['PYTHONPATH']}
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                            '--target', str(vendor), 'playwright==1.56.0'], env=pip_env, check=True)
        subprocess.run([sys.executable, '-m', 'playwright', 'install', 'chromium', '--only-shell'], env=env, check=True)
        subprocess.run([sys.executable, '-c',
            'from playwright.sync_api import sync_playwright\n'
            'with sync_playwright() as p:\n'
            ' b=p.chromium.launch(headless=True)\n'
            ' page=b.new_page();page.set_content("<p>Browser ready</p>")\n'
            ' assert page.inner_text("body")=="Browser ready"\n'
            ' b.close()\n'], env=env, check=True)
    print(json.dumps({'browser_ready': True, 'isolated_vendor': str(vendor),
                      'browser_cache': env['PLAYWRIGHT_BROWSERS_PATH'], 'system_packages_changed': False}))


if __name__ == '__main__':
    main()
