#!/usr/bin/env python3
"""Build a verified v4 addon; exclude private configs, credentials and indices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'e6677081cdfb33c01db3aaa274a497168f314f78'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(destination, wheel, validation=None):
    destination = Path(destination).resolve()
    name = 'dagbt_webwalker_v4_experiment'
    bundle = destination / name
    bundle.mkdir(parents=True, exist_ok=False)
    source_paths = [p for p in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=ROOT,
                                                      text=True).splitlines()
                    if p.endswith('.py') and p.split('/')[0] in ('dagbt', 'dagv2', 'dagresolve', 'package', 'vendor')]
    baseline = {}
    for path in source_paths:
        original = subprocess.check_output(['git', 'show', BASELINE + ':' + path], cwd=ROOT)
        expected = hashlib.sha256(original).hexdigest()
        if sha(ROOT / path) != expected:
            raise ValueError('Do not package changed baseline core: ' + path)
        baseline[path] = expected
    payload = [*source_paths, 'scripts/demo_residual_memory.py', 'scripts/demo_local_terminal.py',
               *['dagbt/' + p.name for p in (ROOT / 'dagbt').glob('webwalker_*.py')],
               'configs/webwalker-v4.json', 'scripts/run_webwalker.sh', 'scripts/webwalker_offline_smoke.py', 'scripts/install_webwalker_browser.py',
               'tests/test_webwalker.py', 'tests/test_webwalker_runner.py', 'tests/test_webwalker_v4.py',
               'tests/test_webwalker_access.py', 'tests/test_webwalker_reasoning.py',
               'tests/test_webwalker_reading.py', 'tests/test_webwalker_diagnose.py',
               'scripts/webwalker_diagnose.py',
               *['data/webwalkerqa/' + p.name for p in (ROOT / 'data/webwalkerqa').iterdir() if p.is_file()]]
    hashes = {}
    for path in sorted(payload):
        target = bundle / 'payload' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
        hashes[path] = sha(target)
    shutil.copy2(ROOT / 'scripts/install_webwalker.py', bundle / 'install_webwalker.py')
    shutil.copy2(ROOT / 'docs/WEBWALKER_UPLOAD.md', bundle / 'README.md')
    shutil.copy2(ROOT / 'docs/WEBWALKER_V4_ALIGNMENT.md', bundle / 'V4_ALIGNMENT.md')
    (bundle / 'deps').mkdir()
    wheel = Path(wheel)
    shutil.copy2(wheel, bundle / 'deps' / wheel.name)
    manifest = {'baseline_commit': BASELINE,
                'web_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'baseline_source_sha256': baseline,
                'payload_sha256': hashes, 'pdf_wheel': 'deps/' + wheel.name,
                'pdf_wheel_sha256': sha(wheel), 'dataset': 'WebWalkerQA main 680 questions',
                'protocol': 'provided_root_observed_link_traversal_v4_access_v2',
                'access_policy': 'paired_root_unavailable_exclusion_v1',
                'upstream_experiment_archive_sha256': '88b72a8e64b64ba4e33e806fb8023c3d24da333a1a467eb831b2e1dd47f9e4fc'}
    (bundle / 'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if validation:
        shutil.copy2(validation, bundle / 'validation.json')
    installer = '''#!/usr/bin/env bash
set -euo pipefail
TASK_BUNDLE="$(cd "$(dirname "$0")" && pwd)"
TASK_TARGET="${1:-${DAGBT_TARGET:-$HOME/dagbt_runs/20261015_webwalker_v4_access_v2_experiment/dagv2}}"
TASK_PYTHON="${DAGBT_PYTHON:-}"
if [[ -z "$TASK_PYTHON" ]]; then
  if [[ -x "$TASK_TARGET/.venv/bin/python" ]]; then
    TASK_PYTHON="$TASK_TARGET/.venv/bin/python"
  elif [[ -x "$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python" ]]; then
    TASK_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
  else
    TASK_PYTHON="python3"
  fi
fi
"$TASK_PYTHON" "$TASK_BUNDLE/install_webwalker.py" --target "$TASK_TARGET" --base "${DAGBT_PERSONAMEM_BASE:-$HOME/dagbt_runs/20261014_personamem_v4_experiment/dagv2}"
cd "$TASK_TARGET"
if [[ "${2:-with-browser}" != "--no-browser" ]]; then
  "$TASK_PYTHON" scripts/install_webwalker_browser.py --target "$TASK_TARGET"
fi
DAGBT_PYTHON="$TASK_PYTHON" bash scripts/run_webwalker.sh offline
echo "Installed and offline-verified. With PersonaMem complete or cleanly stopped: bash scripts/run_webwalker.sh smoke"
'''
    (bundle / 'install_and_smoke.sh').write_text(installer)
    (bundle / 'install_and_smoke.sh').chmod(0o755)
    archive = destination / (name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname=name)
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'payload_files': len(hashes), 'verified_baseline_files': len(baseline)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--pdf-wheel', required=True)
    parser.add_argument('--validation')
    args = parser.parse_args()
    build(args.destination, args.pdf_wheel, args.validation)
