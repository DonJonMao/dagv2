#!/usr/bin/env python3
"""Verify a fresh v4 source release and reuse verified local deployment assets.

No model requests, index builds, base writes or secret logging are performed.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path.home() / 'dagbt_runs/20261011_personamem_a48/dagv2'
CONFIG = 'configs/personamem-v4.json'
BASE_CONFIG = 'configs/residual-memory.personamem16k.json'
VECTOR = re.compile(r'data/(hotpotqa|2wikimultihopqa|musique)/index/passage_vectors\.npy\Z')
ENV_KEYS = {'llm': ('DAG_LLM_API_KEY', 'BRIDGETREE_CHAT_API_KEY'),
            'embedding': ('DAG_EMBED_API_KEY',)}


class InstallError(ValueError):
    pass


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeError, ValueError):
        # JSON exceptions can quote private input; never forward their text.
        raise InstallError(label + ' is unavailable or malformed') from None


def relative_path(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or str(PurePosixPath(value)) != value
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise InstallError('Manifest/config contains an unsafe relative path')
    return value


def within(root, relative, *, symlinks=False):
    relative = relative_path(relative)
    root = Path(root).resolve(); path = root / relative
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise InstallError('Manifest/config path escapes its deployment directory') from None
    if not symlinks:
        current = root
        for part in relative.split('/'):
            current = current / part
            if current.is_symlink():
                raise InstallError('Release and frozen artifacts must not use symlink paths')
    return path


def verify_file(path, expected, label):
    if (not isinstance(expected, dict) or type(expected.get('bytes')) is not int
            or expected['bytes'] < 0 or not isinstance(expected.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', expected['sha256']) is None):
        raise InstallError('Malformed file integrity record')
    if (not path.is_file() or path.stat().st_size != expected['bytes']
            or file_hash(path) != expected['sha256']):
        raise InstallError(label + ' is missing or its integrity check failed')


def release_manifest(root):
    path = within(root, 'DEPLOYMENT.json')
    manifest = read_json(path, 'Release manifest')
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not {CONFIG, 'original_manifest.json'} <= set(files):
        raise InstallError('Release manifest is incomplete')
    for relative, expected in files.items():
        if relative == 'DEPLOYMENT.json':
            raise InstallError('Release manifest must not hash itself')
        verify_file(within(root, relative), expected, 'Packaged artifact ' + relative)
    return manifest


def frozen_assets(root, base):
    manifest = read_json(within(root, 'original_manifest.json'), 'Frozen artifact manifest')
    rows = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise InstallError('Frozen artifact manifest is incomplete')
    seen, missing = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise InstallError('Malformed frozen artifact record')
        relative = relative_path(row.get('path'))
        if relative in seen:
            raise InstallError('Duplicate frozen artifact record')
        seen.add(relative)
        destination = within(root, relative)
        if destination.exists():
            verify_file(destination, row, 'Existing frozen artifact ' + relative)
        else:
            if not VECTOR.fullmatch(relative):
                raise InstallError('Packaged frozen source/data is missing: ' + relative)
            source = within(base, relative)
            verify_file(source, row, 'Base frozen vector ' + relative)
            missing.append((source, destination, row))
    return rows, missing


def endpoint(config, kind):
    key, suffix = ('llm_base_url', '/chat/completions') if kind == 'llm' else ('embedding_base_url', '/embeddings')
    value = config.get(key)
    if not isinstance(value, str):
        raise InstallError('Release config is missing a model endpoint')
    return value.rstrip('/') + suffix


def env_key(kind):
    for name in ENV_KEYS[kind]:
        value = os.environ.get(name)
        if value:
            if '\n' in value or '\r' in value:
                raise InstallError('Model credential environment value must be a single line')
            return True
    return False


def credential_entries(path, config):
    raw = read_json(path, 'Local credential file')
    if not isinstance(raw, dict):
        raise InstallError('Local credential file must be an object')
    entries = {}
    for kind in ENV_KEYS:
        row = raw.get(kind)
        if row is None:
            continue
        if (not isinstance(row, dict) or not isinstance(row.get('api_key'), str)
                or any(c in row['api_key'] for c in ('\n', '\r'))):
            raise InstallError('Local model credential entry is malformed')
        if row.get('endpoint') == endpoint(config, kind):
            entries[kind] = {'endpoint': row['endpoint'], 'api_key': row['api_key']}
    return entries


def legacy_local_llm(base, config):
    """Read the old locally deployed constant without importing/executing it.

    This migrates the old deployment's private credential into a mode-0600 local
    file. Neither its literal value nor a digest is a release artifact or log.
    """
    path = within(base, 'dagbt/model_runtime.py')
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text())
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ('DEFAULT_BT_LLM_ENDPOINT', 'DEFAULT_BT_LLM_API_KEY'):
                    values[name] = ast.literal_eval(node.value)
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError):
        raise InstallError('Cannot inspect legacy local credential configuration') from None
    key = values.get('DEFAULT_BT_LLM_API_KEY')
    if (values.get('DEFAULT_BT_LLM_ENDPOINT') == endpoint(config, 'llm')
            and isinstance(key, str) and key and '\n' not in key and '\r' not in key):
        return {'endpoint': endpoint(config, 'llm'), 'api_key': key}
    return None


def plan_credentials(root, base, config, base_config):
    destination = within(root, config.get('credentials_file', 'configs/credentials.local.json'))
    if destination.exists():
        if stat.S_IMODE(destination.stat().st_mode) != 0o600:
            raise InstallError('Existing local credential file must have permission 0600; it was not changed')
        entries = credential_entries(destination, config)
        if not env_key('llm') and not entries.get('llm', {}).get('api_key'):
            raise InstallError('Existing local LLM credentials are empty or do not match its endpoint; set DAG_LLM_API_KEY')
        # The deployed embedding endpoint is intentionally unauthenticated;
        # missing or empty embedding keys retain the old resolver's behavior.
        env_key('embedding')
        return destination, None
    entries = {}
    configured = base_config.get('credentials_file', 'configs/credentials.local.json')
    if not isinstance(configured, str):
        raise InstallError('Base local credential path is invalid')
    source = Path(configured).expanduser()
    if source.is_absolute():
        try:
            configured = source.resolve().relative_to(base).as_posix()
        except ValueError:
            raise InstallError('Base credential file is outside --base; use model key environment variables') from None
    source = within(base, configured)
    if source.is_file():
        entries = credential_entries(source, config)
    # The old runtime used environment > matching deployed constant > local
    # file. Preserve that effective fallback, including when the old file is
    # stale; the new runtime itself uses environment > this migrated file.
    legacy = legacy_local_llm(base, config)
    if legacy is not None:
        entries['llm'] = legacy
    if not env_key('llm') and not entries.get('llm', {}).get('api_key'):
        raise InstallError('No nonempty matching local LLM credential; set DAG_LLM_API_KEY')
    if not env_key('embedding') and 'embedding' not in entries:
        entries['embedding'] = {'endpoint': endpoint(config, 'embedding'), 'api_key': ''}
    if not entries:
        return destination, None
    return destination, (json.dumps(entries, ensure_ascii=False, indent=2) + '\n').encode()


def ensure_unlisted_sources_absent(root, files, frozen, credential_path):
    allowed = set(files) | {row['path'] for row in frozen} | {'DEPLOYMENT.json', credential_path.relative_to(root).as_posix()}
    ignored = {'.git', '.venv', '.pytest_cache', '__pycache__', 'outputs', 'runs'}
    for directory, dirs, names in os.walk(root, followlinks=False):
        current = Path(directory)
        dirs[:] = [d for d in dirs if d not in ignored and not d.startswith('.venv')
                   and (current / d).relative_to(root).as_posix() != 'data/derived']
        for name in names:
            relative = (current / name).relative_to(root).as_posix()
            if relative not in allowed and not name.endswith(('.pyc', '.pyo')):
                raise InstallError('Unmanifested release artifact: ' + relative)


def make_parents(path, root, created_dirs):
    missing = []
    parent = path.parent
    while parent != root and not parent.exists():
        missing.append(parent); parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(); created_dirs.append(directory)


def install(root=ROOT, base=DEFAULT_BASE, *, assets_only=False):
    root, base = Path(root).resolve(), Path(base).expanduser().resolve()
    if root == base or root in base.parents or base in root.parents:
        raise InstallError('Extract into a separate new directory; --base must not overlap the release')
    if not base.is_dir():
        raise InstallError('Base deployment directory does not exist')
    release = release_manifest(root)  # Must precede imports and every mutation.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dagbt import runner
    from dagbt.resources import inspect_index
    from dagbt.methods import RESIDUAL_V2
    config = runner.validate_config(read_json(within(root, CONFIG), 'Release config'))
    runner.validate_experiment_options(config, SimpleNamespace(arms=['original', RESIDUAL_V2], limit=None))
    base_config_path = within(base, BASE_CONFIG)
    base_config = read_json(base_config_path, 'Base config') if base_config_path.is_file() else {}
    if not isinstance(base_config, dict):
        raise InstallError('Base config must be an object')
    rows, missing = frozen_assets(root, base)
    credential_path, credential_data = plan_credentials(root, base, config, base_config)
    if credential_path.relative_to(root).as_posix() in release['files']:
        raise InstallError('Private credentials must not be part of the release manifest')
    ensure_unlisted_sources_absent(root, release['files'], rows, credential_path)
    if config.get('derived_index_dir', 'data/derived') != 'data/derived':
        raise InstallError('Upload release requires derived_index_dir=data/derived')
    derived = root / 'data/derived'
    within(root, 'data')
    if assets_only and (derived.exists() or derived.is_symlink()):
        raise InstallError('--assets-only requires no existing derived cache; an existing cache was not changed. Use normal install to verify a completed cache.')
    try:
        current = None if assets_only else inspect_index(config, 'personamem', root=root)
    except (OSError, ValueError, TypeError, KeyError):
        raise InstallError('Existing release index failed integrity validation; no artifacts were replaced. Inspect that cache or use a separate new release directory.') from None
    source = None
    if not assets_only and current['status'] != 'ready':
        if derived.exists() or derived.is_symlink():
            raise InstallError('Existing derived index is not compatible; it was not changed. Run prepare-index explicitly in this release.')
        for field in ('embedding_model', 'embedding_base_url'):
            if field in base_config and str(base_config[field]).rstrip('/') != str(config[field]).rstrip('/'):
                raise InstallError('Base embedding configuration differs from this release. Run install --assets-only, then prepare-index explicitly in this release; see README_PERSONAMEM_V4.md.')
        base_index_config = dict(config)
        base_index_config['derived_index_dir'] = base_config.get('derived_index_dir', 'data/derived')
        try:
            existing = inspect_index(base_index_config, 'personamem', root=base)
        except (OSError, ValueError, TypeError, KeyError):
            raise InstallError('Base index failed integrity validation; it was not changed. Run install --assets-only, then prepare-index explicitly in this release; see README_PERSONAMEM_V4.md.') from None
        if existing['status'] != 'ready' or existing['identity'] != current['identity']:
            raise InstallError('No compatible completed PersonaMem index. Run install --assets-only, then prepare-index explicitly in this release; see README_PERSONAMEM_V4.md.')
        source = Path(existing['index_dir']).parents[1].resolve()
    created, created_dirs, staging = [], [], None
    try:
        staging = Path(tempfile.mkdtemp(prefix='.install-v4-', dir=root))
        for index, (source_file, destination, expected) in enumerate(missing):
            staged = staging / ('vector-' + str(index))
            shutil.copyfile(source_file, staged)
            verify_file(staged, expected, 'Staged frozen vector')
            staged.chmod(0o644)
        if credential_data is not None:
            staged_credentials = staging / 'credentials'
            descriptor = os.open(staged_credentials, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(credential_data)
        for index, (_, destination, _) in enumerate(missing):
            make_parents(destination, root, created_dirs)
            os.link(staging / ('vector-' + str(index)), destination)
            created.append(destination)
        if credential_data is not None:
            make_parents(credential_path, root, created_dirs)
            os.link(staged_credentials, credential_path)
            created.append(credential_path)
        if source is not None:
            derived.symlink_to(source, target_is_directory=True)
            created.append(derived)
        release_manifest(root)
        frozen_assets(root, base)
        if not assets_only and inspect_index(config, 'personamem', root=root)['status'] != 'ready':
            raise InstallError('Final derived index verification failed')
        plan_credentials(root, base, config, base_config)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for directory in reversed(created_dirs):
            directory.rmdir()
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging)
    result = {'ready': not assets_only, 'assets_installed': True,
            'release': release.get('release_name', 'personamem-v4'),
            'frozen_artifacts_verified': len(rows), 'reused_frozen_vectors': len(missing),
            'personamem_index': 'not_checked_or_built' if assets_only else 'verified',
            'credentials': 'local_or_environment', 'base_modified': False, 'network_requests': 0}
    if assets_only:
        result['next_commands'] = ['cd ' + shlex.quote(str(root)),
            ' '.join([shlex.quote(sys.executable), 'scripts/personamem_v4.py prepare-index']),
            ' '.join([shlex.quote(sys.executable), 'scripts/install_personamem_v4.py', '--base', shlex.quote(str(base))])]
        result['note'] = 'Only local assets installed. prepare-index is a separate explicit action that makes embedding requests. Re-run normal install before smoke.'
    else:
        result['smoke_command'] = ' '.join([shlex.quote(sys.executable), shlex.quote(str(root / 'scripts/personamem_v4.py')), 'smoke'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=str(DEFAULT_BASE))
    parser.add_argument('--assets-only', action='store_true',
                        help='Explicit fallback: install verified local frozen vectors/credentials only; do not link/build an index or declare readiness')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(install(base=args.base, assets_only=args.assets_only), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        # Our controlled messages contain paths/status only. Arbitrary dependency
        # errors may include private input and are deliberately not echoed.
        message = str(exc) if isinstance(exc, InstallError) else 'Installation validation failed (' + type(exc).__name__ + '); no existing artifacts were replaced.'
        print(json.dumps({'ready': False, 'error': message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
