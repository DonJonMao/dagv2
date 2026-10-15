#!/usr/bin/env python3
"""Install an isolated WebWalker deployment from the exact PersonaMem v4 core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(bundle, target, base):
    bundle, target, base = Path(bundle).resolve(), Path(target).resolve(), Path(base).resolve()
    manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
    if target == base or target in base.parents or base in target.parents:
        raise ValueError('Use a separate WebWalker target outside the PersonaMem deployment')
    if not (base / 'dagbt/engine.py').is_file():
        raise ValueError('Base is not an existing PersonaMem v4 deployment')
    for path, expected in manifest['baseline_source_sha256'].items():
        source = base / path
        if not source.is_file() or sha(source) != expected:
            raise ValueError('Target differs from required e667708 v4 core: ' + path)
    entries = []
    for path, expected in manifest['payload_sha256'].items():
        source = bundle / 'payload' / path
        if not source.is_file() or sha(source) != expected:
            raise ValueError('Bundle payload failed hash verification: ' + path)
        destination = target / path
        if destination.exists() and sha(destination) != expected:
            raise ValueError('Preserving an existing different addon/data file: ' + path)
        entries.append((destination, source.read_bytes()))
    # Local credentials are never part of the archive. Reuse a validated v4
    # credential file in the isolated target, without touching its source.
    config = json.loads((bundle / 'payload/configs/webwalker-v4.json').read_text())
    base_config_path = base / 'configs/personamem-v4.json'
    base_config = json.loads(base_config_path.read_text()) if base_config_path.exists() else config
    credentials = base_config.get('credentials_file')
    credential_source = (base / credentials).resolve() if credentials else None
    credential_target = target / config['credentials_file']
    if not credential_target.exists() and credential_source and credential_source.is_file():
        # Parsing catches a malformed file without displaying any secrets.
        if not isinstance(json.loads(credential_source.read_text()), dict):
            raise ValueError('Base credential file must contain an object')
        entries.append((credential_target, credential_source.read_bytes()))
    wheel = bundle / manifest['pdf_wheel']
    if sha(wheel) != manifest['pdf_wheel_sha256']:
        raise ValueError('Bundled PDF dependency hash mismatch')
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe wheel member')
            if member.is_dir():
                continue
            destination = target / '_webwalker_vendor' / path
            content = archive.read(member)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError('Existing vendored PDF dependency differs: ' + str(path))
            entries.append((destination, content))
    created = []
    try:
        for destination, content in entries:
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + '.webwalker-install.tmp')
            try:
                if destination == credential_target:
                    # Restrict the temporary file before any secret bytes are
                    # written, not only after the atomic rename.
                    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(content)
                else:
                    temp.write_bytes(content)
                temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
            created.append(destination)
            if destination == credential_target:
                destination.chmod(0o600)
        for path, expected in manifest['baseline_source_sha256'].items():
            if sha(target / path) != expected or sha(base / path) != expected:
                raise ValueError('Core changed during installation: ' + path)
    except BaseException:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise
    print(json.dumps({'installed': True, 'target': str(target), 'required_baseline': manifest['baseline_commit'],
                      'base': str(base), 'isolated_from_personamem': True,
                      'new_files': len(created), 'core_overwrites': 0, 'credentials_in_bundle': False,
                      'local_credentials_reused': credential_target.is_file()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--base', default=str(Path.home() / 'dagbt_runs/20261014_personamem_v4_experiment/dagv2'))
    parser.add_argument('--bundle', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    install(args.bundle, args.target, args.base)
