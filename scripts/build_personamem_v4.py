#!/usr/bin/env python3
"""Build and verify a credential-free release from a clean committed checkout."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a5b993a7d23cdd92c34b7ca45ebfb97b652a43e1'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def private_values(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in ('api_key', 'authorization', 'access_token', 'secret') and isinstance(item, str) and item:
                found.append(item.encode())
            found.extend(private_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(private_values(item))
    return found


def build(output, validation_file, private_files):
    if git('status', '--porcelain', '--untracked-files=normal').strip():
        raise ValueError('Commit the reviewed source before building a release')
    commit = git('rev-parse', 'HEAD').decode().strip()
    git('merge-base', '--is-ancestor', BASE, commit)
    validation = json.loads(Path(validation_file).read_text())
    if validation.get('release_ready') is not True or validation.get('source_commit') != commit:
        raise ValueError('Validation must approve this exact source commit')
    secrets = []
    for path in private_files:
        secrets.extend(private_values(json.loads(Path(path).read_text())))
    if not secrets:
        raise ValueError('Supply the local effective credential file for an exact secret scan')
    files, metadata = {}, {}
    for entry in git('ls-tree', '-rz', '--full-tree', commit).decode().split('\0'):
        if not entry:
            continue
        header, relative = entry.split('\t', 1)
        mode, kind, blob = header.split()
        if relative in ('DEPLOYMENT.json', 'VALIDATION_PERSONAMEM_V4.json'):
            raise ValueError('Generated release metadata must not be a tracked source artifact')
        if (kind != 'blob' or mode not in ('100644', '100755')
                or relative.endswith(('.local.json', '.npy', '.tar.gz', '.tgz'))
                or any(part in {'.git', '.venv', 'outputs', 'derived', '__pycache__'} for part in Path(relative).parts)):
            raise ValueError('Disallowed tracked release path: ' + relative)
        # Read committed blobs, never mutable working-tree bytes. An edit/revert
        # concurrent with the build cannot mislabel source as this commit.
        data = git('cat-file', 'blob', blob)
        if any(secret in data for secret in secrets):
            raise ValueError('Private credential found in tracked artifact: ' + relative)
        files[relative] = data
    files['VALIDATION_PERSONAMEM_V4.json'] = (json.dumps(validation, ensure_ascii=False, indent=2) + '\n').encode()
    for relative, data in files.items():
        metadata[relative] = {'bytes': len(data), 'sha256': digest(data)}
    manifest = {'release_name': 'personamem-v4', 'base_commit': BASE, 'source_commit': commit,
                'files': metadata, 'validation': validation}
    files['DEPLOYMENT.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    for relative, data in files.items():
        if any(secret in data for secret in secrets):
            raise ValueError('Private credential found in release artifact: ' + relative)
    output = Path(output).resolve()
    checksum = output.with_name(output.name + '.sha256')
    if output.exists() or checksum.exists():
        raise ValueError('Release destination already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.build-personamem-', dir=output.parent) as temp:
        archive = Path(temp) / output.name
        with tarfile.open(archive, 'w:gz') as tar:
            for relative, data in sorted(files.items()):
                info = tarfile.TarInfo('dagv2/' + relative)
                info.size = len(data)
                info.mode = 0o755 if relative.endswith('.sh') else 0o644
                info.mtime = int(git('show', '-s', '--format=%ct', commit)) if relative == 'DEPLOYMENT.json' else 0
                tar.addfile(info, io.BytesIO(data))
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            if len(members) != len(files) or {m.name for m in members} != {'dagv2/' + p for p in files}:
                raise ValueError('Archive member verification failed')
            for member in members:
                if not member.isfile() or tar.extractfile(member).read() != files[member.name[6:]]:
                    raise ValueError('Archive round-trip verification failed')
        if git('status', '--porcelain', '--untracked-files=normal').strip() or git('rev-parse', 'HEAD').decode().strip() != commit:
            raise ValueError('Source changed during packaging')
        archive_hash = digest(archive.read_bytes())
        archive.rename(output)
        checksum.write_text(archive_hash + '  ' + output.name + '\n')
    return {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': archive_hash,
            'files': len(files), 'source_commit': commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-file', required=True)
    parser.add_argument('--private-credentials', action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.validation_file, args.private_credentials), indent=2))


if __name__ == '__main__':
    main()
