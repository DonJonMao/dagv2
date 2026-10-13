"""Installer transactions use tiny real hashes/matrices, no HTTP or secrets."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat

import numpy as np
import pytest

from dagbt import resources

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_personamem_v4.py'
spec = importlib.util.spec_from_file_location('installer_v4', SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(path):
    return {'bytes': path.stat().st_size, 'sha256': installer.file_hash(path)}


def inventory(root):
    return {p.relative_to(root).as_posix():
            (installer.file_hash(p), stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file() and not p.is_symlink()}


def deployment(root):
    write(root / 'DEPLOYMENT.json', {'release_name': 'test-v4',
        'base_commit': 'fixture', 'source_commit': 'fixture', 'validation': {},
        'files': {p.relative_to(root).as_posix(): record(p) for p in root.rglob('*')
                  if p.is_file() and p.name != 'DEPLOYMENT.json' and 'derived' not in p.parts}})


@pytest.fixture
def release(tmp_path, monkeypatch):
    for names in installer.ENV_KEYS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    root, base = tmp_path / 'new', tmp_path / 'base'
    root.mkdir(); base.mkdir()
    config = {'model_profile': 'bridgetree', 'llm_model': 'fixture-chat',
        'llm_base_url': 'http://fixture.invalid/chat/v1',
        'embedding_model': 'fixture-embedding', 'embedding_base_url': 'http://fixture.invalid/embedding/v1',
        'credentials_file': 'configs/credentials.local.json'}
    write(root / installer.CONFIG, config)
    write(base / installer.BASE_CONFIG, config)
    body = 'I like quiet visits.'
    corpus = {'docid': 'fixture-memory', 'title': 'Memory', 'text': body,
        'shared_context_id': 'fixture', 'source_message_indices': [0], 'roles': ['user'], 'time': {},
        'source_segments': [{'start': 0, 'end': len(body), 'role': 'user', 'source_message_indices': [0]}]}
    for directory in (root, base):
        write(directory / 'data/personamem/corpus.jsonl', corpus)
    vector_rel = 'data/hotpotqa/index/passage_vectors.npy'
    vector = base / vector_rel; vector.parent.mkdir(parents=True)
    vector.write_bytes(b'fixture frozen vector')
    frozen = [{'path': vector_rel, **record(vector)}]
    write(root / 'original_manifest.json', {'files': frozen})
    write(base / 'configs/credentials.local.json', {
        'llm': {'endpoint': installer.endpoint(config, 'llm'), 'api_key': 'stale-file-fixture'},
        'embedding': {'endpoint': installer.endpoint(config, 'embedding'), 'api_key': 'embedding-fixture'}})
    runtime = base / 'dagbt/model_runtime.py'; runtime.parent.mkdir()
    runtime.write_text('DEFAULT_BT_LLM_ENDPOINT = ' + repr(installer.endpoint(config, 'llm')) + '\n'
        'DEFAULT_BT_LLM_API_KEY = "old-effective-fixture"\nraise RuntimeError("must never execute base source")\n')
    report = resources.inspect_index(config, 'personamem', root=base)
    matrix = Path(report['vectors_path']); matrix.parent.mkdir(parents=True)
    np.save(matrix, np.array([[1, 0]], dtype=np.float32))
    write(Path(report['manifest_path']), {**report, 'status': 'complete', 'dimensions': 2,
        'dtype': 'float32', 'normalized': True, 'vectors_sha256': installer.file_hash(matrix)})
    deployment(root)
    return root, base, config, vector_rel, report


def test_install_is_idempotent_preserves_base_and_old_effective_credentials(release, capsys):
    root, base, config, vector_rel, _ = release
    before = inventory(base)
    result = installer.install(root, base)
    assert result['ready'] and result['base_modified'] is False and result['network_requests'] == 0
    assert inventory(base) == before
    assert (root / vector_rel).read_bytes() == (base / vector_rel).read_bytes()
    assert (root / vector_rel).stat().st_ino != (base / vector_rel).stat().st_ino
    assert (root / 'data/derived').is_symlink()
    credentials = root / config['credentials_file']
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600
    private = json.loads(credentials.read_text())
    assert private['llm']['api_key'] == 'old-effective-fixture'
    assert private['embedding']['api_key'] == 'embedding-fixture'
    assert resources.inspect_index(config, 'personamem', root=root)['status'] == 'ready'
    again = installer.install(root, base)
    assert again['ready'] and again['reused_frozen_vectors'] == 0 and inventory(base) == before
    public = json.dumps(result) + (root / 'DEPLOYMENT.json').read_text() + str(capsys.readouterr())
    assert all(secret not in public for secret in ('old-effective-fixture', 'stale-file-fixture', 'embedding-fixture'))


def test_environment_only_keys_do_not_get_copied_into_release(release, monkeypatch):
    root, base, config, _, _ = release
    (base / config['credentials_file']).unlink(); (base / 'dagbt/model_runtime.py').unlink()
    monkeypatch.setenv('DAG_LLM_API_KEY', 'environment-chat-fixture')
    monkeypatch.setenv('DAG_EMBED_API_KEY', 'environment-embed-fixture')
    result = installer.install(root, base)
    assert result['ready'] and not (root / config['credentials_file']).exists()
    assert 'environment-chat-fixture' not in json.dumps(result)


@pytest.mark.parametrize('mutate', ['source', 'base_vector', 'existing_vector', 'index_hash', 'model'])
def test_prevalidation_mismatch_makes_no_changes(release, mutate):
    root, base, config, vector_rel, report = release
    if mutate == 'source':
        (root / installer.CONFIG).write_text('{}')
    elif mutate == 'base_vector':
        (base / vector_rel).write_bytes(b'changed base artifact')
    elif mutate == 'existing_vector':
        target = root / vector_rel; target.parent.mkdir(parents=True); target.write_bytes(b'user existing artifact')
    elif mutate == 'index_hash':
        Path(report['vectors_path']).write_bytes(b'broken matrix')
    else:
        changed = deepcopy(config); changed['embedding_model'] = 'different-model'
        write(base / installer.BASE_CONFIG, changed)
    before_root, before_base = inventory(root), inventory(base)
    with pytest.raises((installer.InstallError, ValueError)):
        installer.install(root, base)
    assert inventory(root) == before_root and inventory(base) == before_base
    assert not (root / 'data/derived').exists()
    assert not list(root.glob('.install-v4-*'))


def test_post_creation_failure_rolls_back_only_created_paths(release, monkeypatch):
    root, base, config, vector_rel, _ = release
    before_root, before_base = inventory(root), inventory(base)
    original = resources.inspect_index
    def inspect(*args, **kwargs):
        if kwargs.get('root') == root and (root / 'data/derived').is_symlink():
            raise ValueError('injected final verification failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(resources, 'inspect_index', inspect)
    with pytest.raises(ValueError, match='injected'):
        installer.install(root, base)
    assert inventory(root) == before_root and inventory(base) == before_base
    assert not (root / vector_rel).exists() and not (root / config['credentials_file']).exists()
    assert not (root / 'data/derived').is_symlink() and not list(root.glob('.install-v4-*'))


@pytest.mark.parametrize('relative', ['../outside', '/tmp/outside', 'data/../outside', 'data//outside', 'data\\outside'])
def test_manifest_path_escape_rejected_before_mutation(release, relative):
    root, base, _, _, _ = release
    path = root / 'DEPLOYMENT.json'; manifest = json.loads(path.read_text())
    manifest['files'][relative] = {'bytes': 0, 'sha256': '0' * 64}; write(path, manifest)
    before_root, before_base = inventory(root), inventory(base)
    with pytest.raises(installer.InstallError, match='unsafe'):
        installer.install(root, base)
    assert inventory(root) == before_root and inventory(base) == before_base


def test_packaged_symlink_cannot_escape_source_verification(release):
    root, base, _, _, _ = release
    config = root / installer.CONFIG; config.unlink()
    config.symlink_to(base / installer.BASE_CONFIG)
    with pytest.raises(installer.InstallError, match='escapes|symlink'):
        installer.install(root, base)
    assert not (root / 'data/derived').exists()


def test_base_overlap_is_rejected(release):
    root, base, _, _, _ = release
    with pytest.raises(installer.InstallError, match='separate'):
        installer.install(root, root)
    with pytest.raises(installer.InstallError, match='separate'):
        installer.install(base / 'nested', base)


def test_missing_index_has_explicit_prepare_command_and_does_not_build(release):
    root, base, config, vector_rel, report = release
    Path(report['manifest_path']).unlink()
    with pytest.raises(installer.InstallError, match='prepare-index'):
        installer.install(root, base)
    assert not (root / vector_rel).exists() and not (root / config['credentials_file']).exists()


def test_existing_private_file_is_never_overwritten_or_chmodded(release):
    root, base, config, _, _ = release
    destination = root / config['credentials_file']
    destination.write_text('private invalid fixture'); destination.chmod(0o644)
    before = inventory(root)
    with pytest.raises(installer.InstallError, match='0600'):
        installer.install(root, base)
    assert inventory(root) == before


def test_unmanifested_source_is_rejected(release):
    root, base, _, _, _ = release
    write(root / 'dagbt/rogue.py', 'unmanifested source')
    with pytest.raises(installer.InstallError, match='Unmanifested'):
        installer.install(root, base)


def test_builtin_key_never_migrates_to_a_different_endpoint(release):
    root, base, config, _, _ = release
    path = base / 'dagbt/model_runtime.py'
    path.write_text('DEFAULT_BT_LLM_ENDPOINT = "http://other.invalid/v1/chat/completions"\n'
                    'DEFAULT_BT_LLM_API_KEY = "must-not-migrate-fixture"\n')
    installer.install(root, base)
    credentials = json.loads((root / config['credentials_file']).read_text())
    assert credentials['llm']['api_key'] == 'stale-file-fixture'
    assert 'must-not-migrate-fixture' not in (root / config['credentials_file']).read_text()


def test_unauthenticated_embedding_preserves_old_empty_key_fallback(release):
    root, base, config, _, _ = release
    write(base / config['credentials_file'], {'llm': {
        'endpoint': installer.endpoint(config, 'llm'), 'api_key': 'file-chat-fixture'}})
    assert installer.install(root, base)['ready']
    private = json.loads((root / config['credentials_file']).read_text())
    assert private['embedding'] == {'endpoint': installer.endpoint(config, 'embedding'), 'api_key': ''}


def test_empty_llm_key_is_not_misreported_as_ready(release):
    root, base, config, _, _ = release
    (base / 'dagbt/model_runtime.py').unlink()
    write(base / config['credentials_file'], {'llm': {
        'endpoint': installer.endpoint(config, 'llm'), 'api_key': ''}})
    with pytest.raises(installer.InstallError, match='nonempty'):
        installer.install(root, base)


def test_assets_only_installs_missing_assets_then_normal_install_requires_new_index(release):
    root, base, config, vector_rel, report = release
    Path(report['manifest_path']).unlink()
    before_base = inventory(base)
    with pytest.raises(installer.InstallError, match='assets-only'):
        installer.install(root, base)
    assert not (root / vector_rel).exists()
    result = installer.install(root, base, assets_only=True)
    assert result['assets_installed'] and not result['ready'] and result['network_requests'] == 0
    assert result['personamem_index'] == 'not_checked_or_built' and 'smoke_command' not in result
    assert any('prepare-index' in command for command in result['next_commands'])
    assert (root / vector_rel).read_bytes() == (base / vector_rel).read_bytes()
    assert (root / config['credentials_file']).is_file()
    assert not (root / 'data/derived').exists() and inventory(base) == before_base
    assert installer.install(root, base, assets_only=True)['reused_frozen_vectors'] == 0
    with pytest.raises(installer.InstallError, match='assets-only'):
        installer.install(root, base)
    # Emulate completion of the user's separate, explicit index build. Matrix
    # and manifest are real and undergo the production inspector's validation.
    own = resources.inspect_index(config, 'personamem', root=root)
    matrix = Path(own['vectors_path']); matrix.parent.mkdir(parents=True)
    np.save(matrix, np.array([[1, 0]], dtype=np.float32))
    write(Path(own['manifest_path']), {**own, 'status': 'complete', 'dimensions': 2,
        'dtype': 'float32', 'normalized': True, 'vectors_sha256': installer.file_hash(matrix)})
    assert installer.install(root, base)['ready']
    assert not (root / 'data/derived').is_symlink() and inventory(base) == before_base


def test_assets_only_ignores_incompatible_base_index_without_link_or_build(release, monkeypatch):
    root, base, config, _, report = release
    Path(report['vectors_path']).write_bytes(b'corrupt index; explicit fallback must not use this')
    def forbidden(*args, **kwargs):
        raise AssertionError('Assets-only must not inspect or build derived indices')
    monkeypatch.setattr(resources, 'inspect_index', forbidden)
    monkeypatch.setattr(resources, 'ensure_index', forbidden)
    assert not installer.install(root, base, assets_only=True)['ready']
    assert not (root / 'data/derived').exists()


def test_assets_only_refuses_existing_shared_cache_without_mutating_it(release):
    root, base, _, _, _ = release
    (root / 'data/derived').symlink_to(base / 'data/derived', target_is_directory=True)
    before_root, before_base = inventory(root), inventory(base)
    with pytest.raises(installer.InstallError, match='existing derived cache'):
        installer.install(root, base, assets_only=True)
    assert inventory(root) == before_root and inventory(base) == before_base


def test_assets_only_rolls_back_on_post_copy_validation_failure(release, monkeypatch):
    root, base, _, vector_rel, _ = release
    before_root, before_base = inventory(root), inventory(base)
    verify = installer.release_manifest
    calls = []
    def fail_final(path):
        calls.append(path)
        if len(calls) > 1:
            raise installer.InstallError('Injected source verification failure')
        return verify(path)
    monkeypatch.setattr(installer, 'release_manifest', fail_final)
    with pytest.raises(installer.InstallError, match='Injected'):
        installer.install(root, base, assets_only=True)
    assert inventory(root) == before_root and inventory(base) == before_base
    assert not (root / vector_rel).exists() and not list(root.glob('.install-v4-*'))
