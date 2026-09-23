from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable

PROJECT_NAME = "bridgetree_preference_rag"
BUNDLE_MANIFEST = "SERVER_BUNDLE_MANIFEST.json"
ROOT_FILES = (
    ".gitignore",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-ascend910b.txt",
    "问题.md",
)
ROOT_DIRECTORIES = ("configs", "data", "docs", "reference", "scripts", "src", "tests")
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "outputs",
}


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _excluded(relative: Path) -> bool:
    if relative.parts[:2] == ("data", "protocol"):
        return True
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return True
    name = relative.name.lower()
    if name == ".ds_store" or name == ".env" or name.startswith(".env."):
        return True
    if "credential" in name and relative.suffix.lower() in {".yaml", ".yml", ".json"}:
        return True
    return relative.suffix.lower() in {
        ".key",
        ".p12",
        ".pem",
        ".pfx",
        ".pyc",
        ".pyo",
        ".tmp",
    }


def server_bundle_files(repository_root: str | Path) -> list[Path]:
    root = Path(repository_root).resolve()
    selected: set[Path] = set()
    for name in ROOT_FILES:
        path = root / name
        if path.is_file():
            selected.add(path)
    for name in ROOT_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"server bundle source directory is missing: {directory}")
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(root)
                if not _excluded(relative):
                    selected.add(path)
    required = {
        root / "configs" / "personamem32k_effect_first.yaml",
        root / "configs" / "train.yaml",
        root / "data" / "raw" / "personamem-v1" / "questions_32k.csv",
        root / "data" / "raw" / "personamem-v1" / "shared_contexts_32k.jsonl",
        root / "scripts" / "run_effect_first_validation.sh",
        root / "scripts" / "background_entrypoint.py",
        root / "scripts" / "run_chain.sh",
        root / "scripts" / "start_chain_linux.sh",
        root / "scripts" / "start_effect_first_background.sh",
        root / "scripts" / "start_train_32k_background.sh",
        root / "scripts" / "train_32k.sh",
        root / "src" / "bridgetree" / "background.py",
        root / "src" / "bridgetree" / "guided_retriever.py",
        root / "src" / "bridgetree" / "ranking.py",
        root / "src" / "bridgetree" / "training.py",
    }
    missing = sorted(str(path.relative_to(root)) for path in required if path not in selected)
    if missing:
        raise FileNotFoundError(f"server bundle is missing required files: {missing}")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _manifest(repository_root: Path, files: Iterable[Path]) -> Dict[str, Any]:
    entries = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(repository_root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size": size,
                "mode": path.stat().st_mode & 0o777,
            }
        )
    return {
        "format_version": 1,
        "project": PROJECT_NAME,
        "archive_root": PROJECT_NAME,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def _add_file(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = archive.gettarinfo(str(source), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def build_server_bundle(repository_root: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    root = Path(repository_root).resolve()
    files = server_bundle_files(root)
    manifest = _manifest(root, files)
    destination_root = Path(output_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    archive_path = destination_root / f"{PROJECT_NAME}_server_{time.time_ns()}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            _add_file(archive, path, f"{PROJECT_NAME}/{relative}")
        payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        info = tarfile.TarInfo(f"{PROJECT_NAME}/{BUNDLE_MANIFEST}")
        info.size = len(payload)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))
    verification = verify_server_bundle(archive_path)
    archive_sha256 = _sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8")
    return {
        "status": "created",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "checksum_file": str(checksum_path),
        "archive_bytes": archive_path.stat().st_size,
        **verification,
    }


def _validated_members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    prefix = f"{PROJECT_NAME}/"
    for member in archive.getmembers():
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.name.startswith(prefix):
            raise ValueError(f"unsafe server bundle member: {member.name}")
        if not member.isfile():
            raise ValueError(f"server bundle contains a non-file member: {member.name}")
        if member.name in members:
            raise ValueError(f"duplicate server bundle member: {member.name}")
        members[member.name] = member
    return members


def verify_server_bundle(archive_path: str | Path) -> Dict[str, Any]:
    path = Path(archive_path).resolve()
    with tarfile.open(path, "r:gz") as archive:
        members = _validated_members(archive)
        manifest_name = f"{PROJECT_NAME}/{BUNDLE_MANIFEST}"
        if manifest_name not in members:
            raise ValueError("server bundle manifest is missing")
        manifest_handle = archive.extractfile(members[manifest_name])
        if manifest_handle is None:
            raise ValueError("server bundle manifest cannot be read")
        manifest = json.loads(manifest_handle.read().decode("utf-8"))
        if (
            manifest.get("format_version") != 1
            or manifest.get("project") != PROJECT_NAME
            or manifest.get("archive_root") != PROJECT_NAME
        ):
            raise ValueError("server bundle manifest identity/version is invalid")
        entries = manifest.get("files")
        if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
            raise ValueError("server bundle manifest file count is invalid")
        entry_paths = [str(entry["path"]) for entry in entries]
        if len(set(entry_paths)) != len(entry_paths):
            raise ValueError("server bundle manifest contains duplicate paths")
        if manifest.get("total_bytes") != sum(int(entry["size"]) for entry in entries):
            raise ValueError("server bundle manifest payload size is invalid")
        expected_names = {f"{PROJECT_NAME}/{entry['path']}" for entry in entries}
        if set(members) != expected_names | {manifest_name}:
            raise ValueError("server bundle file set does not match its manifest")
        for entry in entries:
            member = members[f"{PROJECT_NAME}/{entry['path']}"]
            handle = archive.extractfile(member)
            if handle is None or _sha256_stream(handle) != entry["sha256"]:
                raise ValueError(f"server bundle checksum mismatch: {entry['path']}")
            if member.size != entry["size"] or member.mode & 0o777 != entry["mode"]:
                raise ValueError(f"server bundle metadata mismatch: {entry['path']}")
        launcher = members[f"{PROJECT_NAME}/scripts/train_32k.sh"]
        if not launcher.mode & 0o111:
            raise ValueError("server bundle launcher is not executable")
        executable_launchers = (
            "scripts/run_chain.sh",
            "scripts/start_chain_linux.sh",
            "scripts/run_effect_first_validation.sh",
            "scripts/start_effect_first_background.sh",
            "scripts/start_train_32k_background.sh",
        )
        for launcher_name in executable_launchers:
            launcher = members[f"{PROJECT_NAME}/{launcher_name}"]
            if not launcher.mode & 0o111:
                raise ValueError(f"server bundle launcher is not executable: {launcher_name}")
    return {
        "verified": True,
        "file_count": int(manifest["file_count"]),
        "payload_bytes": int(manifest["total_bytes"]),
        "launcher_executable": True,
    }


def verify_bundle_offline_launcher(archive_path: str | Path, python_executable: str) -> Dict[str, Any]:
    path = Path(archive_path).resolve()
    verify_server_bundle(path)
    with tempfile.TemporaryDirectory(prefix="bridgetree_server_bundle_") as temporary:
        extraction_root = Path(temporary)
        with tarfile.open(path, "r:gz") as archive:
            _validated_members(archive)
            if hasattr(tarfile, "data_filter"):
                archive.extractall(extraction_root, filter="data")
            else:  # Python 3.9-3.11; every member was already path/type validated above.
                archive.extractall(extraction_root)
        project_root = extraction_root / PROJECT_NAME
        environment = {
            **os.environ,
            "PYTHONPATH": "",
            "BRIDGETREE_PYTHON": python_executable,
            "BOOTSTRAP": "false",
            "RUN_CHECKS": "false",
            "CHECK_SERVICES": "false",
            "DOWNLOAD_DATA": "false",
            "PREFLIGHT_ONLY": "true",
        }
        completed = subprocess.run(
            ["bash", "scripts/train_32k.sh"],
            cwd=project_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if '"status": "ready"' not in completed.stdout or "tuning was not started" not in completed.stdout:
            raise RuntimeError("extracted server bundle did not complete the offline launcher preflight")
    return {"offline_launcher_verified": True}
