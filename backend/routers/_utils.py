"""Shared utilities for backend routers."""

import re
import json
import os
import tempfile
from pathlib import Path
from fastapi import HTTPException

# Only allow safe characters in IDs to prevent path traversal
_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def validate_id(value: str, name: str = "ID") -> str:
    """Validate an ID parameter, raising 400 if it contains unsafe characters."""
    if not _SAFE_ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {name}")
    return value


def get_project_dir(project_id: str) -> Path:
    """Get project directory path from project ID."""
    validate_id(project_id, "project_id")
    from config import settings
    return settings.projects_path / project_id


def safe_child_path(parent: Path, filename: str) -> Path:
    """Resolve a direct child path and reject traversal or nested paths."""
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    parent_resolved = parent.resolve()
    candidate = (parent_resolved / filename).resolve()
    if candidate.parent != parent_resolved:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate


def atomic_write_json(path: Path, data: dict | list):
    """Write JSON atomically so interrupted writes do not corrupt project state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _migrate_meta(meta_file: Path, meta: dict) -> dict:
    """Migrate project metadata to schema v2 while preserving the v1 file."""
    if int(meta.get("schema_version", meta.get("schemaVersion", 1))) >= 2:
        return meta

    backup = meta_file.with_name("project.v1.backup.json")
    if not backup.exists():
        atomic_write_json(backup, meta)

    migrated = dict(meta)
    migrated["schema_version"] = 2
    migrated["legacy_models_present"] = any(
        p.suffix == ".joblib" for p in (meta_file.parent / "models").glob("*.joblib")
    ) if (meta_file.parent / "models").exists() else False
    atomic_write_json(meta_file, migrated)
    return migrated


def load_meta(project_id: str) -> dict:
    """Load project metadata, raise 404 if not found."""
    proj_dir = get_project_dir(project_id)
    meta_file = proj_dir / "project.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    return _migrate_meta(meta_file, meta)


def save_meta(project_id: str, meta: dict):
    """Save project metadata to disk."""
    proj_dir = get_project_dir(project_id)
    meta["schema_version"] = 2
    atomic_write_json(proj_dir / "project.json", meta)
