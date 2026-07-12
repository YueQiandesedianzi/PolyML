"""Shared utilities for backend routers."""

import re
import json
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


def load_meta(project_id: str) -> dict:
    """Load project metadata, raise 404 if not found."""
    proj_dir = get_project_dir(project_id)
    meta_file = proj_dir / "project.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    with open(meta_file, encoding="utf-8") as f:
        return json.load(f)


def save_meta(project_id: str, meta: dict):
    """Save project metadata to disk."""
    proj_dir = get_project_dir(project_id)
    with open(proj_dir / "project.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
