"""Project management API routes."""

import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from routers._utils import atomic_write_json, validate_id

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectMeta(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str


def _list_project_dirs(projects_path: Path) -> list[dict]:
    """Scan project directories for project metadata."""
    results = []
    if not projects_path.exists():
        return results

    for proj_dir in projects_path.iterdir():
        meta_file = proj_dir / "project.json"
        if meta_file.exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return results


@router.post("")
async def create_project(body: ProjectCreate, app_data_path: str = ""):
    from config import settings
    projects_path = settings.projects_path
    projects_path.mkdir(parents=True, exist_ok=True)

    project_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    project_data = {
        "id": project_id,
        "name": body.name,
        "description": body.description,
        "created_at": now,
        "updated_at": now,
        "data_filename": None,
        "data_row_count": 0,
        "target_column": None,
        "smiles_column": None,
        "feature_count": 0,
        "schema_version": 2,
        "data_revision": 0,
    }

    proj_dir = projects_path / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "models").mkdir(exist_ok=True)

    atomic_write_json(proj_dir / "project.json", project_data)

    return project_data


@router.get("")
async def list_projects():
    from config import settings
    return _list_project_dirs(settings.projects_path)


@router.get("/{project_id}")
async def get_project(project_id: str):
    validate_id(project_id, "project_id")
    from config import settings
    meta_file = settings.projects_path / project_id / "project.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        with open(meta_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Corrupted project data: {e}")


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    validate_id(project_id, "project_id")
    from config import settings
    proj_dir = settings.projects_path / project_id
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    import shutil
    shutil.rmtree(proj_dir)
    return {"deleted": True}
