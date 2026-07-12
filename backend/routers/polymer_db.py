"""Polymer name → SMILES database management API routes."""

import json
import re
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func, or_
import db.database as db_mod
from db.models import Polymer

router = APIRouter(prefix="/api/polymer-db", tags=["polymer-db"])


class PolymerCreate(BaseModel):
    common_name: str
    abbreviation: str = ""
    smiles: str
    source: str = "user"
    tags: list[str] = []


class PolymerUpdate(BaseModel):
    common_name: str | None = None
    abbreviation: str | None = None
    smiles: str | None = None
    tags: list[str] | None = None


@router.get("/search")
async def search_polymers(q: str = Query(..., min_length=1)):
    async with db_mod.async_session() as session:
        pattern = f"%{q}%"
        stmt = select(Polymer).where(
            or_(
                Polymer.common_name.ilike(pattern),
                Polymer.abbreviation.ilike(pattern),
            )
        ).limit(20)
        result = await session.execute(stmt)
        polymers = result.scalars().all()

        return [_polymer_to_dict(p) for p in polymers]


@router.get("")
async def list_polymers(page: int = 1, limit: int = 50):
    async with db_mod.async_session() as session:
        offset = (page - 1) * limit

        count_stmt = select(func.count(Polymer.id))
        total = (await session.execute(count_stmt)).scalar()

        stmt = select(Polymer).offset(offset).limit(limit).order_by(Polymer.common_name)
        result = await session.execute(stmt)
        polymers = result.scalars().all()

        return {
            "items": [_polymer_to_dict(p) for p in polymers],
            "total": total,
        }


@router.post("")
async def add_polymer(body: PolymerCreate):
    now = datetime.now(timezone.utc).isoformat()

    # Try to canonicalize SMILES with RDKit
    canonical = ""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(body.smiles.replace("*", ""))
        if mol:
            canonical = Chem.MolToSmiles(mol)
    except Exception:
        pass

    async with db_mod.async_session() as session:
        polymer = Polymer(
            common_name=body.common_name,
            abbreviation=body.abbreviation,
            smiles=body.smiles,
            canonical_smiles=canonical,
            source="user",
            tags=json.dumps(body.tags),
            created_at=now,
            updated_at=now,
        )
        session.add(polymer)
        await session.commit()
        await session.refresh(polymer)
        return _polymer_to_dict(polymer)


@router.put("/{polymer_id}")
async def update_polymer(polymer_id: int, body: PolymerUpdate):
    async with db_mod.async_session() as session:
        stmt = select(Polymer).where(Polymer.id == polymer_id)
        result = await session.execute(stmt)
        polymer = result.scalar_one_or_none()

        if not polymer:
            raise HTTPException(status_code=404, detail="Polymer not found")

        if body.common_name is not None:
            polymer.common_name = body.common_name
        if body.abbreviation is not None:
            polymer.abbreviation = body.abbreviation
        if body.smiles is not None:
            polymer.smiles = body.smiles
        if body.tags is not None:
            polymer.tags = json.dumps(body.tags)

        polymer.updated_at = datetime.now(timezone.utc).isoformat()

        await session.commit()
        await session.refresh(polymer)
        return _polymer_to_dict(polymer)


@router.delete("/{polymer_id}")
async def delete_polymer(polymer_id: int):
    async with db_mod.async_session() as session:
        stmt = select(Polymer).where(Polymer.id == polymer_id)
        result = await session.execute(stmt)
        polymer = result.scalar_one_or_none()

        if not polymer:
            raise HTTPException(status_code=404, detail="Polymer not found")

        await session.delete(polymer)
        await session.commit()
        return {"deleted": True}


def _polymer_to_dict(p: Polymer) -> dict:
    return {
        "id": p.id,
        "commonName": p.common_name,
        "abbreviation": p.abbreviation,
        "smiles": p.smiles,
        "canonical_smiles": p.canonical_smiles or "",
        "source": p.source,
        "tags": json.loads(p.tags) if p.tags else [],
    }
