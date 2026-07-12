"""Seed built-in polymer name → SMILES database"""

import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import text


async def seed_polymers(db_session, polymers_json_path: str | None = None):
    """Seed the polymer database with built-in entries if empty."""
    from sqlalchemy import select, func
    from db.models import Polymer

    result = await db_session.execute(select(func.count(Polymer.id)))
    count = result.scalar()

    if count > 0:
        return

    if polymers_json_path is None:
        polymers_json_path = Path(__file__).parent.parent / "data" / "polymers.json"

    with open(polymers_json_path, encoding="utf-8") as f:
        polymers = json.load(f)

    now = datetime.now(timezone.utc).isoformat()

    for item in polymers:
        polymer = Polymer(
            common_name=item["common_name"],
            abbreviation=item.get("abbreviation", ""),
            smiles=item["smiles"],
            canonical_smiles=item.get("canonical_smiles", ""),
            source="builtin",
            tags=json.dumps(item.get("tags", [])),
            created_at=now,
            updated_at=now,
        )
        db_session.add(polymer)

    await db_session.commit()
    print(f"[Seed] Inserted {len(polymers)} built-in polymers")
