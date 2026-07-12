"""SQLAlchemy ORM Models"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from db.database import Base


class Polymer(Base):
    __tablename__ = "polymers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    common_name = Column(String(200), nullable=False, index=True)
    abbreviation = Column(String(50), index=True)
    smiles = Column(String(2000), nullable=False)
    canonical_smiles = Column(String(2000))
    source = Column(String(20), default="builtin")  # 'builtin' or 'user'
    tags = Column(Text)  # JSON array as text
    created_at = Column(String(30), nullable=False)
    updated_at = Column(String(30), nullable=False)
