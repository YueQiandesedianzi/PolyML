"""Shared Pydantic base model with camelCase alias support."""

from pydantic import BaseModel, ConfigDict


def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelModel(BaseModel):
    """Base model that accepts both snake_case and camelCase fields."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)
