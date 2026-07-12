"""Application Configuration"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


def _default_app_data_path() -> str:
    """Default app data path: ~/.PolyML"""
    return str(Path.home() / ".PolyML")


class Settings(BaseSettings):
    app_data_path: str = _default_app_data_path()
    backend_port: int = 18921
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    @property
    def projects_path(self) -> Path:
        return Path(self.app_data_path) / "projects"

    @property
    def polymers_db_path(self) -> Path:
        return Path(self.app_data_path) / "polymers.db"

    @property
    def config_path(self) -> Path:
        return Path(self.app_data_path) / "config.json"

    class Config:
        env_prefix = "POLYML_"


settings = Settings()

# Load LLM API key from config.json on disk if not set via environment
if not settings.llm_api_key and settings.config_path.exists():
    try:
        import json
        with open(settings.config_path, encoding="utf-8") as f:
            _cfg = json.load(f)
        settings.llm_api_key = _cfg.get("llm_api_key", "")
    except Exception:
        pass
