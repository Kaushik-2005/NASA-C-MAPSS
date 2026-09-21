"""Typed configuration primitives shared by the EngineGuard pipeline."""

from pathlib import Path

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Immutable project constants that protect the fixed roadmap contract."""

    dataset: str = "fd001"
    random_seed: int = 42
    rul_cap: int = Field(default=125, gt=0)
    minimum_history_cycles: int = Field(default=20, gt=0)
    feature_schema_version: str = "v1"
    raw_data_dir: Path = Path("data/raw")


PROJECT_CONFIG = ProjectConfig()
