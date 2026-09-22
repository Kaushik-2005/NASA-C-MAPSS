import json
from pathlib import Path

from src.api.service import ModelService


def test_vercel_config_excludes_non_runtime_files() -> None:
    config = json.loads(Path("vercel.json").read_text(encoding="utf-8"))
    function_config = config["functions"]["api/**/*.py"]

    assert "docs/**" in function_config["excludeFiles"]
    assert "data/**" in function_config["excludeFiles"]
    assert "models/*_final_v1.joblib" in function_config["excludeFiles"]


def test_model_service_resolves_artifacts_independently_of_working_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    service = ModelService()

    assert type(service.model).__name__ == "XGBRegressor"
    assert service.feature_schema_version == "v1"
