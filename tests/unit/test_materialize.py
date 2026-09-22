import json
from pathlib import Path

from src.data.materialize import materialize_manifests
from src.features.materialize import materialize_development_preprocessor


def test_materialize_manifests_writes_reproducible_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "manifests"

    materialize_manifests(output_dir=output_dir)

    engines = json.loads((output_dir / "engine_manifests.json").read_text())
    samples = json.loads((output_dir / "sample_manifest.json").read_text())
    assert len(engines["development"]) == 80
    assert len(engines["validation"]) == 20
    assert samples["validation_samples"] == 80
    assert samples["final_test_samples"] == 100


def test_materialize_development_preprocessor_writes_artifact_and_report(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "feature-preprocessor.joblib"
    report_path = tmp_path / "feature-pipeline.json"

    materialize_development_preprocessor(
        artifact_path=artifact_path,
        report_path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert artifact_path.exists()
    assert report["feature_schema_version"] == "v1"
    assert report["fit_partition"] == "development"
    assert report["selected_feature_count"] > 0
