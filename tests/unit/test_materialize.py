import json
from pathlib import Path

from src.data.materialize import materialize_manifests


def test_materialize_manifests_writes_reproducible_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "manifests"

    materialize_manifests(output_dir=output_dir)

    engines = json.loads((output_dir / "engine_manifests.json").read_text())
    samples = json.loads((output_dir / "sample_manifest.json").read_text())
    assert len(engines["development"]) == 80
    assert len(engines["validation"]) == 20
    assert samples["validation_samples"] == 80
    assert samples["final_test_samples"] == 100
