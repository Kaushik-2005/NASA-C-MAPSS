"""Fit and serialize the development-only temporal feature preprocessor."""

import json
from pathlib import Path

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_training_samples
from src.data.splits import build_engine_manifests
from src.features.build_features import build_feature_matrix
from src.features.preprocessing import FeaturePreprocessor


def materialize_development_preprocessor(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    artifact_path: Path = Path("models/feature_preprocessor_v1.joblib"),
    report_path: Path = Path("reports/feature-pipeline-v1.json"),
) -> None:
    """Fit filtering on development features and serialize the result."""
    labeled_train = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled_train["unit_id"].unique())
    samples = generate_training_samples(labeled_train, manifests["development"])
    development_features = build_feature_matrix(labeled_train, samples)

    preprocessor = FeaturePreprocessor(scale=False).fit(development_features)
    preprocessor.save(artifact_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "feature_schema_version": "v1",
                "fit_partition": "development",
                "development_samples": len(development_features),
                "raw_feature_count": development_features.shape[1],
                "selected_feature_count": len(preprocessor.selected_feature_names_),
                "variance_threshold": preprocessor.variance_threshold,
                "scaling": preprocessor.scale,
                "artifact": str(artifact_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    materialize_development_preprocessor()
    print("Development feature preprocessor written")
