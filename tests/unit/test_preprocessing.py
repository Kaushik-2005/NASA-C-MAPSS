import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS
from src.features.preprocessing import FeaturePreprocessor


def make_feature_frame(rows: int = 4) -> pd.DataFrame:
    values = np.arange(rows * len(FEATURE_COLUMNS), dtype=float).reshape(
        rows,
        len(FEATURE_COLUMNS),
    )
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame[FEATURE_COLUMNS[0]] = 1.0
    return frame


def test_preprocessor_filters_development_only_constant_features() -> None:
    features = make_feature_frame()
    preprocessor = FeaturePreprocessor(scale=False)

    transformed = preprocessor.fit_transform(features)

    assert FEATURE_COLUMNS[0] not in transformed.columns
    assert tuple(transformed.columns) == preprocessor.selected_feature_names_


def test_preprocessor_scales_when_requested() -> None:
    features = make_feature_frame()
    preprocessor = FeaturePreprocessor(scale=True)

    transformed = preprocessor.fit_transform(features)

    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert np.allclose(transformed.std(axis=0, ddof=0), 1.0)


def test_preprocessor_rejects_wrong_feature_order() -> None:
    features = make_feature_frame()
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(features)
    reordered = features[list(reversed(FEATURE_COLUMNS))]

    with pytest.raises(ValueError, match="v1 feature contract"):
        preprocessor.transform(reordered)


def test_preprocessor_save_load_preserves_transform(tmp_path) -> None:
    features = make_feature_frame()
    preprocessor = FeaturePreprocessor(scale=True).fit(features)
    path = tmp_path / "preprocessor.joblib"
    preprocessor.save(path)

    loaded = FeaturePreprocessor.load(path)

    pd.testing.assert_frame_equal(
        preprocessor.transform(features),
        loaded.transform(features),
    )
