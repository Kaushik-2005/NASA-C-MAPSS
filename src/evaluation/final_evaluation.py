"""One-time official FD001 test evaluation and explainability artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from src.data.ingest import load_fd001_file, load_rul_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_final_test_samples, generate_training_samples
from src.data.splits import build_engine_manifests
from src.evaluation.metrics import regression_metrics
from src.features.build_features import build_feature_matrix
from src.features.preprocessing import FeaturePreprocessor
from src.training.trees import build_xgboost

RUL_CAP = 125


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the C-MAPSS asymmetric score for capped RUL values."""
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true_values.size == 0:
        raise ValueError("Score inputs cannot be empty")

    errors = predicted_values - true_values
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(penalties.sum())


def _load_candidate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen candidate configuration not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("feature_schema_version") != "v1":
        raise ValueError("Final evaluation requires feature schema version v1")
    return candidate


def _prepare_data(
    train_path: Path,
    test_path: Path,
    rul_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, FeaturePreprocessor]:
    labeled_train = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled_train["unit_id"].unique())
    training_samples = generate_training_samples(
        labeled_train,
        manifests["development"],
    )

    test_trajectories = load_fd001_file(test_path)
    test_samples = generate_final_test_samples(test_trajectories)
    official_rul = load_rul_file(rul_path)

    expected_ids = np.arange(1, 101)
    actual_ids = test_samples["unit_id"].to_numpy(dtype=int)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError("Final test samples must contain engines 1 through 100 in order")

    test_samples = test_samples.copy()
    test_samples["raw_rul"] = official_rul.to_numpy(dtype=int)
    test_samples["target_rul"] = test_samples["raw_rul"].clip(upper=RUL_CAP)

    training_raw = build_feature_matrix(labeled_train, training_samples)
    test_raw = build_feature_matrix(test_trajectories, test_samples)
    preprocessor = FeaturePreprocessor(scale=False).fit(training_raw)

    return (
        preprocessor.transform(training_raw).to_numpy(dtype=float),
        preprocessor.transform(test_raw).to_numpy(dtype=float),
        training_samples["target_rul"].to_numpy(dtype=float),
        test_samples["target_rul"].to_numpy(dtype=float),
        test_samples,
        preprocessor,
    )


def _shap_artifacts(
    model: Any,
    X_test: np.ndarray,
    test_samples: pd.DataFrame,
    feature_names: tuple[str, ...],
    report_dir: Path,
) -> dict[str, Any]:
    plt.switch_backend("Agg")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim == 3:
        shap_array = shap_array[0]
    if shap_array.shape != X_test.shape:
        raise ValueError("SHAP output shape does not match the feature matrix")

    mean_absolute = np.abs(shap_array).mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": mean_absolute,
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance_path = report_dir / "shap-global-importance.csv"
    importance.to_csv(importance_path, index=False)

    plt.figure(figsize=(10, 7))
    top = importance.head(20).sort_values("mean_absolute_shap")
    plt.barh(top["feature"], top["mean_absolute_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP importance — final XGBoost")
    plt.tight_layout()
    global_plot_path = report_dir / "shap-global-importance.png"
    plt.savefig(global_plot_path, dpi=150)
    plt.close()

    lifecycle_order = test_samples["cycle"].sort_values().index
    selected_positions = [
        int(lifecycle_order[0]),
        int(lifecycle_order[len(lifecycle_order) // 2]),
        int(lifecycle_order[-1]),
    ]
    local_records: list[dict[str, Any]] = []
    for position in selected_positions:
        contributions = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": shap_array[position],
            }
        )
        contributions["absolute"] = contributions["shap_value"].abs()
        top_local = contributions.nlargest(10, "absolute")
        local_records.append(
            {
                "row": position,
                "unit_id": int(test_samples.iloc[position]["unit_id"]),
                "cycle": int(test_samples.iloc[position]["cycle"]),
                "top_contributions": top_local[
                    ["feature", "shap_value"]
                ].to_dict(orient="records"),
            }
        )
        plot_data = top_local.sort_values("shap_value")
        plt.figure(figsize=(9, 5))
        colors = ["#377eb8" if value < 0 else "#e41a1c" for value in plot_data["shap_value"]]
        plt.barh(plot_data["feature"], plot_data["shap_value"], color=colors)
        plt.xlabel("SHAP contribution to predicted RUL")
        plt.title(f"Local SHAP explanation — engine {int(test_samples.iloc[position]['unit_id'])}")
        plt.tight_layout()
        plt.savefig(report_dir / f"shap-local-engine-{int(test_samples.iloc[position]['unit_id']):03d}.png", dpi=150)
        plt.close()

    local_path = report_dir / "shap-local-explanations.json"
    local_path.write_text(json.dumps(local_records, indent=2) + "\n", encoding="utf-8")
    return {
        "global_importance": str(importance_path),
        "global_plot": str(global_plot_path),
        "local_explanations": str(local_path),
        "local_rows": selected_positions,
    }


def _write_model_card(path: Path, metrics: dict[str, Any], candidate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# EngineGuard Model Card",
                "",
                "## Model",
                "",
                "Frozen XGBoost Regressor trained on all 80 FD001 development engines.",
                "",
                f"- Feature schema: `{candidate['feature_schema_version']}`",
                f"- Random seed: `{candidate['random_seed']}`",
                f"- Final test RMSE: `{metrics['regression']['rmse']:.4f}`",
                f"- Final test MAE: `{metrics['regression']['mae']:.4f}`",
                "",
                "## Intended use",
                "",
                "Portfolio demonstration of RUL prediction on simulated NASA C-MAPSS FD001 data. It is not certified for aircraft-maintenance decisions.",
                "",
                "## Limitations",
                "",
                "FD001 contains one operating condition and one fault mode. SHAP explains model behavior for these inputs; it does not establish physical sensor causality.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_final_evaluation(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    test_path: Path = Path("data/raw/test_FD001.txt"),
    rul_path: Path = Path("data/raw/RUL_FD001.txt"),
    candidate_path: Path = Path("configs/xgboost-candidate-v1.json"),
    model_path: Path = Path("models/xgboost_final_v1.joblib"),
    preprocessor_path: Path = Path("models/feature_preprocessor_final_v1.joblib"),
    prediction_path: Path = Path("reports/final-test-predictions.csv"),
    metrics_path: Path = Path("reports/final-evaluation.json"),
    report_path: Path = Path("reports/final-evaluation.md"),
    model_card_path: Path = Path("docs/model-card.md"),
    explainability_dir: Path = Path("reports/final-explainability"),
) -> dict[str, Any]:
    """Run the official holdout evaluation once using the frozen candidate."""
    if metrics_path.exists():
        raise FileExistsError(
            "Final evaluation artifacts already exist; do not overwrite frozen evidence."
        )

    candidate = _load_candidate(candidate_path)
    explainability_dir.mkdir(parents=True, exist_ok=True)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_samples,
        preprocessor,
    ) = _prepare_data(train_path, test_path, rul_path)

    model = build_xgboost().set_params(**candidate["best_params"])
    model.fit(X_train, y_train)
    raw_predictions = np.asarray(model.predict(X_test), dtype=float)
    predictions = np.clip(raw_predictions, 0, RUL_CAP)

    if len(predictions) != 100:
        raise ValueError(f"Expected exactly 100 predictions, received {len(predictions)}")
    if not np.isfinite(predictions).all() or not ((predictions >= 0) & (predictions <= RUL_CAP)).all():
        raise ValueError("Final predictions must be finite and bounded between 0 and 125")

    regression = regression_metrics(y_test, predictions)
    errors = predictions - y_test
    within = {
        f"within_{tolerance}": int(np.sum(np.abs(errors) <= tolerance))
        for tolerance in (10, 20, 40)
    }
    bands = {
        "0-30": y_test <= 30,
        "31-60": (y_test >= 31) & (y_test <= 60),
        ">60": y_test > 60,
    }
    band_metrics = {
        name: {
            "count": int(mask.sum()),
            **regression_metrics(y_test[mask], predictions[mask]),
        }
        for name, mask in bands.items()
        if mask.any()
    }

    prediction_frame = pd.DataFrame(
        {
            "unit_id": test_samples["unit_id"].astype(int),
            "cycle": test_samples["cycle"].astype(int),
            "official_raw_rul": test_samples["raw_rul"].astype(int),
            "target_rul": y_test,
            "raw_prediction": raw_predictions,
            "predicted_rul": predictions,
            "error": errors,
        }
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)

    under = prediction_frame[prediction_frame["error"] < 0].nsmallest(5, "error")
    over = prediction_frame[prediction_frame["error"] > 0].nlargest(5, "error")
    top_errors = {
        "underpredictions": under.to_dict(orient="records"),
        "overpredictions": over.to_dict(orient="records"),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    preprocessor.save(preprocessor_path)
    shap_metadata = _shap_artifacts(
        model,
        X_test,
        test_samples,
        tuple(preprocessor.selected_feature_names_ or ()),
        explainability_dir,
    )

    metrics: dict[str, Any] = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation_boundary": "official test holdout, one-time evaluation",
        "feature_schema_version": candidate["feature_schema_version"],
        "candidate_configuration": candidate,
        "prediction_count": len(predictions),
        "prediction_range": [float(predictions.min()), float(predictions.max())],
        "regression": regression,
        "nasa_asymmetric_score": nasa_asymmetric_score(y_test, predictions),
        "within_tolerance": within,
        "error_bands": band_metrics,
        "top_errors": top_errors,
        "artifacts": {
            "predictions": str(prediction_path),
            "model": str(model_path),
            "preprocessor": str(preprocessor_path),
            **shap_metadata,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_report(metrics, report_path)
    _write_model_card(model_card_path, metrics, candidate)

    mlflow.set_experiment("engineguard-final-evaluation")
    with mlflow.start_run(run_name="fd001-official-test-final"):
        mlflow.log_params(
            {
                "dataset": "NASA C-MAPSS FD001",
                "feature_schema_version": candidate["feature_schema_version"],
                "evaluation_boundary": "official test holdout",
                "prediction_count": len(predictions),
            }
        )
        mlflow.log_metrics(
            {
                "test_rmse": regression["rmse"],
                "test_mae": regression["mae"],
                "test_r_squared": regression["r_squared"],
                "nasa_asymmetric_score": metrics["nasa_asymmetric_score"],
            }
        )
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(model_path))

    return metrics


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_report(metrics: dict[str, Any], report_path: Path) -> None:
    regression = metrics["regression"]
    lines = [
        "# Final FD001 Evaluation",
        "",
        "This is the one-time evaluation of the frozen XGBoost candidate on the official FD001 test holdout. No test result was used for tuning.",
        "",
        f"- Predictions: {metrics['prediction_count']}",
        f"- Prediction range: {metrics['prediction_range']}",
        f"- RMSE: {regression['rmse']:.4f}",
        f"- MAE: {regression['mae']:.4f}",
        f"- R-squared: {regression['r_squared']:.4f}",
        f"- NASA asymmetric score: {metrics['nasa_asymmetric_score']:.4f}",
        "",
        "## Accuracy windows",
        "",
        f"- Within 10 cycles: {metrics['within_tolerance']['within_10']}",
        f"- Within 20 cycles: {metrics['within_tolerance']['within_20']}",
        f"- Within 40 cycles: {metrics['within_tolerance']['within_40']}",
        "",
        "## Error bands",
        "",
        "| True target RUL band | Count | RMSE | MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for band, values in metrics["error_bands"].items():
        lines.append(f"| {band} | {values['count']} | {values['rmse']:.4f} | {values['mae']:.4f} |")
    lines.extend(
        [
            "",
            "## Explainability",
            "",
            "Global and three local SHAP explanations were generated. SHAP describes model behavior for the evaluated feature rows; it does not prove physical sensor causality.",
            "",
            "## Limitations",
            "",
            "FD001 is simulated, has one operating condition and one fault mode, and does not establish performance on other engines or operating regimes.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_final_evaluation()
    print("Final evaluation written")
