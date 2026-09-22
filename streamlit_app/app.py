"""Interactive EngineGuard portfolio demonstration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_app.api_client import ApiClientError, call_api
except ModuleNotFoundError:
    # Direct Streamlit execution may put only this directory on sys.path.
    from api_client import ApiClientError, call_api

SUMMARY_PATH = Path(__file__).resolve().parent / "assets" / "eda_summary.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://nasa-c-mapss.vercel.app"


def _api_url() -> str:
    """Read the API URL from deployment secrets or local environment."""
    try:
        configured = st.secrets.get("API_BASE_URL")
    except (FileNotFoundError, KeyError):
        configured = None
    return str(configured or os.getenv("API_BASE_URL", DEFAULT_API_URL))


def _load_doc(relative_path: str) -> str | None:
    """Load a maintained project document when it is available in the deployment."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _observation(
    cycle: int, sensor_2: float, sensor_11: float, sensor_20: float
) -> dict[str, float | int]:
    observation: dict[str, float | int] = {
        "cycle": cycle,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
    }
    observation.update({f"sensor_{index}": 0.0 for index in range(1, 22)})
    observation.update({"sensor_2": sensor_2, "sensor_11": sensor_11, "sensor_20": sensor_20})
    return observation


def _prediction_page() -> None:
    st.subheader("Test the deployed model")
    st.caption("This page sends the history to Vercel; no second model is loaded in Streamlit.")
    with st.expander("What is being tested?", expanded=True):
        st.write(
            "The API validates 20 ordered observations, builds the shared temporal feature row, "
            "clips RUL to 0-125 cycles, and maps it to deterministic maintenance risk."
        )
    col1, col2, col3 = st.columns(3)
    sensor_2 = col1.slider("Sensor 2", -5.0, 5.0, 0.0)
    sensor_11 = col2.slider("Sensor 11", -5.0, 5.0, 0.0)
    sensor_20 = col3.slider("Sensor 20", -5.0, 5.0, 0.0)
    unit_id = st.text_input("Engine ID", "demo-engine-001")
    history = [_observation(cycle, sensor_2, sensor_11, sensor_20) for cycle in range(1, 21)]

    if st.button("Predict RUL", type="primary"):
        try:
            result = call_api(
                _api_url(),
                "/v1/predict",
                {"unit_id": unit_id, "observations": history},
            )
        except ApiClientError as error:
            st.error(str(error))
            return
        left, right = st.columns(2)
        left.metric("Predicted RUL", f"{float(result['predicted_rul']):.1f} cycles")
        right.metric("Risk level", str(result["risk_level"]).upper())
        with st.expander("Traceability details"):
            st.json(result)


def _eda_page() -> None:
    st.subheader("FD001 exploratory analysis")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    left, middle, right = st.columns(3)
    left.metric("Development engines", summary["development_engine_count"])
    middle.metric("Validation engines", summary["validation_engine_count"])
    right.metric("Minimum history", "20 cycles")

    lifecycle = pd.Series(summary["lifecycle_lengths"], name="cycles")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.write("Development engine lifecycle distribution")
        bins = pd.cut(lifecycle, bins=8)
        histogram = bins.value_counts().sort_index()
        histogram_frame = pd.DataFrame(
            {"Lifecycle range": histogram.index.astype(str), "Engines": histogram.values}
        ).set_index("Lifecycle range")
        st.bar_chart(histogram_frame)
    with chart_right:
        st.write("Lifecycle summary")
        st.dataframe(
            pd.DataFrame(
                {
                    "Statistic": ["Minimum", "Median", "Mean", "Maximum"],
                    "Cycles": [
                        int(lifecycle.min()),
                        int(lifecycle.median()),
                        round(float(lifecycle.mean()), 1),
                        int(lifecycle.max()),
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.write("Representative sensor trajectories")
    st.info(
        "Sensors 2, 11, and 20 were selected because they showed useful variation and "
        "degradation behavior in development analysis."
    )
    selected_sensor = st.selectbox("Trajectory sensor", summary["selected_sensors"])
    trajectory_choice = st.selectbox("Representative engine", ["shortest", "median", "longest"])
    trajectory = summary["trajectories"][trajectory_choice]
    trajectory_frame = pd.DataFrame(
        {
            "cycle": trajectory["cycle"],
            **{sensor: trajectory["values"][sensor] for sensor in summary["selected_sensors"]},
        }
    ).set_index("cycle")
    st.line_chart(trajectory_frame, height=320)
    st.caption(
        f"Engine {trajectory['unit_id']} - {trajectory_choice} lifecycle; selected sensor: {selected_sensor}"
    )

    st.write("Operating settings across a representative engine")
    settings = summary["operating_settings"]
    settings_frame = pd.DataFrame({"cycle": settings["cycle"], **settings["values"]}).set_index(
        "cycle"
    )
    st.line_chart(settings_frame, height=260)
    st.caption(
        f"Engine {settings['unit_id']} - operating settings remain distinct from sensor degradation signals"
    )

    st.write("Selected-sensor correlation matrix")
    correlation = pd.DataFrame(summary["selected_sensor_correlations"])
    st.dataframe(
        correlation.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
        use_container_width=True,
    )

    st.write("Development-only sensor variance")
    variance = pd.Series(summary["sensor_variance"], name="variance").sort_values(ascending=False)
    st.bar_chart(variance, height=360)
    st.caption(
        "Variance was calculated using development engines only; zero and near-zero sensors were excluded from the feature contract."
    )
    st.write("Excluded near-constant features:", ", ".join(summary["excluded_features"]))


def _modeling_page() -> None:
    st.subheader("Modeling choices and measured evidence")
    st.write("All comparisons use the fixed engine-level split and the same 80 validation samples.")
    comparison = pd.DataFrame(
        [
            {
                "Model": "Median baseline",
                "Validation RMSE": 50.2057,
                "Validation MAE": 44.7375,
                "Role": "Reference",
            },
            {
                "Model": "Ridge",
                "Validation RMSE": 20.4961,
                "Validation MAE": 17.0469,
                "Role": "Linear baseline",
            },
            {
                "Model": "Random Forest",
                "Validation RMSE": 19.2289,
                "Validation MAE": 15.0383,
                "Role": "Bagging",
            },
            {
                "Model": "XGBoost",
                "Validation RMSE": 16.4279,
                "Validation MAE": 12.3607,
                "Role": "Selected candidate",
            },
        ]
    )
    st.dataframe(
        comparison.style.highlight_min(
            subset=["Validation RMSE", "Validation MAE"], color="#d9f2e6"
        ).format({"Validation RMSE": "{:.2f}", "Validation MAE": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Development engines", "80")
    col2.metric("Validation samples", "80")
    col3.metric("Feature schema", "v1")
    with st.expander("Key decisions", expanded=True):
        st.markdown(
            """
            - Engine-level partitions prevent rows from the same engine crossing boundaries.
            - RUL is capped at 125 cycles to reduce the influence of early-life labels.
            - Temporal features use only current and earlier observations.
            - XGBoost was selected from validation evidence, not official test labels.
            - SHAP explains model behavior; it does not prove physical sensor causality.
            """
        )


def _project_page() -> None:
    st.subheader("Project definition and architecture")
    st.write(
        "EngineGuard predicts remaining useful life in cycles for NASA C-MAPSS FD001 "
        "and maps the bounded prediction to a maintenance-risk level."
    )
    doc = _load_doc("docs/problem-definition.md")
    if doc:
        with st.expander("Read the maintained problem definition"):
            st.markdown(doc)
    architecture = _load_doc("docs/architecture.md")
    if architecture:
        with st.expander("Read the maintained architecture"):
            st.markdown(architecture)
    st.write("Fixed product contract")
    st.dataframe(
        pd.DataFrame(
            [
                {"Contract": "Dataset", "Decision": "NASA C-MAPSS FD001 only"},
                {"Contract": "Target", "Decision": "min(max_training_cycle - current_cycle, 125)"},
                {"Contract": "Risk", "Decision": "0-30 critical, 31-60 warning, 61-125 healthy"},
                {"Contract": "Split", "Decision": "Deterministic engine-level partitions"},
                {"Contract": "Minimum history", "Decision": "20 strictly increasing cycles"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _data_page() -> None:
    st.subheader("Data, labels, and leakage controls")
    st.write(
        "The project keeps raw data immutable, validates its schema before feature generation, "
        "and uses engine-level partitions instead of row-level random splits."
    )
    doc = _load_doc("docs/data-source.md")
    if doc:
        with st.expander("Data source"):
            st.markdown(doc)
    st.write("Partition policy")
    st.code(
        "validation:        unit_id % 5 == 0\n"
        "development:       unit_id % 5 != 0\n"
        "initial training:  unit_id <= 75 and unit_id % 5 != 0\n"
        "incremental data:  unit_id > 75 and unit_id % 5 != 0",
        language="text",
    )
    with st.expander("Why this prevents leakage", expanded=True):
        st.markdown(
            """
            - Rows from one engine never appear in both training and validation.
            - Future cycles are never used to build an earlier prediction row.
            - Lifecycle length, raw RUL, and target RUL are labels, never input features.
            - Official test labels remain isolated until the final evaluation.
            """
        )
    leakage = _load_doc("docs/leakage-analysis.md")
    if leakage:
        with st.expander("Read the maintained leakage analysis"):
            st.markdown(leakage)


def _features_page() -> None:
    st.subheader("Temporal feature engineering")
    st.write(
        "The same feature builder is used during training, validation, batch scoring, and API inference."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature family": "Current value",
                    "Windows": "Current cycle",
                    "Purpose": "Instantaneous state",
                },
                {
                    "Feature family": "Rolling mean",
                    "Windows": "5, 10, 20 cycles",
                    "Purpose": "Smoothed level",
                },
                {
                    "Feature family": "Rolling standard deviation",
                    "Windows": "5, 10, 20 cycles",
                    "Purpose": "Local volatility",
                },
                {
                    "Feature family": "Delta",
                    "Windows": "Current minus 5 cycles ago",
                    "Purpose": "Recent change",
                },
                {
                    "Feature family": "Slope",
                    "Windows": "Last 20 cycles",
                    "Purpose": "Trend direction",
                },
                {
                    "Feature family": "History metadata",
                    "Windows": "Cycle and capped count",
                    "Purpose": "Temporal context",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Feature safety rules", expanded=True):
        st.markdown(
            """
            - Minimum history is 20 observations.
            - Features use only current and earlier observations.
            - Variance filtering is fitted on development data only.
            - Feature order and schema version `v1` are persisted with the preprocessor.
            """
        )
    design = _load_doc("docs/design.md")
    if design:
        with st.expander("Read the current design document"):
            st.markdown(design)


def _evaluation_page() -> None:
    st.subheader("Evaluation and limitations")
    st.write(
        "Final evaluation is a one-time holdout measurement. The official test set is not used "
        "for feature selection, tuning, or promotion decisions."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Official test engines", "100")
    col2.metric("Prediction coverage", "100 / 100")
    col3.metric("RUL bounds", "0-125")
    evaluation = _load_doc("reports/final-evaluation.md")
    if evaluation:
        with st.expander("Read the final evaluation report", expanded=True):
            st.markdown(evaluation)
    else:
        st.info(
            "The full generated evaluation report is kept outside the lightweight UI deployment bundle."
        )
    with st.expander("Interpretation limits", expanded=True):
        st.markdown(
            """
            - FD001 has one operating condition and one fault mode.
            - Validation results and final-test results are separate evidence.
            - SHAP explains model behavior for evaluated rows, not physical causality.
            - This portfolio system is not certified for real aircraft maintenance.
            """
        )


def _reproduction_page() -> None:
    st.subheader("Reproduce the project")
    st.write(
        "The project is organized as a reproducible pipeline rather than a notebook-only experiment."
    )
    st.code(
        "# Install\n"
        'python -m pip install -e ".[training,monitoring,ui,dev]"\n\n'
        "# Validate data\n"
        "python -m dvc repro validate_raw\n\n"
        "# Train and evaluate\n"
        "python -m dvc repro train\n"
        "python -m src.evaluation.final_evaluation\n\n"
        "# Test and serve\n"
        "python -m pytest -q\n"
        "python -m uvicorn src.api.app:app --reload\n\n"
        "# Run this UI\n"
        "python -m streamlit run streamlit_app/app.py",
        language="powershell",
    )
    st.write("Evidence locations")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Evidence": "Final evaluation",
                    "Location": "reports/final-evaluation.json and .md",
                },
                {"Evidence": "Monitoring", "Location": "reports/monitoring-summary.md"},
                {"Evidence": "Retraining", "Location": "reports/retraining/"},
                {"Evidence": "Deployment", "Location": "reports/deployment-load-test.md"},
                {"Evidence": "Tests", "Location": "tests/ and CI workflow"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _overview_page() -> None:
    st.subheader("EngineGuard")
    st.write("A NASA C-MAPSS FD001 remaining-useful-life prediction project.")
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Validation RMSE", "16.43 cycles")
    metric_middle.metric("Prediction range", "0-125 cycles")
    metric_right.metric("API deployment", "Vercel")
    st.divider()
    st.markdown(
        """
        **Workflow:** ingest -> validate -> label -> engineer temporal features -> train ->
        evaluate -> serve -> monitor -> retrain safely.

        **Important limitation:** FD001 is simulated with one operating condition and one
        fault mode. This demonstration is not certified for aircraft-maintenance decisions.
        """
    )
    st.write("System flow")
    st.code(
        "Raw FD001 -> validation -> labels/splits -> temporal features -> XGBoost -> FastAPI/Vercel\n"
        "                                                                    |\n"
        "                                             monitoring -> retraining -> gated promotion/rollback",
        language="text",
    )


def _mlops_page() -> None:
    st.subheader("MLOps decisions")
    st.write(
        "The project separates service health, data drift, model performance, and promotion decisions."
    )
    alert_left, alert_right = st.columns(2)
    alert_left.metric("Drift scenario", "Detected")
    alert_right.metric("Challenger decision", "Promoted")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Stage": "Monitoring",
                    "Evidence": "PSI >= 0.20 in consecutive batches",
                    "Action": "Trigger review",
                },
                {
                    "Stage": "Validation",
                    "Evidence": "RMSE improvement >= 3% and MAE no worse",
                    "Action": "Continue gates",
                },
                {
                    "Stage": "Serving",
                    "Evidence": "p95 latency < 100 ms",
                    "Action": "Continue gates",
                },
                {
                    "Stage": "Promotion",
                    "Evidence": "Lineage and tests complete",
                    "Action": "Move champion alias",
                },
                {
                    "Stage": "Rollback",
                    "Evidence": "Previous alias and artifact retained",
                    "Action": "Restore without retraining",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("What the monitoring simulation proved"):
        st.write(
            "Batches 6-8 contained sensor drift, batch 9 was quarantined for a missing sensor, and batch 10 demonstrated recovery."
        )
        st.write(
            "Drift is evidence of distribution change, not proof that a model should be promoted."
        )


def _sidebar() -> str:
    with st.sidebar:
        st.header("EngineGuard")
        st.caption("NASA C-MAPSS FD001 - schema v1")
        selected_page = st.radio(
            "Navigate",
            [
                "Project",
                "Data",
                "EDA",
                "Features",
                "Modeling",
                "Test model",
                "Evaluation",
                "MLOps",
                "Reproduction",
            ],
        )
        st.divider()
        st.write("Live service")
        if st.button("Check API readiness", use_container_width=True):
            try:
                ready = call_api(_api_url(), "/ready")
                st.success(f"Ready - {ready.get('model_version', 'unknown')}")
            except ApiClientError as error:
                st.error(str(error))
        st.divider()
        st.caption("Portfolio demonstration only; not certified aircraft-maintenance software.")
    return selected_page


def main() -> None:
    st.set_page_config(page_title="EngineGuard", page_icon="E", layout="wide")
    selected_page = _sidebar()
    st.title("EngineGuard: Turbofan RUL Prediction")
    st.caption(f"Prediction API: {_api_url()}")
    if selected_page == "Project":
        _project_page()
    elif selected_page == "Data":
        _data_page()
    elif selected_page == "EDA":
        _eda_page()
    elif selected_page == "Features":
        _features_page()
    elif selected_page == "Modeling":
        _modeling_page()
    elif selected_page == "Test model":
        _prediction_page()
    elif selected_page == "Evaluation":
        _evaluation_page()
    elif selected_page == "MLOps":
        _mlops_page()
    elif selected_page == "Reproduction":
        _reproduction_page()
    else:
        _overview_page()


if __name__ == "__main__":
    main()
