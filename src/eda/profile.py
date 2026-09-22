"""Generate a reproducible HTML profile for the FD001 training data."""

from html import escape
from pathlib import Path

import pandas as pd

from src.data.ingest import load_fd001_file

RAW_TRAIN = Path("data/raw/train_FD001.txt")
REPORT_PATH = Path("reports/fd001-profile.html")
VARIANCE_THRESHOLD = 1e-6


def build_profile(raw_path: Path = RAW_TRAIN) -> dict[str, object]:
    """Calculate Module 4 profile measurements from training data."""
    train = load_fd001_file(raw_path)
    development = train[train["unit_id"] % 5 != 0].copy()
    validation = train[train["unit_id"] % 5 == 0].copy()
    sensor_columns = [column for column in train if column.startswith("sensor_")]
    setting_columns = [column for column in train if column.startswith("op_setting_")]
    lifecycle = development.groupby("unit_id")["cycle"].max()
    variance = development[sensor_columns].var(ddof=0).sort_values()
    correlations = development[sensor_columns].corr()

    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(sensor_columns):
        for right in sensor_columns[left_index + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value):
                pairs.append({"sensor_a": left, "sensor_b": right, "correlation": float(value)})
    pairs.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)

    return {
        "train": train,
        "development": development,
        "validation": validation,
        "lifecycle": lifecycle,
        "variance": variance,
        "settings": development[setting_columns].describe().T,
        "top_correlations": pd.DataFrame(pairs[:20]),
        "excluded_features": variance[variance <= VARIANCE_THRESHOLD].index.tolist(),
    }


def write_profile(report_path: Path = REPORT_PATH) -> None:
    """Write the HTML profile report to disk."""
    profile = build_profile()
    train = profile["train"]
    development = profile["development"]
    validation = profile["validation"]
    lifecycle = profile["lifecycle"]
    variance = profile["variance"]
    settings = profile["settings"]
    top_correlations = profile["top_correlations"]
    excluded = profile["excluded_features"]

    summary = pd.DataFrame(
        [
            {"metric": "training rows", "value": len(train)},
            {"metric": "training engines", "value": train["unit_id"].nunique()},
            {"metric": "development rows", "value": len(development)},
            {"metric": "development engines", "value": development["unit_id"].nunique()},
            {"metric": "validation rows", "value": len(validation)},
            {"metric": "validation engines", "value": validation["unit_id"].nunique()},
            {"metric": "lifecycle minimum", "value": lifecycle.min()},
            {"metric": "lifecycle median", "value": lifecycle.median()},
            {"metric": "lifecycle maximum", "value": lifecycle.max()},
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>FD001 profile</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;line-height:1.4}} table{{border-collapse:collapse;margin:1rem 0}} th,td{{border:1px solid #ccc;padding:.35rem .55rem}} th{{background:#eee}} code{{background:#f4f4f4;padding:.1rem .25rem}}</style>
</head>
<body>
<h1>NASA C-MAPSS FD001 profile</h1>
<p>Generated from training data. Variance and correlation analysis uses development engines only (<code>unit_id % 5 != 0</code>).</p>
<h2>Summary</h2>{summary.to_html(index=False)}
<h2>Operating settings</h2>{settings.to_html()}
<h2>Sensor variance</h2>{variance.rename("variance").to_frame().to_html()}
<p>Excluded at variance threshold {VARIANCE_THRESHOLD}: {escape(", ".join(excluded))}</p>
<h2>Strongest sensor correlations</h2>{top_correlations.to_html(index=False) if not top_correlations.empty else "<p>No finite correlation pairs.</p>"}
<h2>Limitations</h2>
<ul><li>FD001 has one operating condition and one fault mode.</li><li>The data is simulated.</li><li>Correlation indicates association, not physical causality.</li><li>Findings do not automatically generalize to other C-MAPSS subsets or real engines.</li></ul>
</body></html>
"""
    report_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    write_profile()
    print(f"Wrote {REPORT_PATH}")
