"""Target-label construction for NASA C-MAPSS FD001."""

import pandas as pd


RUL_CAP = 125


def add_rul_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the trajectory data with fixed RUL labels."""
    labeled = frame.copy()

    maximum_cycle = (
        labeled.groupby("unit_id")["cycle"]
        .transform("max")
    )

    labeled["raw_rul"] = maximum_cycle - labeled["cycle"]
    labeled["target_rul"] = labeled["raw_rul"].clip(upper=RUL_CAP)
    labeled["failure_within_30"] = (labeled["raw_rul"] <= 30).astype(int)

    return labeled
