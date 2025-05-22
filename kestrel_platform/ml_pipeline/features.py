import pandas as pd
import numpy as np

def compute_rolling_crew_score(crew_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw maritime rolling metrics over historical voyages.
    """
    crew_history_df = crew_history_df.sort_values(by=["vessel_id", "timestamp"])
    crew_history_df["rolling_incident_average"] = (
        crew_history_df.groupby("vessel_id")["incidents"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    return crew_history_df
