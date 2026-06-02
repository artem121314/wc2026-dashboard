"""Real-data-only loading helpers for the dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline import (
    PROCESSED_DASHBOARD_PATH,
    load_processed_data,
    refresh_dashboard_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROCESSED_DASHBOARD_PATH


def load_player_data(processed_path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load dashboard-ready real data.

    No data is generated here. If the processed file is absent, an empty frame
    is returned so the app can show a clear refresh/data-missing message.
    """

    return load_processed_data(Path(processed_path))


def refresh_player_data() -> dict[str, object]:
    """Refresh the dashboard data from local real CSV inputs."""

    return refresh_dashboard_data()


if __name__ == "__main__":
    result = refresh_player_data()
    if result.get("success"):
        print(f"Wrote processed dashboard data to {PROCESSED_DATA_PATH}")
    else:
        print(result.get("message", "Refresh failed."))
