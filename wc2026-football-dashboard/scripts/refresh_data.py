"""Refresh the processed dashboard dataset from real raw CSV inputs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_pipeline import refresh_dashboard_data


def main() -> int:
    result = refresh_dashboard_data()
    print(result.get("message", "Refresh finished."))
    for warning in result.get("warnings", []):
        print(f"WARN: {warning}")
    for error in result.get("errors", []):
        print(f"ERROR: {error}")
    if result.get("path"):
        print(f"Processed dataset: {result['path']}")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
