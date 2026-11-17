#!/usr/bin/env python3
"""make.fun gta6-twitch-ccu-forecast-01 entry.

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input/), in name order, and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output/) with five quantiles of
the Twitch viewer count for each of hours 2 to 24 of the release day.
Standard library only, deterministic, no network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import baseline_forecast, encode_forecast, forecast_case, hour1_print

DEFAULT_INPUT_DIR = "/input"
DEFAULT_OUTPUT_DIR = "/output"


def resolve_dir(variable: str, default: str) -> Path:
    """The directory named by an environment variable, or the default when unset or empty."""
    return Path(os.environ.get(variable) or default)


def load_case(path: Path) -> dict:
    """The case's JSON, or an empty case when the file cannot be read as one."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_forecast(case: dict) -> dict:
    """The model's forecast, or the house baseline when the model cannot run on this case.

    An invalid or missing output costs far more than a baseline forecast, so no
    failure inside the model is allowed to escape.
    """
    try:
        return forecast_case(case)
    except Exception:
        return baseline_forecast(hour1_print(case))


def case_files(input_dir: Path) -> list[Path]:
    """Every *.in file of the input directory, in name order."""
    return sorted(input_dir.glob("*.in"), key=lambda path: path.name)


def main() -> int:
    input_dir = resolve_dir("MAKEFUN_INPUT_DIR", DEFAULT_INPUT_DIR)
    output_dir = resolve_dir("MAKEFUN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_path in case_files(input_dir):
        forecast = safe_forecast(load_case(case_path))
        (output_dir / f"{case_path.stem}.out").write_bytes(encode_forecast(forecast))
    return 0


if __name__ == "__main__":
    sys.exit(main())
