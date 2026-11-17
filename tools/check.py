#!/usr/bin/env python3
"""Validate forecast files against the contest's output rules.

For every <id>.out in the output directory with a matching <id>.in in the
input directory: exactly one key `hours`, 23 rows for hours 2..24 each once,
five finite quantiles each above 0 and at most 1e9, non-decreasing, and the
file at most 65,536 bytes. Prints each case's median curve as a ratio to the
hour-1 print. Exits 1 on any failure.

Usage: python tools/check.py <input dir> <output dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import model  # noqa: E402

MAX_BYTES = 65_536
SUMMARY_HOURS = (2, 4, 6, 8, 12, 16, 20, 24)


def validate(forecast: object) -> list[str]:
    """Every rule the forecast breaks, as messages; empty when it is valid."""
    problems: list[str] = []
    if not isinstance(forecast, dict) or set(forecast) != {"hours"}:
        return ["top level must be an object with exactly one key, hours"]
    rows = forecast["hours"]
    if not isinstance(rows, list) or len(rows) != 23:
        return ["hours must be a list of exactly 23 rows"]
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"hour", "q"}:
            problems.append(f"row must have exactly the keys hour and q: {row!r}")
            continue
        hour, quantiles = row["hour"], row["q"]
        if not isinstance(hour, int) or hour not in model.FORECAST_HOURS or hour in seen:
            problems.append(f"bad or repeated hour {hour!r}")
        seen.add(hour)
        if not isinstance(quantiles, list) or len(quantiles) != 5:
            problems.append(f"hour {hour}: q must hold five numbers")
            continue
        previous = 0.0
        for level, value in zip(model.LEVELS, quantiles):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                problems.append(f"hour {hour} q{level}: not a finite number: {value!r}")
                continue
            if value <= 0 or value > model.MAX_QUANTILE:
                problems.append(f"hour {hour} q{level}: {value} outside (0, 1e9]")
            if value < previous:
                problems.append(f"hour {hour} q{level}: {value} below the previous quantile {previous}")
            previous = max(previous, value)
    if seen != set(model.FORECAST_HOURS):
        problems.append(f"hours present {sorted(seen)} are not 2..24")
    return problems


def median_summary(forecast: dict, hour1: float) -> str:
    medians = {row["hour"]: row["q"][2] / hour1 for row in forecast["hours"]}
    return "  ".join(f"h{hour}={medians[hour]:.2f}" for hour in SUMMARY_HOURS if hour in medians)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    failures = 0
    for case_path in sorted(input_dir.glob("*.in"), key=lambda path: path.name):
        out_path = output_dir / f"{case_path.stem}.out"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        hour1 = model.hour1_print(case)
        if not out_path.is_file():
            print(f"{case_path.stem}: MISSING {out_path}")
            failures += 1
            continue
        raw = out_path.read_bytes()
        problems = [] if len(raw) <= MAX_BYTES else [f"file is {len(raw)} bytes, over {MAX_BYTES}"]
        try:
            forecast = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            problems.append(f"not UTF-8 JSON: {error}")
            forecast = None
        if forecast is not None:
            problems.extend(validate(forecast))
        if problems:
            failures += 1
            print(f"{case_path.stem}: INVALID")
            for problem in problems:
                print(f"  - {problem}")
            continue
        name = case.get("game", {}).get("name", "?")
        print(f"{case_path.stem} ({name}, p={hour1:.0f}, {len(raw)} bytes): valid")
        print(f"  median / p: {median_summary(forecast, hour1)}")
    print(f"\n{failures} invalid" if failures else "\nall outputs valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
