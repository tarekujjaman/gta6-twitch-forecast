#!/usr/bin/env python3
"""Leave-one-out check of the model over a case's comparables.

Each qualifying launch is held out in turn: the model is fitted on the others,
the launch's own hour 1 is the print and its own 12 prior hours give its
feature, and the forecast is scored with the contest's pinball loss on the log
scale against the launch's real hours 2..24. The house baseline is scored the
same way for comparison.

Usage: python tools/loo.py [path/to/case.in]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import model  # noqa: E402

DEFAULT_CASE = REPO_ROOT.parent / "gta6-twitch-ccu-forecast-01-kit" / "input" / "p00.in"

HYPOTHETICAL_HOUR1 = 1_500_000.0
HYPOTHETICAL_FEATURE = 0.3
HYPOTHETICAL_HOURS = (2, 6, 9, 12, 16, 20, 24)


def pinball_loss(log_actual: float, quantiles: list[float]) -> float:
    """Mean over the five levels of 2 * max(tau * d, (tau - 1) * d), d = ln y - ln q."""
    total = 0.0
    for tau, quantile in zip(model.LEVELS, quantiles):
        d = log_actual - math.log(quantile)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(model.LEVELS)


def score_rows(launch: model.Launch, quantile_rows: list[list[float]]) -> float:
    """Mean pinball loss over hours 2..24 of the finalized quantiles against the launch."""
    log_hour1 = math.log(launch.hour1)
    losses = [
        pinball_loss(log_hour1 + log_ratio, model.finalize_quantiles(row))
        for log_ratio, row in zip(launch.log_ratios, quantile_rows)
    ]
    return sum(losses) / len(losses)


def held_out_forecast(held: model.Launch, others: list[model.Launch]) -> list[list[float]]:
    """The model's raw quantile rows for a held-out launch, trained on the others.

    The held-out launch's own hour 1 is the print, so the training weights
    favour launches of its size, exactly as they would for a real case.
    """
    if len(others) < model.MIN_TRAINING_LAUNCHES:
        return model.baseline_quantiles(held.hour1)
    fitted = model.fit_model(others, held.hour1)
    return model.predict(fitted, held.hour1, held.feature)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_hypothetical(launches: list[model.Launch]) -> None:
    """Median ratios and the heaviest training weights for a made-up case over all launches."""
    hour1, feature = HYPOTHETICAL_HOUR1, HYPOTHETICAL_FEATURE
    fitted = model.fit_model(launches, hour1)
    rows = model.predict(fitted, hour1, feature)
    medians = {hour: row[2] / hour1 for hour, row in zip(model.FORECAST_HOURS, rows)}
    print(f"\nhypothetical case: p = {hour1:,.0f}, x = {feature} over all {len(launches)} launches")
    print("median ratio  " + "  ".join(f"h{hour}={medians[hour]:.2f}" for hour in HYPOTHETICAL_HOURS))
    weights = model.training_weights(launches, hour1)
    total = sum(weights)
    print("largest normalised training weights:")
    for weight, launch in sorted(zip(weights, launches), key=lambda pair: -pair[0])[:5]:
        print(f"  {launch.name[:44]:<44} {weight / total:.3f}  (v1 = {launch.hour1:,.0f})")


def run(case_path: Path) -> None:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    launches = model.training_launches(case.get("comparables") or [])
    print(f"case {case.get('id')}: {len(case.get('comparables') or [])} comparables, "
          f"{len(launches)} qualifying launches\n")
    print(f"{'launch':<44} {'w':>3} {'x':>7} {'model':>8} {'base':>8}")
    results = []
    for index, held in enumerate(launches):
        others = launches[:index] + launches[index + 1 :]
        model_loss = score_rows(held, held_out_forecast(held, others))
        baseline_loss = score_rows(held, model.baseline_quantiles(held.hour1))
        results.append((held, model_loss, baseline_loss))
        print(f"{held.name[:44]:<44} {int(held.unlock_weight):>3} {held.feature:>7.3f} "
              f"{model_loss:>8.4f} {baseline_loss:>8.4f}")

    local = [row for row in results if row[0].midnight_local]
    print()
    print(f"{'group':<20} {'n':>3} {'model':>8} {'base':>8}")
    for label, rows in (("midnight-local", local), ("all launches", results)):
        print(f"{label:<20} {len(rows):>3} {mean([r[1] for r in rows]):>8.4f} "
              f"{mean([r[2] for r in rows]):>8.4f}")
    print_hypothetical(launches)


def main() -> int:
    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASE
    if not case_path.is_file():
        print(f"case file not found: {case_path}", file=sys.stderr)
        return 1
    run(case_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
