"""Launch-day Twitch viewer model shared by main.py and tools/loo.py.

The forecast for hours 2..24 of a release day is a ratio to the hour-1 print,
learned from past launches with one feature per launch: how far the peak of
the 12 hours before the day sat above hour 1, on the log scale. Each hour gets
its own weighted ridge regression of ln(v_h / v1) on that feature, and the
coefficients are smoothed over neighbouring hours before use.

A training launch's weight depends on the case being forecast: an unlock
weight (20 for midnight-local or rolling unlocks, else 1) times a size weight,
exp(-|ln(v1_c / p)| / 0.5) + 0.05, that favours launches whose hour 1 was
about as big as the case's. The regression is therefore fitted per case.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536)
FORECAST_HOURS = tuple(range(2, 25))

LOOKBACK = timedelta(hours=12)
RIDGE = 20.0
LOCAL_UNLOCK_WEIGHT = 20.0
LOCAL_UNLOCK_MARKERS = ("midnight local", "rolling")
NOT_A_LAUNCH_MARKER = "not a launch"
SIZE_SCALE = 0.5
SIZE_FLOOR = 0.05
MIN_TRAINING_LAUNCHES = 5
FEATURE_MARGIN = 0.5
SPREAD_FLOOR = 0.1
SPREAD_SCALE = 0.9
BASELINE_SPREAD = 0.15

MIN_QUANTILE = 0.01
MAX_QUANTILE = 1e9
MAX_EXPONENT = 100.0


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------


def parse_utc(text: str) -> datetime:
    """Parse an ISO 8601 timestamp into an aware UTC datetime."""
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def viewers_by_hour(rows: list[dict]) -> dict[datetime, float]:
    """Map each hourly row's parsed `from` time to its viewers."""
    table: dict[datetime, float] = {}
    for row in rows:
        viewers = row.get("viewers")
        if viewers is None:
            continue
        table[parse_utc(row["from"])] = float(viewers)
    return table


def hour1_print(case: dict) -> float:
    """The case's hour-1 print, at least 1."""
    known = case.get("known") or []
    try:
        viewers = float(known[0]["viewers"])
    except (IndexError, KeyError, TypeError, ValueError):
        return 1.0
    if not math.isfinite(viewers) or viewers < 1.0:
        return 1.0
    return viewers


# --------------------------------------------------------------------------
# Training launches
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """One past launch reduced to what the regression needs."""

    name: str
    unlock_weight: float
    hour1: float
    feature: float
    log_ratios: tuple[float, ...]  # ln(v_h / v1) for h = 2..24

    @property
    def midnight_local(self) -> bool:
        return self.unlock_weight == LOCAL_UNLOCK_WEIGHT


def is_launch(comparable: dict) -> bool:
    """False for the series flagged as not being a launch (GTA VI's own history)."""
    return NOT_A_LAUNCH_MARKER not in str(comparable.get("unlock", "")).lower()


def unlock_weight(unlock: str) -> float:
    """20 for launches that unlocked at midnight local time, else 1."""
    text = unlock.lower()
    if any(marker in text for marker in LOCAL_UNLOCK_MARKERS):
        return LOCAL_UNLOCK_WEIGHT
    return 1.0


def size_weight(launch_hour1: float, hour1: float) -> float:
    """exp(-|ln(v1_c / p)| / 0.5) + 0.05: about 1.05 for a launch the size of the case.

    The floor keeps every launch in play, so a case bigger than every comparable
    is not handed entirely to the largest one or two.
    """
    return math.exp(-abs(math.log(launch_hour1 / hour1)) / SIZE_SCALE) + SIZE_FLOOR


def training_weight(launch: Launch, hour1: float) -> float:
    """A launch's weight when forecasting a case whose hour-1 print is hour1."""
    return launch.unlock_weight * size_weight(launch.hour1, hour1)


def training_weights(launches: list[Launch], hour1: float) -> list[float]:
    return [training_weight(launch, hour1) for launch in launches]


def release_day_viewers(by_hour: dict[datetime, float], day_start: datetime) -> list[float] | None:
    """The 24 hourly viewers of the release day, or None if any is missing or not positive."""
    day = []
    for offset in range(24):
        viewers = by_hour.get(day_start + timedelta(hours=offset))
        if viewers is None or viewers <= 0:
            return None
        day.append(viewers)
    return day


def peak_before(by_hour: dict[datetime, float], day_start: datetime) -> float | None:
    """Max viewers of the points in [day_start - 12 h, day_start), or None without any."""
    window = [viewers for at, viewers in by_hour.items() if day_start - LOOKBACK <= at < day_start]
    if not window:
        return None
    return max(window)


def log_peak_ratio(peak: float, hour1: float) -> float:
    """The feature: ln(pre-day peak / hour 1), with the peak floored at 1 viewer."""
    return math.log(max(peak, 1.0) / hour1)


def extract_launch(comparable: dict) -> Launch | None:
    """Turn a comparable into a training launch, or None when it does not qualify."""
    if not is_launch(comparable):
        return None
    day_start = parse_utc(comparable["day_start"])
    by_hour = viewers_by_hour(comparable.get("hourly") or [])
    day = release_day_viewers(by_hour, day_start)
    peak = peak_before(by_hour, day_start)
    if day is None or peak is None:
        return None
    hour1 = day[0]
    return Launch(
        name=str(comparable.get("name", "")),
        unlock_weight=unlock_weight(str(comparable.get("unlock", ""))),
        hour1=hour1,
        feature=log_peak_ratio(peak, hour1),
        log_ratios=tuple(math.log(viewers / hour1) for viewers in day[1:]),
    )


def training_launches(comparables: list[dict]) -> list[Launch]:
    """Every comparable that qualifies as a training launch, in input order."""
    launches = []
    for comparable in comparables:
        launch = extract_launch(comparable)
        if launch is not None:
            launches.append(launch)
    return launches


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HourFit:
    """y = intercept + slope * x with a residual spread, for one hour."""

    intercept: float
    slope: float
    spread: float


@dataclass(frozen=True)
class Model:
    """Smoothed per-hour fits plus the training feature range."""

    fits: tuple[HourFit, ...]
    feature_mean: float
    feature_low: float
    feature_high: float


def fit_weighted_ridge(xs: list[float], ys: list[float], ws: list[float]) -> HourFit:
    """Weighted least squares of y on x with a ridge of RIDGE on the slope."""
    total = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs)) + RIDGE
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residual = sum(w * (y - intercept - slope * x) ** 2 for w, x, y in zip(ws, xs, ys))
    return HourFit(intercept=intercept, slope=slope, spread=math.sqrt(residual / total))


def smooth(values: list[float]) -> list[float]:
    """Centred +-1 moving average; the ends average the two values available."""
    count = len(values)
    smoothed = []
    for index in range(count):
        window = values[max(0, index - 1) : min(count, index + 2)]
        smoothed.append(sum(window) / len(window))
    return smoothed


def fit_model(launches: list[Launch], hour1: float) -> Model:
    """Fit one ridge regression per forecast hour, weighted for a case with print hour1, and smooth."""
    xs = [launch.feature for launch in launches]
    ws = training_weights(launches, hour1)
    raw_fits = [
        fit_weighted_ridge(xs, [launch.log_ratios[index] for launch in launches], ws)
        for index in range(len(FORECAST_HOURS))
    ]
    intercepts = smooth([fit.intercept for fit in raw_fits])
    slopes = smooth([fit.slope for fit in raw_fits])
    spreads = smooth([fit.spread for fit in raw_fits])
    fits = tuple(HourFit(a, b, s) for a, b, s in zip(intercepts, slopes, spreads))
    return Model(
        fits=fits,
        feature_mean=sum(w * x for w, x in zip(ws, xs)) / sum(ws),
        feature_low=min(xs) - FEATURE_MARGIN,
        feature_high=max(xs) + FEATURE_MARGIN,
    )


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def scaled_exp(base: float, exponent: float) -> float:
    """base * exp(exponent) without overflow."""
    return base * math.exp(min(exponent, MAX_EXPONENT))


def case_feature(history_hourly: list[dict], day_start: datetime, hour1: float) -> float | None:
    """The case's own feature from its history, or None when the 12 prior hours are empty."""
    peak = peak_before(viewers_by_hour(history_hourly), day_start)
    if peak is None:
        return None
    return log_peak_ratio(peak, hour1)


def predict(model: Model, hour1: float, feature: float) -> list[list[float]]:
    """Five raw quantiles for each forecast hour, the feature clamped to the training range."""
    feature = clamp(feature, model.feature_low, model.feature_high)
    rows = []
    for fit in model.fits:
        centre = fit.intercept + fit.slope * feature
        spread = max(SPREAD_FLOOR, fit.spread) * SPREAD_SCALE
        rows.append([scaled_exp(hour1, centre + z * spread) for z in Z_SCORES])
    return rows


def baseline_quantiles(hour1: float) -> list[list[float]]:
    """The house baseline: flat median, spread 0.15 * sqrt(h - 1)."""
    return [
        [scaled_exp(hour1, z * BASELINE_SPREAD * math.sqrt(hour - 1)) for z in Z_SCORES]
        for hour in FORECAST_HOURS
    ]


def finalize_quantiles(raw: list[float]) -> list[float]:
    """Round to 2 decimals, keep inside (0, 1e9] and make non-decreasing."""
    final = []
    floor = MIN_QUANTILE
    for value in raw:
        if not math.isfinite(value):
            value = MAX_QUANTILE
        value = clamp(round(value, 2), MIN_QUANTILE, MAX_QUANTILE)
        value = max(value, floor)
        floor = value
        final.append(value)
    return final


def hour_rows(quantile_rows: list[list[float]]) -> list[dict]:
    return [{"hour": hour, "q": finalize_quantiles(row)} for hour, row in zip(FORECAST_HOURS, quantile_rows)]


def forecast_case(case: dict) -> dict:
    """The full forecast object for one case."""
    hour1 = hour1_print(case)
    launches = training_launches(case.get("comparables") or [])
    if len(launches) < MIN_TRAINING_LAUNCHES:
        return {"hours": hour_rows(baseline_quantiles(hour1))}
    model = fit_model(launches, hour1)
    feature = case_feature(case.get("history_hourly") or [], parse_utc(case["day_start"]), hour1)
    if feature is None:
        feature = model.feature_mean
    return {"hours": hour_rows(predict(model, hour1, feature))}


def baseline_forecast(hour1: float) -> dict:
    return {"hours": hour_rows(baseline_quantiles(hour1))}


def encode_forecast(forecast: dict) -> bytes:
    """Compact, byte-stable JSON."""
    return json.dumps(forecast, separators=(",", ":"), allow_nan=False).encode("utf-8")
