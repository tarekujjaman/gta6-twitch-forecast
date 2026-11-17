# gta6-twitch-forecast
[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tarekujjaman/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tarekujjaman/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts the Twitch concurrent viewers of Grand Theft Auto VI's category for
hours 2 to 24 of its release day, given the day's first hour and the hourly
history of past big launches. Python 3.13, standard library only, deterministic.

## The model

Every forecast is a ratio to the hour-1 print `p`, learned from the case's
comparable launches:

- **Training launches.** Each comparable whose release day has all 24 hourly
  points with positive viewers and at least one point in the 12 hours before
  the day. The GTA VI category's own pre-release history is skipped (it is not
  a launch). A launch that unlocked at midnight local time (its unlock note
  says "midnight local" or "rolling") weighs 20, any other 1.
- **One feature.** `x = ln(peak of the 12 hours before the day / hour 1)`: how
  much of the pre-launch audience had already drained away by hour 1.
- **Per-hour weighted ridge regression** of `ln(v_h / v1)` on `x` for each
  hour 2..24, ridge 20 on the slope, with the weighted residual standard
  deviation as that hour's spread. Intercept, slope and spread are smoothed
  with a centred +-1 hour window.
- **The case's quantiles.** Its own feature from `history_hourly` (clamped to
  the training range +-0.5) gives the median, and the spread is
  `max(0.1, s_h) * 0.75` around it on the log scale at the normal quantiles of
  the five levels. With fewer than 5 training launches the house baseline is
  written instead.

### Leave-one-out on the practice comparables

`python tools/loo.py` holds out each qualifying launch of `input/p00.in` in
turn, forecasts it from its own hour 1 and 12 prior hours, and scores it with
the contest's pinball loss on the log scale (lower is better):

| launches        | n  | this model | house baseline |
|-----------------|----|-----------:|---------------:|
| midnight-local  | 12 |     0.2898 |         0.3568 |
| all             | 37 |     0.7842 |         0.9296 |

Median curve on the kit's practice cases, as a ratio to the hour-1 print:

| case | game                  |  h2  |  h6  |  h12 |  h16 |  h20 |  h24 |
|------|-----------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| p00  | Marvel's Wolverine    | 0.72 | 1.03 | 1.55 | 2.90 | 3.95 | 3.41 |
| p01  | Red Dead Redemption 2 | 0.79 | 0.95 | 1.21 | 2.03 | 2.57 | 2.24 |
| p02  | God of War Ragnarök   | 0.79 | 0.96 | 1.24 | 2.09 | 2.67 | 2.33 |
| p03  | Marvel's Spider-Man 2 | 0.79 | 0.95 | 1.22 | 2.06 | 2.62 | 2.28 |

## Run it

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
```

`main.py` reads every `*.in` in the input directory (default `/input/`), in
name order, and writes `<id>.out` to the output directory (default
`/output/`, created if missing). `python tools/check.py input out` validates
the outputs against the contest's shape rules and prints each median curve.

## Layout

- `main.py`: the entrypoint, reads the cases and writes the forecasts.
- `model.py`: the model, shared by the entrypoint and the tools.
- `tools/loo.py`: leave-one-out scoring over a case's comparables.
- `tools/check.py`: output validator.

---

<sub>Entered on [make.fun](https://make.fun/?ref=y2gg1pq1&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
