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
  a launch).
- **Case-dependent weights.** A training launch `c` weighs
  `unlock_weight(c) * (exp(-|ln(v1_c / p)| / 0.5) + 0.05)`: the unlock weight
  is 20 when it unlocked at midnight local time (its unlock note says
  "midnight local" or "rolling"), else 1, and the size term favours launches
  whose hour 1 `v1_c` was about as big as the case's print `p`. The 0.05
  floor keeps every launch in play, so a case bigger than every comparable is
  not handed entirely to the largest one or two. Because the weights depend
  on `p`, the regression is fitted per case.
- **One feature.** `x = ln(peak of the 12 hours before the day / hour 1)`: how
  much of the pre-launch audience had already drained away by hour 1.
- **Per-hour weighted ridge regression** of `ln(v_h / v1)` on `x` for each
  hour 2..24, ridge 20 on the slope, with the weighted residual standard
  deviation as that hour's spread. Intercept, slope and spread are smoothed
  with a centred +-1 hour window.
- **The case's quantiles.** Its own feature from `history_hourly` (clamped to
  the training range +-0.5) gives the median, and the spread is
  `max(0.1, s_h) * 0.9` around it on the log scale at the normal quantiles of
  the five levels. With fewer than 5 training launches the house baseline is
  written instead.

### Leave-one-out on the practice comparables

`python tools/loo.py` holds out each qualifying launch of `input/p00.in` in
turn, forecasts it from its own hour 1 (which also sets the training weights)
and its own 12 prior hours, and scores it with the contest's pinball loss on
the log scale (lower is better):

| launches        | n  | this model | house baseline |
|-----------------|----|-----------:|---------------:|
| midnight-local  | 12 |     0.2497 |         0.3568 |
| all             | 37 |     0.7150 |         0.9296 |

The earlier version without the size weights (spread 0.75) scored 0.2898 and
0.7842 on the same study, and 0.218 on the practice leaderboard against the
baseline's 0.387. The tool also prints a hypothetical case (p = 1,500,000,
x = 0.3, all launches): its median ratios for hours 2, 6, 9, 12, 16, 20 and 24
are 0.88, 0.77, 0.70, 0.78, 1.28, 1.50 and 1.23, and ELDEN RING and Cyberpunk
2077 carry about half of the training weight.

Median curve on the kit's practice cases, as a ratio to the hour-1 print:

| case | game                  |  h2  |  h6  |  h12 |  h16 |  h20 |  h24 |
|------|-----------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| p00  | Marvel's Wolverine    | 0.65 | 1.06 | 1.55 | 2.62 | 3.58 | 3.15 |
| p01  | Red Dead Redemption 2 | 0.81 | 1.05 | 1.27 | 1.90 | 2.33 | 2.14 |
| p02  | God of War Ragnarök   | 0.77 | 0.84 | 0.95 | 1.55 | 1.87 | 1.69 |
| p03  | Marvel's Spider-Man 2 | 0.81 | 1.06 | 1.27 | 1.91 | 2.35 | 2.15 |

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
