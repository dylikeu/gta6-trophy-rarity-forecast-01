# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dylikeu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dylikeu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase at
2026-11-26T00:00:00Z. One program, `main.py` (Python 3.13, standard library
only, no network), reads each case in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The model

An ensemble of three median forecasts, combined on the logit scale:

1. **PRIOR — a cohort model.** Exophase's tracked population grows (adoption
   ~t^2.5 early, saturating later) while players progress through the game.
   Each trophy is classified from its description (story chapter, story
   mission, side strand, grind/collectible, money, activity, misc, 100%
   completion, platinum) with a characteristic play-time to earn it and an
   eventual ceiling; a player who joined at time *s* has earned it by time
   *t* with probability *c·(1 − exp(−(t−s)/τ))*. Earners are the old cohort
   (the last read's counts, progressing toward their ceiling) plus the new
   joiners. The forecast rate is earners / players-tracked, so dilution from
   the flood of new, mostly casual players is explicit: most rates fall even
   while absolute earner counts rise. The intro trophy ("Complete Gone
   Fishing") is earned by ~everyone within minutes, so it stays at ~100%.
   The launch-night refresh the kit's rules record (page frozen at 638
   players from 23:48 UTC 18 Nov, refreshed at 04:30 UTC 19 Nov to 3,868
   players, every rate but the 100% one lower) anchors the flat-history case.
2. **TREND — counts power law.** When the history actually moves, fit a power
   law in time-since-release to each trophy's earned count and to
   players-tracked separately (players-tracked with a saturating log-logistic
   fit, since adoption saturates and a straight log-log line overshoots),
   extrapolate both to the target, and divide.
3. **CURVE — the house curve baseline's shape.** Logit rate linear in
   ln(1 + hours since release), fit over the case's reads.

Two details matter. First, the launch-night page is a stale snapshot of a
reviewer-heavy population, not on the trend the later reads follow, so the
fitted members (TREND, CURVE) and the players-tracked fit all skip the first
distinct read. Second, with a flat history both fitted members just repeat
the last read, so PRIOR carries the forecast alone; with a rich history the
blend is roughly 1/3 PRIOR, 2/5 TREND, 1/3 CURVE.

Spreads are honest: a base term, a term growing with the square root of the
horizon, the disagreement between the ensemble members, and extra width when
the history is flat or the trophy is rare — scaled to the pinball-optimal
width found on the holdouts below. Medians obey the story chain
(chapter 1 ≥ 2 ≥ 3 ≥ 4 ≥ final mission), the platinum cannot exceed the final
mission or the 100% trophy, and the 100% trophy cannot exceed chapter 1.

## Calibration and validation

The prior's three numbers (adoption exponent 2.5, ~30 minutes of play a day
for the average tracked player, τ multiplier 4×) are the joint fit that best
matched the public Exophase trophy page as fetched live on 20 Nov 2026
(rates for all 55 trophies) while keeping every 04:30 rate below its
launch-night value, as the rules record, and putting the house baseline's
practice loss near its published ~1.51.

Validation (mean pinball loss on the logit scale, lower is better):

| Test | This entry | House baseline |
| --- | --- | --- |
| Practice p00, reconstructed 04:30 answer | 0.12 | 1.61 |
| Practice p01, reconstructed 05:00 answer | 0.13 | 1.82 |
| Live Exophase snapshot, 8 h horizon (real data) | 0.24 | 2.46 |
| Live Exophase snapshot, 27 h horizon (real data) | 0.47 | 2.25 |
| Synthetic final (365 reads, 4.5-day horizon) | 0.26 | 0.85 |

Two honest caveats. The practice answers are *reconstructions* from the
cohort model, not the house's true answers (those are unpublished), so the
practice rows measure the model against its own family — the independent
evidence is the live-snapshot holdout, where the entry beats the baseline
~5–10× on real Exophase rates. And the synthetic final is self-consistent
by construction (same model family generated it), so its row tests the
fitting machinery and the blend, not the calibration. Coverage on the
freshest real holdout is 0.91, i.e. the 90% bands are honest.

Deterministic (no randomness, no clock, no network): two runs on the same
input produce byte-identical outputs. Runtime is ~0.1 s per case (limit 8 s);
outputs are ~3.2 KB (limit 65,536 bytes). Edge cases (single read, null
rates, empty history) all produce valid output.

## License

MIT. See [LICENSE](LICENSE).
