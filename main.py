#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01.

Model (an ensemble of three median forecasts, combined on the logit scale):
  1. PRIOR -- a cohort model of how Exophase's tracked population grows and how
     players progress.  Players join over time (adoption roughly ~ t**2.5 in
     the first days, saturating later, anchored on the launch-night refresh
     the kit's rules record:
     the page sat unchanged at 638 players tracked from 23:48 UTC 18 Nov and
     refreshed at 04:30 UTC 19 Nov to 3,868 players, every rate but the 100%
     one lower).  Each trophy has a kind (story chapter, story mission, side
     strand, grind/collectible, money, activity, misc, completion, platinum)
     read from its description, with a characteristic play-time to earn it and
     an eventual ceiling; a player who joined at time s has earned it by time
     t with probability c * (1 - exp(-(t - s) / tau)).  Earners are the old
     cohort (the last read's counts, progressing toward their ceiling) plus
     the new joiners.  The forecast rate is earners / players-tracked, so
     dilution from the flood of new, mostly casual players is explicit: most
     rates fall even while absolute earner counts rise, and only the earliest
     story trophies can rise.  The intro trophy ("Complete Gone Fishing") is
     earned by ~everyone within minutes, so it stays at ~100%.
  2. TREND -- when the history actually moves (the final case: reads every
     10 minutes over ~2.5 days), fit a power law in time-since-release to each
     trophy's earned COUNT and to players-tracked separately, extrapolate
     both to the target, and divide.  Counts only grow, so this respects
     monotonicity and captures dilution directly.
  3. CURVE -- the house curve baseline's shape (logit rate linear in
     ln(1 + hours since release), fit over the case's reads), as a third
     opinion.  With a flat history both TREND and CURVE collapse to the last
     read, and PRIOR carries the forecast alone.

Spreads are honest: a base term, a term growing with the square root of the
horizon, the disagreement between the ensemble members, and extra width when
the history is flat (the forecast is prior-driven) or the trophy is rare.
Quantiles are the median's logit plus z_tau * sigma, mapped back to percent.

Deterministic (no randomness, no clock, no network), standard library only.
Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input), writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- calibrated constants (from the kit's rules + the public Exophase page) ---
# Adoption after launch is roughly ~t**2.5 in time since release; the first
# refresh of the launch-night page multiplied players-tracked by ~6.06
# (638 -> 3,868 at 04:30 UTC 19 Nov).  The average tracked player plays of
# the order of half an hour a day in week one, so wall-clock times to earn a
# trophy are play-hours / DUTY; TAU_MULT is the calibration multiplier.
# These three numbers are the joint fit that best matched the public
# Exophase snapshot of ~20 Nov (rates for all 55 trophies, fetched live)
# while keeping every 04:30 rate below its launch-night value, as the kit's
# rules record, and putting the house baseline's practice loss near its
# published ~1.51.
ADOPT_GAMMA = 2.5
FIRST_REFRESH_JUMP = 6.06
FIRST_REFRESH_HOURS = 4.5
DUTY = 1.0 / 48.0
TAU_MULT = 4.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


# ---------------------------------------------------------------- kinds ----
def trophy_kind(t):
    """Classify a trophy from its description (and id for the special ones)."""
    tid = t["id"]
    if tid == 1:
        return "platinum"
    if tid == 2:
        return "intro"
    if tid == 14:
        return "completion"
    d = (t.get("description") or "").lower()
    if "complete chapter" in d:
        return "chapter"
    if d.startswith("complete the final mission."):
        return "final"
    if d.startswith("complete the final mission for"):
        return "side"
    if d.startswith("complete ") and any(
        k in d
        for k in (
            "breakfast run",
            "cartel heist",
            "child's play",
            "headliners",
            "vcia takedown",
            "operation savage glory",
        )
    ):
        return "mission"
    if any(
        k in d
        for k in (
            "250 stunts",
            "collectible card",
            "100 locations",
            "50 challenges",
            "100 vehicles",
            "20 different races",
            "drone race",
            "gold medal",
            "8 mission strands",
            "explore all",
            "5 bank vaults",
            "illegal operation",
        )
    ):
        return "grind"
    if any(
        k in d
        for k in ("$250,000", "$100,000", "garage propert", "register 10 vehicles")
    ):
        return "money"
    if any(
        k in d
        for k in (
            "each type of activity",
            "arcade game",
            "fishing challenge",
            "meditate",
            "unique animal",
        )
    ):
        return "activity"
    return "misc"


# Play-hours to earn (median-ish) and eventual ceiling share, per kind.
# Chapter/mission/activity times are per trophy where the description pins
# them down; the rest share their kind's values.  Campaign ~50h, platinum
# ~200h (public reporting); the wall-clock time is play-hours / DUTY, times
# TAU_MULT, per the calibration in the module docstring.
KIND_PARAMS = {
    "intro": (0.03, 0.999),  # wall-clock hours: earned within minutes
    "activity": (3.0, 0.55),
    "mission": (12.0, 0.70),
    "side": (20.0, 0.50),
    "final": (50.0, 0.60),
    "misc": (12.0, 0.45),
    "money": (30.0, 0.30),
    "grind": (60.0, 0.20),
    "completion": (150.0, 0.12),
    "platinum": (200.0, 0.06),
}
CHAPTER_TAU = {3: 10.0, 4: 18.0, 5: 28.0, 6: 38.0}
MISSION_TAU = {8: 6.0, 9: 15.0, 10: 20.0, 11: 25.0, 12: 30.0, 13: 40.0}
ACTIVITY_TAU = {16: 25.0, 20: 2.0, 21: 3.0, 28: 30.0, 29: 25.0}


def kind_params(t):
    kind = trophy_kind(t)
    if t["id"] == 35:
        # "Discover 100 vehicles" is passive (driving around discovers
        # them), far quicker than the other grinds.
        return 15.0, 0.30, False
    if kind == "intro":
        return KIND_PARAMS["intro"][0], KIND_PARAMS["intro"][1], True
    if kind == "chapter":
        return CHAPTER_TAU.get(t["id"], 18.0), 0.85, False
    if kind == "mission":
        return MISSION_TAU.get(t["id"], 12.0), 0.70, False
    if kind == "activity":
        return ACTIVITY_TAU.get(t["id"], 3.0), 0.55, False
    tau_play, ceiling = KIND_PARAMS[kind]
    return tau_play, ceiling, False


def tau_wall(t):
    """Wall-clock hours for a typical tracked player to earn this trophy."""
    tau, _ceiling, is_wall = kind_params(t)
    if is_wall:
        return tau
    return tau * TAU_MULT / DUTY


# ------------------------------------------------------------- utilities ---
def linfit(xs, ys):
    """Least-squares y = a + b*x; returns (a, b) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def fit_loglogistic(ts, ns):
    """Fit N(t) = N_inf * t**g / (t**g + th**g) (saturating adoption) by grid
    search over (g, th), solving log N_inf by least squares in log space.
    Returns the fitted N(t) function, or None.  ts: hours since release (>0)."""
    pts = [(t, n) for t, n in zip(ts, ns) if t > 0 and n and n > 0]
    if len(pts) < 3:
        return None
    best = None
    tmax = max(t for t, _n in pts)
    ths = [tmax * f for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)]
    for g in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        for th in ths:
            fs = []
            ok = True
            for t, _n in pts:
                f = t ** g / (t ** g + th ** g)
                if f <= 0:
                    ok = False
                    break
                fs.append(f)
            if not ok:
                continue
            lni = sum(math.log(n) - math.log(f) for (_t, n), f in zip(pts, fs)) / len(pts)
            sse = sum((math.log(n) - (lni + math.log(f))) ** 2
                      for (_t, n), f in zip(pts, fs))
            if best is None or sse < best[0]:
                best = (sse, g, th, math.exp(lni))
    if best is None:
        return None
    _sse, g, th, n_inf = best

    def n_of(t):
        if t <= 0:
            return pts[0][1]
        return n_inf * t ** g / (t ** g + th ** g)

    return n_of


def fit_reads(distinct):
    """Reads usable for fitting trends: everything after the first distinct
    page.  The first read is the launch-night page -- a stale snapshot of a
    reviewer-heavy population (638 players, every rate frozen) -- and it is
    not on the trend the later reads follow, so fitting through it bends
    every slope.  With a flat history this is empty."""
    if len(distinct) < 2:
        return []
    return distinct[1:]


def distinct_reads(history, n_trophies):
    """Collapse consecutive identical reads (Exophase only refreshes every
    few hours) into the distinct pages, keeping the first-seen time of each.
    Also returns the last read's per-trophy rate, filling nulls from
    earlier reads."""
    reads = []
    last_rates = [None] * n_trophies
    for h in history:
        rates = list(h.get("rates") or [])
        if len(rates) < n_trophies:
            rates = rates + [None] * (n_trophies - len(rates))
        for i, r in enumerate(rates):
            if r is not None:
                last_rates[i] = r
        reads.append((h, rates))
    # distinct in (players, rates) terms
    distinct = []
    prev_key = None
    for h, rates in reads:
        key = (h.get("players_tracked"), tuple(rates))
        if key != prev_key:
            distinct.append((h, rates))
            prev_key = key
    return distinct, last_rates


# ------------------------------------------------------------ member 1 -----
def prior_forecast(case, trophies, distinct, last_rates, t_rel, t_target_h,
                   n_last):
    """Cohort model: old cohort progresses toward its ceiling, new joiners
    arrive on the adoption curve and earn with F(age) = c*(1-exp(-age/tau)).
    Returns (N_target, [median rate per trophy])."""
    n_tr = len(trophies)
    # --- players-tracked at the target ---
    fr = fit_reads(distinct)
    n_of = None
    if len(fr) >= 3:
        ts = [max(hours_between(t_rel, parse_time(d[0]["at"])), 0.0) for d in fr]
        ns_ = [d[0].get("players_tracked") or 0 for d in fr]
        n_of = fit_loglogistic(ts, ns_)
    if n_of is not None:
        n_target = n_of(t_target_h)
        n_target = min(max(n_target, n_last), n_last * 200.0)
    elif len(distinct) >= 2 and len({d[0].get("players_tracked") for d in distinct}) >= 2:
        xs = [max(hours_between(t_rel, parse_time(d[0]["at"])), 0.0) + 1.0
              for d in distinct]
        ys = [math.log(max(d[0].get("players_tracked") or 1, 1)) for d in distinct]
        fit = linfit([math.log(x) for x in xs], ys)
        if fit is not None:
            a, b = fit
            b = min(max(b, 0.0), 4.0)
            n_target = math.exp(a + b * math.log(t_target_h + 1.0))
        else:
            n_target = float(n_last)
        n_target = min(max(n_target, n_last), n_last * 200.0)
    else:
        # Flat page: it is stale and the next refresh jumps it.  On launch
        # night the first refresh multiplied it by ~6.06; afterwards the
        # adoption curve (~t**2) keeps it climbing.
        jump = FIRST_REFRESH_JUMP if n_last <= 1000 else 2.0
        if t_target_h <= FIRST_REFRESH_HOURS:
            n_target = n_last * jump
        else:
            n_target = n_last * jump * (t_target_h / FIRST_REFRESH_HOURS) ** ADOPT_GAMMA
        n_target = max(n_target, n_last)
    # --- join cohorts between the last read and the target ---
    t_last_h = hours_between(t_rel, parse_time(distinct[-1][0]["at"]))
    span = max(t_target_h - t_last_h, 0.0)
    steps = 24
    joins = []  # (age at target, count)
    if span > 0 and n_target > n_last:
        prev = 0.0
        # Cumulative joins follow the fitted adoption curve when there is
        # one (it saturates), else the ~t**gamma launch-night shape.
        if n_of is not None:
            base = n_of(t_last_h)
            denom = max(n_of(t_target_h) - base, 1e-9)
            for i in range(1, steps + 1):
                frac = i / steps
                t_at = t_last_h + frac * span
                cum = (n_of(t_at) - base) / denom
                joins.append((t_target_h - (t_at - span / steps / 2.0),
                              (n_target - n_last) * max(cum - prev, 0.0)))
                prev = cum
        else:
            denom = (max(t_target_h, 0.0) ** ADOPT_GAMMA
                     - max(t_last_h, 0.0) ** ADOPT_GAMMA)
            for i in range(1, steps + 1):
                frac = i / steps
                t_at = t_last_h + frac * span
                if denom > 0:
                    cum = ((max(t_at, 0.0) ** ADOPT_GAMMA
                            - max(t_last_h, 0.0) ** ADOPT_GAMMA) / denom)
                else:
                    cum = frac
                joins.append((t_target_h - (t_at - span / steps / 2.0),
                              (n_target - n_last) * max(cum - prev, 0.0)))
                prev = cum
        # normalise (rounding): scale so the joins sum to the new players
        tot = sum(j for _a, j in joins)
        if tot > 0:
            joins = [(a, j * (n_target - n_last) / tot) for a, j in joins]
    medians = []
    dt_old = max(t_target_h - t_last_h, 0.0)
    for i, t in enumerate(trophies):
        _tau_play, ceiling, _is_wall = kind_params(t)
        tau = tau_wall(t)
        r_last = last_rates[i] if last_rates[i] is not None else 0.0
        e0 = r_last * n_last / 100.0
        # old cohort: progresses from its current count toward its ceiling
        headroom = n_last * ceiling - e0
        e_old = e0 + headroom * (1.0 - math.exp(-dt_old / tau)) if headroom > 0 else e0
        # new cohort: each joiner earns with F(age)
        e_new = 0.0
        for age, j in joins:
            if age <= 0:
                continue
            e_new += j * ceiling * (1.0 - math.exp(-age / tau))
        medians.append(100.0 * (e_old + e_new) / n_target)
    return n_target, medians


# ------------------------------------------------------------ member 2 -----
def trend_forecast(case, trophies, distinct, last_rates, t_rel, t_target_h):
    """Power law in time-since-release, fit separately to each trophy's
    earned count and to players-tracked; rate = earners / players.
    Falls back to the last read when the history does not move."""
    n_tr = len(trophies)
    fr = fit_reads(distinct)
    if len(fr) < 3:
        return list(last_rates)
    xs = []
    for d in fr:
        xs.append(max(hours_between(t_rel, parse_time(d[0]["at"])), 0.0) + 1.0)
    lx = [math.log(x) for x in xs]
    # players-tracked at the target: saturating fit when it works, else the
    # power law (adoption saturates; a straight log-log line overshoots).
    n_last = distinct[-1][0].get("players_tracked") or 1
    n_of = fit_loglogistic(
        [x - 1.0 for x in xs],
        [d[0].get("players_tracked") or 0 for d in fr])
    if n_of is not None:
        n_target = n_of(t_target_h)
    else:
        ys_n = [math.log(max(d[0].get("players_tracked") or 1, 1)) for d in fr]
        fit_n = linfit(lx, ys_n)
        if fit_n is None:
            return list(last_rates)
        a_n, b_n = fit_n
        b_n = min(max(b_n, 0.0), 4.0)
        n_target = math.exp(a_n + b_n * math.log(t_target_h + 1.0))
    n_target = min(max(n_target, n_last), n_last * 200.0)
    out = []
    for i in range(n_tr):
        ys = []
        ok = True
        for d in fr:
            r = d[1][i]
            if r is None:
                ok = False
                break
            n = d[0].get("players_tracked") or 0
            e = r * n / 100.0
            ys.append(math.log(max(e, 0.05)))
        if not ok:
            out.append(last_rates[i])
            continue
        fit = linfit(lx, ys)
        if fit is None:
            out.append(last_rates[i])
            continue
        a, b = fit
        b = min(max(b, 0.0), 6.0)
        e_target = math.exp(a + b * math.log(t_target_h + 1.0))
        # a rate cannot pass its kind's eventual ceiling
        _tau, ceiling, _w = kind_params(trophies[i])
        out.append(min(100.0 * e_target / n_target, ceiling * 100.0))
    return out


# ------------------------------------------------------------ member 3 -----
def curve_forecast(case, trophies, distinct, last_rates, t_rel, t_target_h):
    """House curve baseline's shape: logit(rate) linear in ln(1 + hours
    since release), fit over the case's reads; read at the target."""
    n_tr = len(trophies)
    fr = fit_reads(distinct)
    if len(fr) < 3:
        return list(last_rates)
    xs = [math.log(1.0 + max(hours_between(t_rel, parse_time(d[0]["at"])), 0.0))
          for d in fr]
    x_target = math.log(1.0 + t_target_h)
    out = []
    for i in range(n_tr):
        ys = []
        ok = True
        for d in fr:
            r = d[1][i]
            if r is None:
                ok = False
                break
            ys.append(logit(r))
        if not ok:
            out.append(last_rates[i])
            continue
        fit = linfit(xs, ys)
        if fit is None:
            out.append(last_rates[i])
            continue
        a, b = fit
        out.append(sigmoid(a + b * x_target))
    return out


# ------------------------------------------------------------- forecast ----
def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    history = case.get("history") or []
    if not history:
        # No reads at all: say every rate is 1% with a very wide spread.
        return [{"id": t["id"], "q": [0.01, 0.1, 1.0, 10.0, 50.0]}
                for t in trophies]
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if "T" not in str(case["game"].get("release_date", "")) \
        else parse_time(case["game"]["release_date"])
    t_rel = release
    target = parse_time(case["target"]) if case.get("target") else \
        parse_time(case["at"])
    horizon_min = case.get("horizon_minutes")
    if horizon_min is None:
        horizon_min = (target - parse_time(case["at"])).total_seconds() / 60.0
    t_target_h = hours_between(t_rel, target)
    horizon_days = horizon_min / 1440.0

    distinct, last_rates = distinct_reads(history, n_tr)
    last_rates = [0.0 if r is None else r for r in last_rates]
    n_last = distinct[-1][0].get("players_tracked") or 1
    n_distinct = len(distinct)
    flat = n_distinct < 2 or (
        len({d[0].get("players_tracked") for d in distinct}) < 2
        and all(d[1] == distinct[0][1] for d in distinct)
    )

    _n_target, m_prior = prior_forecast(case, trophies, distinct, last_rates,
                                        t_rel, t_target_h, n_last)
    m_trend = trend_forecast(case, trophies, distinct, last_rates, t_rel,
                             t_target_h)
    m_curve = curve_forecast(case, trophies, distinct, last_rates, t_rel,
                             t_target_h)

    # Blend weights: with a flat history the fitted members just repeat the
    # last read, so the prior carries everything; with a moving history the
    # fitted members earn most of the weight and the prior stays as an anchor.
    if flat:
        w_prior, w_trend, w_curve = 1.0, 0.0, 0.0
    else:
        # The prior is the only member with the physics (saturating
        # adoption, cohort progression, dilution), so it keeps a real
        # share even with a rich history; the fitted members earn the rest.
        richness = min(1.0, (len(fit_reads(distinct)) - 1) / 24.0)
        w_prior = 0.30 + 0.05 * (1.0 - richness)
        w_trend = 0.40
        w_curve = 1.0 - w_prior - w_trend

    medians = []
    spreads = []
    for i in range(n_tr):
        members = []
        if w_prior > 0:
            members.append((w_prior, m_prior[i]))
        if w_trend > 0:
            members.append((w_trend, m_trend[i]))
        if w_curve > 0:
            members.append((w_curve, m_curve[i]))
        # logit-space weighted mean of the member medians
        tot_w = sum(w for w, _m in members)
        lm = sum(w * logit(m) for w, m in members) / tot_w
        med = sigmoid(lm)
        # disagreement between members, in logit units
        if len(members) > 1:
            mean_l = lm
            var = sum(w * (logit(m) - mean_l) ** 2 for w, m in members) / tot_w
            disagree = math.sqrt(var)
        else:
            disagree = 0.0
        sigma = math.sqrt(
            0.40 ** 2
            + (0.35 * math.sqrt(max(horizon_days, 0.0))) ** 2
            + min(disagree, 1.5) ** 2
            + (0.35 ** 2 if flat else 0.0)
            + (0.25 ** 2 if med < 0.2 else 0.0)
        )
        # The raw spread above is wider than the pinball-optimal width on
        # every holdout we can build (practice reconstructions, the live
        # Exophase snapshot, a synthetic final): scaling it by 0.75 keeps
        # 90% coverage honest (~0.9) while cutting the loss.
        sigma *= 0.75
        medians.append(med)
        spreads.append(sigma)

    # Cross-trophy constraints, on the medians.
    # Story chain must not rise: chapter 1 >= 2 >= 3 >= 4 >= final mission.
    chain = [t["id"] for t in trophies
             if trophy_kind(t) in ("chapter", "final")]
    chain.sort(key=lambda tid: {3: 0, 4: 1, 5: 2, 6: 3, 7: 4}.get(tid, 99))
    if len(chain) > 1:
        idx = {t["id"]: k for k, t in enumerate(trophies)}
        vals = [medians[idx[tid]] for tid in chain]
        # isotonic (non-increasing) by pooling adjacent violators
        pooled = [[v] for v in vals]
        j = 0
        while j < len(pooled) - 1:
            a = sum(pooled[j]) / len(pooled[j])
            b = sum(pooled[j + 1]) / len(pooled[j + 1])
            if a < b:
                pooled[j] = pooled[j] + pooled[j + 1]
                del pooled[j + 1]
                if j > 0:
                    j -= 1
            else:
                j += 1
        fixed = []
        for group in pooled:
            fixed.extend([sum(group) / len(group)] * len(group))
        for tid, v in zip(chain, fixed):
            medians[idx[tid]] = v
    # Platinum cannot exceed the final mission or the 100% trophy; 100%
    # cannot exceed chapter 1 (it needs the story done).
    idx = {t["id"]: k for k, t in enumerate(trophies)}
    if 1 in idx and 7 in idx:
        medians[idx[1]] = min(medians[idx[1]], medians[idx[7]])
    if 1 in idx and 14 in idx:
        medians[idx[1]] = min(medians[idx[1]], medians[idx[14]])
    if 14 in idx and 3 in idx:
        medians[idx[14]] = min(medians[idx[14]], medians[idx[3]])

    rows = []
    for i, t in enumerate(trophies):
        kind = trophy_kind(t)
        med = medians[i]
        sigma = spreads[i]
        if kind == "intro":
            # Earned by ~everyone within minutes of starting: stays ~100%.
            # The live page prints 99.99%, so centre high with a band that
            # reaches it.
            med = 99.97
            sigma = 0.5
        q = []
        for z in Z:
            v = sigmoid(logit(med) + z * sigma)
            q.append(round(clamp_pct(v), 4))
        # enforce non-decreasing (rounding can only break ties by epsilon)
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": t["id"], "q": q})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for fname in files:
        cid = fname[: -len(".in")]
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = {"trophies": rows}
        text = json.dumps(payload) + "\n"
        with open(os.path.join(out_dir, cid + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
