#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lambda_blindness_check.py
=========================

Version:
    0.4.0 (2026-10-16)

Supplementary research code for:

    Peter Kahl, 'What a Quantum Milestone Warrants: Error Correction, Evidence
    and Scaling Decisions' (2026), §6.3 and §6.8.

Author:
    Peter Kahl
    Independent researcher, Lex et Ratio
    https://www.lexetratio.com
    ORCID: 0009-0003-1616-4843

Repository:
    https://github.com/Peter-Kahl/quantum-milestone-warrants

Copyright (c) 2026 Peter Kahl. MIT License (see LICENSE).
SPDX-License-Identifier: MIT

Purpose
-------
Is the headline metric of a below-threshold memory experiment, the suppression
factor Lambda fitted at code distances 3, 5 and 7, sensitive to rare
correlated events that would change a decision to scale? And how long would a
direct demonstration of the target take?

This is NOT a model of Google's Willow processor or of any real device.
Willow-like magnitudes (Google Quantum AI and Collaborators, Nature 638 (2025)
920-926, doi:10.1038/s41586-024-08449-y) are used only so that orders of
magnitude are realistic. Every number printed is a synthetic consequence of
the assumptions below.

Two noise hypotheses
--------------------
    M0: p0(d) = P7 * LAM**((7 - d)/2)          pure suppression
    M1: p1(d) = p0(d) + B7 * (d/7)**kappa      plus a rare-event floor

    kappa = 0   constant floor
    kappa > 0   larger patches intercept more rare events
    kappa < 0   larger codes absorb part of each event

Two experiments, compared at equal machine time
-----------------------------------------------
    O2  extend the headline measurement: logical-failure counting at d = 3, 5
        and 7, machine time split equally. Evaluated in its BEST case (M0's
        amplitude treated as known), so the result is an upper bound for O2.
        Each cycle is an independent Bernoulli trial (a simplification; real
        memory experiments count failures per multi-cycle shot).
    O3  count detectable rare events in a long run: Poisson with rate
        eta*r + r_bg under M1 and r_bg under M0, where B7 = r * mu and mu, the
        probability that one event causes a distance-7 logical failure, is an
        ASSUMPTION swept in Table 3.

Measure of discrimination: expected log-likelihood ratio (Kullback-Leibler
divergence) when M1 is true. 'Decisive' means an expected log-likelihood ratio
of ln 20 (about 20:1 odds), reported as machine-hours needed.

Tables
------
    1  Floor sweep: Lambda shift and time to decisive evidence     (§6.3)
    2  Shape of the floor (kappa)                                  (§6.3)
    3  Sensitivity of O3 to mu
    4  Does the floor change the scaling decision?                 (§6.3)
    5  Window of floors invisible to Lambda but decision-changing  (§6.3)
    6  Zero-failure demonstration: run length and pass probability (§6.8)

Usage
-----
    python3 lambda_blindness_check.py | tee output/lambda_blindness_check_output.txt

Writes output/lambda_blindness_check_results.json (or the path given as the
first argument). Run time a few seconds.
"""

import json
import math
import os
import sys

import numpy as np

__version__ = "0.4.0"

# ============================================================================
# Fixed parameters (Willow-like magnitudes; see module docstring)
# ============================================================================
P7 = 1.43e-3          # logical error per cycle at d = 7 (Willow-like magnitude)
LAM = 2.14            # suppression factor per +2 in distance (Willow-like)
LAM_SE = 0.02         # reported uncertainty on Lambda (Willow-like)
T_CYCLE = 1.1e-6      # seconds per QEC cycle (Willow-like)
CYCLES_PER_HOUR = 3600.0 / T_CYCLE
DISTANCES = np.array([3, 5, 7])
DECISIVE_LLR = math.log(20.0)

# O3 defaults
ETA = 1.0             # detection efficiency for rare events
R_BG = 1.0 / (100 * CYCLES_PER_HOUR)   # false-event background: 1 per 100 h
MU_DEFAULT = 0.3      # mapping parameter (swept below)

# Decision
TARGETS = [1e-9, 1e-12]   # logical-memory error per cycle that d* must meet


# ============================================================================
# Model and discrimination measures
# ============================================================================

def p0(d):
    """Logical error per cycle at distance d under pure suppression (M0)."""
    return P7 * LAM ** ((7.0 - d) / 2.0)


def floor(d, b7, kappa):
    """Rare-event floor at distance d (M1 minus M0)."""
    return b7 * (d / 7.0) ** kappa


def p1(d, b7, kappa):
    """Logical error per cycle at distance d under M1."""
    return p0(d) + floor(d, b7, kappa)


def fit_lambda(ps):
    """Least-squares fit of log p = a - ((d+1)/2) log LAM over d = 3, 5, 7."""
    x = (DISTANCES + 1) / 2.0
    slope, _ = np.polyfit(x, np.log(ps), 1)
    return math.exp(-slope)


def kl_bernoulli(a, b):
    """KL(Bern(a) || Bern(b)).

    For tiny relative differences the direct formula suffers catastrophic
    cancellation in double precision, so the second-order expansion
    (a - b)**2 / (2 b (1 - b)) is used when |a - b| / b < 1e-4.
    """
    if abs(a - b) / b < 1e-4:
        return (a - b) ** 2 / (2.0 * b * (1.0 - b))
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def kl_poisson(l1, l0):
    """KL(Poisson(l1) || Poisson(l0)) per unit exposure."""
    return l1 * math.log(l1 / l0) - l1 + l0


def o2_hours(b7, kappa):
    """Machine-hours for O2 to reach decisive evidence for M1 over M0."""
    per_cycle = sum(kl_bernoulli(p1(d, b7, kappa), p0(d)) for d in DISTANCES) / len(DISTANCES)
    if per_cycle <= 0:
        return math.inf
    return DECISIVE_LLR / (per_cycle * CYCLES_PER_HOUR)


def o3_hours(b7, mu=MU_DEFAULT, eta=ETA, r_bg=R_BG):
    """Machine-hours for O3 to reach decisive evidence for M1 over M0."""
    r = b7 / mu
    per_cycle = kl_poisson(eta * r + r_bg, r_bg)
    return DECISIVE_LLR / (per_cycle * CYCLES_PER_HOUR)


def d_star(target):
    """Smallest odd distance at which M0 meets the target."""
    d = 7
    while p0(d) > target:
        d += 2
    return d


def fmt_hours(h):
    """Human-readable duration."""
    if h == math.inf:
        return "inf"
    if h < 1:
        return f"{h*60:.1f} min"
    if h < 24 * 365:
        return f"{h:,.1f} h"
    return f"{h/(24*365):,.1f} yr"


# ============================================================================
# Main
# ============================================================================

def main():
    out = {"version": __version__, "parameters": {
        "P7": P7, "LAM": LAM, "LAM_SE": LAM_SE, "T_CYCLE_s": T_CYCLE,
        "ETA": ETA, "R_BG_per_cycle": R_BG, "MU_DEFAULT": MU_DEFAULT,
        "DECISIVE_LLR": DECISIVE_LLR, "TARGETS": TARGETS}}

    print(f"lambda_blindness_check.py v{__version__}  (What a Quantum Milestone Warrants, §6.3, §6.8)")
    print("Toy model. NOT a model of Willow or of any real device. Every number below")
    print("is a synthetic consequence of the stated assumptions.")
    print()

    # ------------------------------------------------ Table 1: floor sweep
    print("TABLE 1  Floor sweep (kappa = 0, mu = 0.3)")
    print(f"{'B7 floor':>10} {'Lambda fit':>11} {'shift/SE':>9} {'O2 time':>14} {'O3 time':>12} {'ratio O2/O3':>12}")
    t1 = []
    for b7 in [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4]:
        lam_fit = fit_lambda([p1(d, b7, 0) for d in DISTANCES])
        shift = (LAM - lam_fit) / LAM_SE
        h2, h3 = o2_hours(b7, 0), o3_hours(b7)
        t1.append({"B7": b7, "lambda_fit": lam_fit, "shift_in_SE": shift,
                   "O2_hours": h2, "O3_hours": h3, "ratio": h2 / h3})
        print(f"{b7:>10.0e} {lam_fit:>11.4f} {shift:>9.3f} {fmt_hours(h2):>14} {fmt_hours(h3):>12} {h2/h3:>12.2e}")
    out["table1_floor_sweep"] = t1
    print()

    # ------------------------------------------------ Table 2: shape of floor
    print("TABLE 2  Shape of the rare-event contribution (B7 = 1e-10, mu = 0.3)")
    print(f"{'kappa':>6} {'Lambda fit':>11} {'shift/SE':>9} {'O2 time':>14} {'O3 time':>12}")
    t2 = []
    for kappa in [-2, 0, 2]:
        lam_fit = fit_lambda([p1(d, 1e-10, kappa) for d in DISTANCES])
        shift = (LAM - lam_fit) / LAM_SE
        h2, h3 = o2_hours(1e-10, kappa), o3_hours(1e-10)
        t2.append({"kappa": kappa, "lambda_fit": lam_fit, "shift_in_SE": shift,
                   "O2_hours": h2, "O3_hours": h3})
        print(f"{kappa:>6} {lam_fit:>11.6f} {shift:>9.2e} {fmt_hours(h2):>14} {fmt_hours(h3):>12}")
    out["table2_shape"] = t2
    print()

    # ------------------------------------------------ Table 3: mapping sweep
    print("TABLE 3  Mapping parameter mu (B7 = 1e-10, kappa = 0): O3 time to decisive evidence")
    print("        mu = probability that one detected rare event causes a d=7 logical failure (ASSUMED)")
    t3 = []
    for mu in [0.01, 0.03, 0.1, 0.3, 1.0]:
        h3 = o3_hours(1e-10, mu=mu)
        t3.append({"mu": mu, "O3_hours": h3})
        print(f"  mu = {mu:<5}  O3: {fmt_hours(h3):>10}   (rare-event rate r = {1e-10/mu:.1e} per cycle)")
    print("  Note: smaller mu means more events per unit floor, so O3 is FASTER; the risk lies")
    print("  in mu itself being unknown, which O3 alone does not resolve (see README).")
    out["table3_mapping"] = t3
    print()

    # ------------------------------------------------ Table 4: decision divergence
    print("TABLE 4  Does the floor change the scaling decision?")
    t4 = []
    for target in TARGETS:
        ds = d_star(target)
        print(f"  Target {target:.0e} per cycle: M0 meets it at d* = {ds} "
              f"(p0 = {p0(ds):.2e}; {2*ds*ds-1} physical qubits per logical qubit in a rotated surface code)")
        for kappa in [-2, 0, 2]:
            for b7 in [1e-11, 1e-10, 1e-9]:
                pm1 = p1(ds, b7, kappa)
                met = pm1 <= target
                t4.append({"target": target, "d_star": ds, "kappa": kappa, "B7": b7,
                           "p1_at_dstar": pm1, "M1_meets_target": met})
                print(f"    kappa={kappa:>2}, B7={b7:.0e}: M1 gives {pm1:.2e} -> {'meets' if met else 'MISSES'} target")
    out["table4_decision"] = t4
    print()

    # ------------------------------------------------ Table 5: invisibility window
    print("TABLE 5  Invisibility window (kappa = 0): floors that change the decision")
    print("         yet shift fitted Lambda by less than one reported SE")
    # floor at which Lambda shift = 1 SE
    grid = np.logspace(-12, -3, 2000)
    shifts = np.array([(LAM - fit_lambda([p1(d, b, 0) for d in DISTANCES])) / LAM_SE for b in grid])
    b_vis = float(grid[np.argmax(shifts >= 1.0)])
    t5 = []
    for target in TARGETS:
        # smallest constant floor that makes M1 miss the target at M0's d*
        lower = target - p0(d_star(target))
        width = math.log10(b_vis / lower)
        t5.append({"target": target, "floor_lower": lower, "floor_visible": b_vis,
                   "window_decades": width})
        print(f"  Target {target:.0e}: floors from {lower:.0e} to {b_vis:.1e} -> "
              f"{width:.1f} decades invisible to Lambda but decision-changing")
    out["table5_window"] = t5
    print()

    # ------------------------------------------------ Table 6: demonstrability
    print("TABLE 6  Demonstrability: zero-failure run needed at 95% confidence (independent cycles)")
    t6 = []
    for target in TARGETS:
        n = -math.log(0.05) / target
        hours = n * T_CYCLE / 3600.0
        row = {"target": target, "cycles": n, "hours": hours, "pass_prob_if_M1": {}}
        ds = d_star(target)
        pass_m0 = math.exp(-n * p0(ds))
        row["pass_prob_if_M0"] = pass_m0
        print(f"  Target {target:.0e}: N = {n:.2e} cycles = {fmt_hours(hours)} at {T_CYCLE*1e6:.1f} us/cycle")
        print(f"    device at d* = {ds} under M0 (p = {p0(ds):.2e}): P(run passes) = {pass_m0:.3f}")
        for b7 in [1e-11, 1e-10, 1e-9]:
            pm1 = p1(ds, b7, 0)
            pass_prob = math.exp(-n * pm1)
            row["pass_prob_if_M1"][f"{b7:.0e}"] = pass_prob
            print(f"    same device under M1, floor {b7:.0e} (p = {pm1:.2e}): P(run passes) = {pass_prob:.3f}")
        # how far below the target a device must be to pass with 90% probability
        p_needed = -math.log(0.9) / n
        row["rate_for_90pct_pass"] = p_needed
        print(f"    to pass with 90% probability a device needs p <= {p_needed:.1e} "
              f"({target/p_needed:.0f} times better than the target)")
        t6.append(row)
    out["table6_demonstrability"] = t6
    print()

    # ------------------------------------------------ verdict
    ratio_at_1e10 = next(r["ratio"] for r in t1 if r["B7"] == 1e-10)
    shift_at_1e10 = next(r["shift_in_SE"] for r in t1 if r["B7"] == 1e-10)
    verdict = (shift_at_1e10 < 1.0) and (ratio_at_1e10 > 100) and all(w["window_decades"] > 1 for w in t5)
    out["verdict"] = {"PASS": bool(verdict),
                      "lambda_shift_SE_at_1e-10": shift_at_1e10,
                      "O2_over_O3_time_ratio_at_1e-10": ratio_at_1e10}
    print("SUMMARY CHECK (Lambda blind, events visible, window > 1 decade):", "PASS" if verdict else "FAIL")
    print(f"  At a Willow-like floor of 1e-10: Lambda shifts by {shift_at_1e10:.1e} SE;")
    print(f"  O2 needs {ratio_at_1e10:.1e} times longer than O3 to reach 20:1 odds.")

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "lambda_blindness_check_results.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
