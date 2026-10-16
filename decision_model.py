#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_model.py
=================

Version:
    0.4.0 (2026-10-16)

Supplementary research code for:

    Peter Kahl, 'What a Quantum Milestone Warrants: Error Correction, Evidence
    and Scaling Decisions' (2026), §6.1-6.7 and Figure 6.1.

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
A toy decision model. A company has demonstrated a below-threshold logical
memory at small code distances and must decide whether to commit to a much
larger one. Before committing, it may buy evidence. This script computes what
each candidate experiment is worth to that decision, as the expected value of
sample information (EVSI), and compares it with the value of perfect and of
partial perfect information.

This is NOT a model of Google's Willow processor or of any real device.
Willow-like magnitudes (Google Quantum AI and Collaborators, Nature 638 (2025)
920-926, doi:10.1038/s41586-024-08449-y) are used only so that orders of
magnitude are realistic. Every number printed is a synthetic consequence of
the assumptions below.

Hypotheses
----------
Rare correlated events occur under every hypothesis; hypotheses differ in what
the events do. A hypothesis theta = (r, mu, kappa) gives the logical error per
cycle at code distance d as

    p(d | theta) = P7 * LAM**((7 - d)/2)  +  r * q(d),
    q(d)         = min(1, mu * (d/7)**kappa),

where
    r      rare events per cycle                     {1e-11, 1e-10, 1e-9}
    mu     probability that one event causes a       {0.01, 0.1, 1}
           logical failure in the distance-7 memory
    kappa  how that probability changes with         {-2, 0, +2}
           distance (-2: larger codes absorb events;
           +2: larger patches intercept more)

q(d) is a probability, so it is capped at 1. The prior is uniform over the
27 hypotheses unless a case says otherwise.

Decision
--------
Target: 1e-9 logical errors per cycle. Under pure suppression the standard
design needs distance d* = 45.

    build standard    cost C_std = 40; pays V = 100 if p(d* | theta) <= target
    build mitigated   cost C_mit = 60; pays V if the residual rare-event
                      contribution (times mit_factor) still meets the target;
                      with mit_factor = 0 it always does
    do not commit     0 (included for completeness; never chosen)

Units are arbitrary.

Experiments (equal machine time T hours)
----------------------------------------
    O2   headline metric: more logical-failure counts at d = 3, 5, 7
         (best case: the amplitude of suppression is treated as known)
    O3   count events: a long run counting detectable rare events (informs r)
    O4a  tag events at d = 7: count events and record which ones cause a
         logical failure (informs r and mu)
    O4b  O4a plus a per-event 'footprint' signal that informs kappa; valid
         ONLY if a validated mechanism model links footprint to scaling.
         Stipulated; reported for comparison, not used in the paper.
    O5   tag events at d = 7 and d = 15 on one device at once (informs kappa
         through the ratio of the two failure probabilities; needs about 550
         qubits, hence a possible set-up cost)

Method
------
EVSI by preposterior Monte Carlo: for each hypothesis, simulate M datasets,
update the posterior over the grid, choose the Bayes action for each dataset,
and score that action under the hypothesis that generated the data. Misleading
datasets therefore count against the experiment. Net value subtracts machine
time at COST_PER_HOUR and any set-up cost. Seed fixed for reproducibility.

The partial value of perfect information for a parameter (or group) is
computed exactly over the discrete grid: group hypotheses by the value(s) of
that parameter, choose the best action within each group, and compare with the
best action under the prior.

Usage
-----
    python3 decision_model.py | tee output/decision_model_output.txt

Writes output/decision_model_results.json (or the path given as the first
argument). Run time about 90 seconds.
"""

import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

__version__ = "0.4.0"

# ============================================================================
# Fixed parameters (Willow-like magnitudes; see module docstring)
# ============================================================================

SEED = 20261016          # Monte Carlo seed
P7 = 1.43e-3             # logical error per cycle at distance 7
LAM = 2.14               # suppression factor per increase of 2 in distance
T_CYCLE = 1.1e-6         # seconds per error-correction cycle
CPH = 3600.0 / T_CYCLE   # cycles per machine-hour

T_GRID = [1, 3, 10, 30, 100, 300]           # machine-hours compared
OPTIONS = ["O2", "O3", "O4a", "O4b", "O5"]
OPTION_NAMES = {
    "O2": "headline metric only (more logical-failure counts at d=3,5,7)",
    "O3": "count events",
    "O4a": "tag events at d=7 (rate and failure probability)",
    "O4b": "O4a + footprint, valid only with a mechanism model (stipulated)",
    "O5": "tag events at d=7 and d=15 together (needs larger device)",
}
ACTIONS = ["build standard", "build mitigated", "do not commit"]


def p_base(d):
    """Logical error per cycle at distance d under pure suppression."""
    return P7 * LAM ** ((7.0 - d) / 2.0)


# ============================================================================
# The model
# ============================================================================

class Model:
    """Hypothesis grid, prior, utilities and value-of-information methods.

    Keyword arguments set one case (see CASES below). Defaults give the
    reference case.
    """

    def __init__(self, target=1e-9, r_grid=(1e-11, 1e-10, 1e-9),
                 mu_grid=(0.01, 0.1, 1.0), kappa_grid=(-2.0, 0.0, 2.0),
                 kappa_weights=None, V=100.0, c_std=40.0, c_mit=60.0,
                 mit_factor=0.0, eta=1.0, r_bg=None, sigma_foot=3.0,
                 cost_per_hour=0.02, setup=None):
        self.target = target

        # Distance the standard design needs under pure suppression (odd d).
        d = 7
        while p_base(d) > target:
            d += 2
        self.d_star = d

        # Hypothesis grid and prior (uniform, or weighted on kappa).
        self.thetas = list(itertools.product(r_grid, mu_grid, kappa_grid))
        self.kappa_grid = list(kappa_grid)
        if kappa_weights is None:
            w = np.ones(len(self.thetas))
        else:
            w = np.array([kappa_weights[self.kappa_grid.index(t[2])]
                          for t in self.thetas], float)
        self.prior = w / w.sum()

        # Experiment parameters.
        self.eta = eta                                   # event detection efficiency
        self.r_bg = r_bg if r_bg is not None else 1.0 / (100 * CPH)  # false events: 1 per 100 h
        self.sigma_foot = sigma_foot                     # footprint noise (O4b only)
        self.cost_per_hour = cost_per_hour               # machine time and delay
        self.setup = setup or {}                         # set-up cost per option

        # Does each design meet the target under each hypothesis?
        self.meets_std = np.array([self.p(self.d_star, t) <= target
                                   for t in self.thetas])
        self.meets_mit = np.array([p_base(self.d_star) + mit_factor * self.extra(self.d_star, t)
                                   <= target for t in self.thetas])

        # Utility matrix U[action, hypothesis].
        self.U = np.vstack([V * self.meets_std - c_std,
                            V * self.meets_mit - c_mit,
                            np.zeros(len(self.thetas))])

    # ------------------------------------------------------------------
    # Physics of the toy model
    # ------------------------------------------------------------------
    @staticmethod
    def q(d, th):
        """Per-event probability of a logical failure at distance d."""
        _, mu, ka = th[0], th[1], th[2]
        return min(1.0, mu * (d / 7.0) ** ka)

    def extra(self, d, th):
        """Rare-event contribution to logical error per cycle at distance d."""
        return th[0] * self.q(d, th)

    def p(self, d, th):
        """Total logical error per cycle at distance d under hypothesis th."""
        return p_base(d) + self.extra(d, th)

    # ------------------------------------------------------------------
    # Experiments: simulation and likelihood
    # ------------------------------------------------------------------
    def simulate(self, opt, th, T, rng, M):
        """Simulate M datasets of option opt, T machine-hours, under th."""
        r, ka = th[0], th[2]
        C = T * CPH                                   # cycles run
        if opt == "O2":
            # logical-failure counts at d = 3, 5, 7; machine time split equally
            return np.stack([rng.poisson((C / 3) * self.p(d, th), size=M)
                             for d in (3, 5, 7)], axis=1)
        if opt == "O3":
            # detected events plus false-event background
            return rng.poisson((self.eta * r + self.r_bg) * C, size=M)[:, None]
        # tagged options: n events, k7 of which cause a distance-7 failure
        n = rng.poisson(self.eta * r * C, size=M)
        k7 = rng.binomial(n, self.q(7, th))
        if opt == "O4a":
            return np.stack([n, k7], axis=1)
        if opt == "O4b":
            # mean footprint over n events, centred on kappa (stipulated link)
            sbar = np.where(n > 0,
                            rng.normal(ka, self.sigma_foot / np.sqrt(np.maximum(n, 1))),
                            np.nan)
            return np.stack([n, k7, sbar], axis=1)
        if opt == "O5":
            # the same events seen by both memories; failures conditionally
            # independent given the event
            k15 = rng.binomial(n, self.q(15, th))
            return np.stack([n, k7, k15], axis=1)
        raise ValueError(opt)

    def loglik(self, opt, data, T):
        """Log-likelihood of each dataset (rows) under each hypothesis (columns)."""
        C = T * CPH
        L = np.zeros((data.shape[0], len(self.thetas)))
        for j, th in enumerate(self.thetas):
            r = th[0]
            if opt == "O2":
                # Gaussian approximation to the Poisson counts
                for i, d in enumerate((3, 5, 7)):
                    m = (C / 3) * self.p(d, th)
                    L[:, j] += -0.5 * (data[:, i] - m) ** 2 / m - 0.5 * math.log(m)
                continue
            if opt == "O3":
                L[:, j] = stats.poisson.logpmf(data[:, 0], (self.eta * r + self.r_bg) * C)
                continue
            n, k7 = data[:, 0], data[:, 1]
            L[:, j] = (stats.poisson.logpmf(n, self.eta * r * C)
                       + stats.binom.logpmf(k7, n, self.q(7, th)))
            if opt == "O4b":
                s, has = data[:, 2], n > 0
                L[has, j] += stats.norm.logpdf(s[has], th[2], self.sigma_foot / np.sqrt(n[has]))
            if opt == "O5":
                L[:, j] += stats.binom.logpmf(data[:, 2], n, self.q(15, th))
        return L

    # ------------------------------------------------------------------
    # Value of information
    # ------------------------------------------------------------------
    def evsi(self, opt, T, M=1500, seed=SEED):
        """Expected value of sample information of option opt after T hours."""
        rng = np.random.default_rng(seed)
        pr, U = self.prior, self.U
        base = (U @ pr).max()                         # best expected utility now
        total = 0.0
        for j, th in enumerate(self.thetas):
            data = self.simulate(opt, th, T, rng, M)
            # posterior over the grid for each simulated dataset
            L = self.loglik(opt, data, T) + np.log(pr)
            L -= L.max(axis=1, keepdims=True)
            post = np.exp(L)
            post /= post.sum(axis=1, keepdims=True)
            # Bayes action per dataset, scored under the true hypothesis j
            chosen = (post @ U.T).argmax(axis=1)
            total += pr[j] * U[chosen, j].mean()
        return total - base

    def evpi(self):
        """Expected value of perfect information."""
        return float((self.prior * self.U.max(axis=0)).sum() - (self.U @ self.prior).max())

    def partial_evpi(self, key):
        """Partial value of perfect information about key(theta), exact on the grid."""
        groups = defaultdict(list)
        for j, th in enumerate(self.thetas):
            groups[key(th)].append(j)
        val = sum((self.U[:, idx] @ self.prior[idx]).max() for idx in groups.values())
        return float(val - (self.U @ self.prior).max())


# ============================================================================
# Cases reported in the paper (§6.6) and the README
# ============================================================================

CASES = [
    ("A", "Reference: uniform prior, V=100, C_std=40, C_mit=60, 0.02/h, no set-up", {}),
    ("B", "O5 needs a larger device: set-up 5", dict(setup={"O5": 5.0})),
    ("C", "O5 set-up 10", dict(setup={"O5": 10.0})),
    ("D", "Far from the decision boundary: C_mit = 85", dict(c_mit=85.0)),
    ("E", "Partial mitigation: removes 90% of the rare-event effect", dict(mit_factor=0.1)),
    ("E2", "Partial mitigation: removes 50% of the rare-event effect", dict(mit_factor=0.5)),
    ("F", "Narrower kappa range (-1, 0, +1)", dict(kappa_grid=(-1.0, 0.0, 1.0))),
    ("G", "Wider kappa range (-3, 0, +3)", dict(kappa_grid=(-3.0, 0.0, 3.0))),
    ("H", "Half detection efficiency (eta = 0.5)", dict(eta=0.5)),
    ("I", "Prior weighted to kappa = +2 (3:1:1)", dict(kappa_weights=(1, 1, 3))),
    ("J", "Mitigation cheaper (C_mit = 50)", dict(c_mit=50.0)),
    ("K", "Mitigation dearer (C_mit = 70)", dict(c_mit=70.0)),
]


def report(key, label, m, out, options=OPTIONS):
    """Print and record one case."""
    eu = m.U @ m.prior
    res = {"key": key, "label": f"{key}  {label}", "d_star": m.d_star,
           "P_std_meets": float(m.prior @ m.meets_std),
           "EU": dict(zip(ACTIONS, map(float, eu))),
           "best_prior_action": ACTIONS[int(eu.argmax())],
           "EVPI": m.evpi(),
           "partial_EVPI_r_mu": m.partial_evpi(lambda t: (t[0], t[1])),
           "partial_EVPI_kappa": m.partial_evpi(lambda t: t[2]),
           "options": {}}
    print(f"  P(standard design meets target) = {res['P_std_meets']:.3f}; "
          + "; ".join(f"{a} {v:.1f}" for a, v in zip(ACTIONS, eu))
          + f" -> {res['best_prior_action']}")
    print(f"  Value of perfect information {res['EVPI']:.2f}; of knowing (r, mu) only "
          f"{res['partial_EVPI_r_mu']:.2f}; of knowing kappa only {res['partial_EVPI_kappa']:.2f}")
    print(f"  {'option':<5}" + "".join(f"{('T=' + str(t) + 'h'):>8}" for t in T_GRID) + "   best net")
    for opt in options:
        vals = [m.evsi(opt, T) for T in T_GRID]
        nets = [v - m.cost_per_hour * T - m.setup.get(opt, 0.0) for v, T in zip(vals, T_GRID)]
        k = int(np.argmax(nets))
        res["options"][opt] = {"EVSI": dict(zip(map(str, T_GRID), map(float, vals))),
                               "best_T": T_GRID[k], "best_net": float(nets[k]),
                               "setup": m.setup.get(opt, 0.0)}
        print(f"  {opt:<5}" + "".join(f"{v:>8.2f}" for v in vals)
              + f"   {nets[k]:6.2f} at {T_GRID[k]}h")
    out.append(res)
    print()


def main():
    print(f"decision_model.py v{__version__}  (What a Quantum Milestone Warrants, §6)")
    print("Toy model. NOT a model of Willow or of any real device. Every number below")
    print("is a synthetic consequence of the stated assumptions.")
    print("\nOptions:")
    for k, v in OPTION_NAMES.items():
        print(f"  {k:<4} {v}")
    print()

    # ---------------------------------------------------------------- cases
    cases = []
    for key, label, kw in CASES:
        print(f"CASE {key}  {label}")
        report(key, label, Model(**kw), cases)

    # ------------------------------------- hypotheses on which the standard fails
    m = Model()
    print("Hypotheses where the standard design misses the target (reference case):")
    failing = []
    for t in m.thetas:
        if not m.p(m.d_star, t) <= m.target:
            failing.append({"r": t[0], "mu": t[1], "kappa": t[2], "q_at_dstar": m.q(m.d_star, t)})
            print(f"  r={t[0]:.0e} mu={t[1]} kappa={t[2]:+.0f}  "
                  f"per-event failure prob at d={m.d_star}: {m.q(m.d_star, t):.2f}")
    print()

    # ------------------------------------------------ Monte Carlo stability
    print("Monte Carlo stability at T=100h (seeds 1-3):")
    stab = {}
    for opt in ["O3", "O4a", "O5"]:
        vals = [m.evsi(opt, 100, seed=s) for s in (1, 2, 3)]
        stab[opt] = vals
        print(f"  {opt}: " + ", ".join(f"{v:.2f}" for v in vals))

    # ---------- why an untagged count at intermediate distance cannot see kappa
    print("\nVisibility of the rare-event contribution in untagged logical error")
    vis = []
    for d in (7, 15, 25, 45):
        worst = max(m.extra(d, t) for t in m.thetas)
        vis.append({"d": d, "p_base": p_base(d), "max_extra": worst, "ratio": worst / p_base(d)})
        print(f"  d={d:>2}: ordinary {p_base(d):.1e}; largest rare-event contribution {worst:.1e} "
              f"({worst / p_base(d):.1e} of ordinary)")

    # ---------------------------------------------------------------- save
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "decision_model_results.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"version": __version__, "cases": cases, "failing_hypotheses": failing,
                   "stability": stab, "visibility": vis, "seed": SEED, "T_GRID": T_GRID},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
