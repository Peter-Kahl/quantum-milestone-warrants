#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_checks.py
====================

Version:
    0.4.0 (2026-10-16)

Supplementary research code for:

    Peter Kahl, 'What a Quantum Milestone Warrants: Error Correction, Evidence
    and Scaling Decisions' (2026), §6.6 (findings 3-5 and 'Coincidences in
    tagging').

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
Three robustness checks on the toy model in decision_model.py. Toy model; NOT
a model of Willow or of any real device.

  P  Partial value of perfect information for each parameter separately, and
     for (r, mu) together: which parameter does the decision hinge on?

  S  Scaling break. decision_model.py assumes one exponent kappa governs how
     the per-event failure probability changes from distance 7 all the way to
     45, so an experiment at distances 7 and 15 fixes it to 45 by assumption.
     Here the exponent may change at distance 15: k1 governs 7 to 15 (visible
     to a two-distance experiment), k2 governs 15 to 45 (visible to no
     experiment at distance 15 or below). Each takes -2, 0 or +2: 81
     hypotheses. The value no experiment at d <= 15 can reach is the value of
     perfect information minus the partial value of knowing (r, mu, k1).

  M  Misattribution. Tagging matches each event to the logical failures it
     causes. An ordinary failure that happens to fall inside an event's
     attribution window of w cycles is mistaken for an event-caused one, with
     probability about f(d) = 1 - (1 - p_ordinary(d))**w. The observed
     event-caused failure probability is then q + (1 - q) f. The analyst either
     models f correctly or ignores it (assumes f = 0).

Usage
-----
    python3 robustness_checks.py | tee output/robustness_checks_output.txt

Writes output/robustness_checks_results.json (or the path given as the first
argument). Run time about 15 seconds.
"""

import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

import decision_model as dm

__version__ = "0.4.0"

M_SIM = 800     # simulated datasets per hypothesis (smaller than the main model's 1,500)


# ============================================================================
# S: scaling break at distance 15
# ============================================================================

class BreakModel(dm.Model):
    """Hypotheses (r, mu, k1, k2): the scaling exponent changes at d = 15.

    The mitigated design is assumed to meet the target under every hypothesis,
    as in the reference case of decision_model.py.
    """

    def __init__(self, c_mit=60.0, **kw):
        super().__init__(c_mit=c_mit, **kw)
        self.thetas = list(itertools.product((1e-11, 1e-10, 1e-9), (0.01, 0.1, 1.0),
                                             (-2.0, 0.0, 2.0), (-2.0, 0.0, 2.0)))
        n = len(self.thetas)
        self.prior = np.ones(n) / n
        self.meets_std = np.array([self.p(self.d_star, t) <= self.target for t in self.thetas])
        self.meets_mit = np.ones(n, bool)
        self.U = np.vstack([100.0 * self.meets_std - 40.0,
                            np.full(n, 100.0 - c_mit),
                            np.zeros(n)])

    @staticmethod
    def q(d, th):
        """Per-event failure probability, with exponent k1 to d = 15 and k2 beyond."""
        mu, k1 = th[1], th[2]
        k2 = th[3] if len(th) > 3 else k1
        if d <= 15:
            return min(1.0, mu * (d / 7.0) ** k1)
        return min(1.0, mu * (15 / 7.0) ** k1 * (d / 15.0) ** k2)


# ============================================================================
# M: misattribution of ordinary failures to events
# ============================================================================

def false_rate(d, window):
    """Probability that an attribution window of `window` cycles catches an ordinary failure."""
    return 1.0 - (1.0 - dm.p_base(d)) ** window


class MisattribModel(dm.Model):
    """Data generated with coincidence rate f; the analyst assumes f (knows) or 0 (ignores)."""

    def __init__(self, window=10, analyst_knows=True, **kw):
        super().__init__(**kw)
        self.f = {7: false_rate(7, window), 15: false_rate(15, window)}
        self.fa = self.f if analyst_knows else {7: 0.0, 15: 0.0}

    def _obs(self, d, th, f):
        """Observed probability that an event is tagged with a failure at distance d."""
        qd = self.q(d, th)
        return qd + (1.0 - qd) * f[d]

    def simulate(self, opt, th, T, rng, M):
        if opt not in ("O4a", "O5"):
            return super().simulate(opt, th, T, rng, M)
        n = rng.poisson(self.eta * th[0] * T * dm.CPH, size=M)
        k7 = rng.binomial(n, self._obs(7, th, self.f))
        if opt == "O4a":
            return np.stack([n, k7], axis=1)
        k15 = rng.binomial(n, self._obs(15, th, self.f))
        return np.stack([n, k7, k15], axis=1)

    def loglik(self, opt, data, T):
        if opt not in ("O4a", "O5"):
            return super().loglik(opt, data, T)
        C = T * dm.CPH
        L = np.zeros((data.shape[0], len(self.thetas)))
        for j, th in enumerate(self.thetas):
            n = data[:, 0]
            L[:, j] = stats.poisson.logpmf(n, self.eta * th[0] * C)
            L[:, j] += stats.binom.logpmf(data[:, 1], n, self._obs(7, th, self.fa))
            if opt == "O5":
                L[:, j] += stats.binom.logpmf(data[:, 2], n, self._obs(15, th, self.fa))
        return L


# ============================================================================
# Main
# ============================================================================

def pct(v, ref):
    """Value and its share of a reference value, e.g. ' 7.88 ( 80%)'."""
    return f"{v:5.2f} ({100 * v / ref:3.0f}%)"


def main():
    print(f"robustness_checks.py v{__version__}  (What a Quantum Milestone Warrants, §6.6)")
    print("Toy model. NOT a model of Willow or of any real device.\n")
    out = {"version": __version__}

    # ------------------------------------------------------------- Table P
    print("TABLE P  Partial value of perfect information, single power law (decision_model.py)")
    rows = []
    for label, kw in [("Reference", {}), ("Far from boundary (C_mit = 85)", dict(c_mit=85.0))]:
        m = dm.Model(**kw)
        e = m.evpi()
        row = {"case": label, "EVPI": e,
               "r": m.partial_evpi(lambda t: t[0]),
               "mu": m.partial_evpi(lambda t: t[1]),
               "kappa": m.partial_evpi(lambda t: t[2]),
               "r_mu": m.partial_evpi(lambda t: (t[0], t[1]))}
        rows.append(row)
        print(f"  {label}: perfect {e:.2f}; r alone {row['r']:.2f}; mu alone {row['mu']:.2f}; "
              f"kappa alone {row['kappa']:.2f}; r and mu {row['r_mu']:.2f}")
    out["partial"] = rows
    print()

    # ------------------------------------------------------------- Table S
    print("TABLE S  Scaling may change beyond distance 15 (81 hypotheses)")
    rows = []
    for label, cm in [("Reference", 60.0), ("Far from boundary (C_mit = 85)", 85.0)]:
        m = BreakModel(c_mit=cm)
        e = m.evpi()
        row = {"case": label, "EVPI": e, "P_std_meets": float(m.prior @ m.meets_std),
               "partial_r_mu_k1": m.partial_evpi(lambda t: (t[0], t[1], t[2])),
               "partial_k2": m.partial_evpi(lambda t: t[3])}
        print(f"  {label}: perfect {e:.2f}; P(standard meets) {row['P_std_meets']:.3f}; "
              f"knowing everything a d<=15 experiment can see {row['partial_r_mu_k1']:.2f}")
        for opt in ("O3", "O4a", "O5"):
            v = m.evsi(opt, 100, M=M_SIM)
            row[opt] = v
            print(f"    {opt} at 100 h: {pct(v, e)}")
        row["unreachable"] = e - row["partial_r_mu_k1"]
        print(f"    unreachable by any experiment at d <= 15: {pct(row['unreachable'], e)}")
        rows.append(row)
    out["scaling_break"] = rows
    print()

    # ------------------------------------------------------------- Table M
    print("TABLE M  Misattribution of ordinary failures to events (reference case, 100 h)")
    rows = []
    ref = dm.Model()
    base = {opt: ref.evsi(opt, 100, M=M_SIM) for opt in ("O3", "O4a", "O5")}
    print(f"  No misattribution: O3 {base['O3']:.2f}, O4a {base['O4a']:.2f}, O5 {base['O5']:.2f}")
    for window in (10, 100):
        for knows in (True, False):
            m = MisattribModel(window=window, analyst_knows=knows)
            row = {"window_cycles": window, "analyst_models_f": knows,
                   "f7": m.f[7], "f15": m.f[15],
                   "O4a": m.evsi("O4a", 100, M=M_SIM), "O5": m.evsi("O5", 100, M=M_SIM)}
            rows.append(row)
            print(f"  window {window:>3} cycles (f at d=7 {m.f[7]:.3f}, d=15 {m.f[15]:.4f}); "
                  f"analyst {'models' if knows else 'ignores'} it: "
                  f"O4a {row['O4a']:.2f}, O5 {row['O5']:.2f}")
    out["misattribution"] = {"baseline": base, "rows": rows}

    # ---------------------------------------------------------------- save
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "robustness_checks_results.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
