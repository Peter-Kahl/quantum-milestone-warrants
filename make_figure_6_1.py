#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure_6_1.py
==================

Version:
    0.4.0 (2026-10-16)

Supplementary research code for:

    Peter Kahl, 'What a Quantum Milestone Warrants: Error Correction, Evidence
    and Scaling Decisions' (2026), Figure 6.1.

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
Draws Figure 6.1: the expected value of sample information of each experiment
against machine time, in the reference case (case A) and far from the
decision boundary (case D). Reads the results written by decision_model.py;
computes nothing itself.

Design notes: four categorical colours in fixed order (checked for colour-
vision-deficiency separation); each line also has its own marker and a direct
label, so identity never depends on colour alone. Lines that end within 0.3
units of each other share one label ('A = B').

Usage
-----
    python3 decision_model.py          # writes output/decision_model_results.json
    python3 make_figure_6_1.py         # writes output/figure_6_1.png and .svg

Requires matplotlib.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                  # no display needed
import matplotlib.pyplot as plt       # noqa: E402

__version__ = "0.4.0"

# ============================================================================
# Inputs and style
# ============================================================================

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "decision_model_results.json")
OUT_DIR = "output"
T = [1, 3, 10, 30, 100, 300]           # machine-hours (must match decision_model.T_GRID)

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COL = {"O2": "#2a78d6", "O3": "#eb6834", "O4a": "#1baf7a", "O5": "#eda100"}
NAME = {"O2": "Headline metric", "O3": "Count events",
        "O4a": "Tag events at d = 7", "O5": "Tag events at d = 7 and 15"}
MARK = {"O2": "o", "O3": "s", "O4a": "^", "O5": "D"}
PANELS = [("A", "Reference case (near the decision boundary)"),
          ("D", "Far from the decision boundary")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def direct_labels(ends):
    """Group lines whose end values lie within 0.3 units; return [(y, text)], top first."""
    order = ["O5", "O4a", "O3", "O2"]
    labels, used = [], set()
    for a in order:
        if a in used:
            continue
        group = [a] + [b for b in order if b != a and b not in used and abs(ends[a] - ends[b]) < 0.3]
        used.update(group)
        txt = " = ".join(NAME[g].lower() if i else NAME[g] for i, g in enumerate(group))
        labels.append([ends[a], txt])
    labels.sort(key=lambda lab: -lab[0])
    return labels


def main():
    with open(RESULTS) as f:
        R = json.load(f)
    cases = {c["key"]: c for c in R["cases"]}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), facecolor=SURF)
    for ax, (key, title) in zip(axes, PANELS):
        c = cases[key]
        evpi = c["EVPI"]
        span = evpi * 1.15                         # y-axis headroom
        ax.set_facecolor(SURF)

        # value of perfect information: dashed ceiling, labelled just above
        ax.axhline(evpi, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(1.05, evpi + 0.03 * span, f"Perfect information {evpi:.1f}",
                color=INK2, fontsize=7.5, va="bottom", ha="left")

        # one line per experiment (small negative Monte Carlo values clipped to 0)
        ends = {}
        for opt in ["O2", "O3", "O4a", "O5"]:
            y = [max(0.0, c["options"][opt]["EVSI"][str(t)]) for t in T]
            ax.plot(T, y, color=COL[opt], lw=2, marker=MARK[opt], ms=6,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=NAME[opt], zorder=3)
            ends[opt] = y[-1]

        # direct labels at the right end, nudged apart if they would collide
        last = None
        for lab in direct_labels(ends):
            if last is not None and last - lab[0] < 0.07 * span:
                lab[0] = last - 0.07 * span
            last = lab[0]
            ax.text(380, lab[0], lab[1], fontsize=8, color=INK, va="center", ha="left")

        # axes
        ax.set_xscale("log")
        ax.set_xticks(T)
        ax.minorticks_off()
        ax.set_xticklabels([str(t) for t in T])
        ax.set_xlim(0.8, 340)
        ax.set_ylim(-0.4, span)
        ax.set_xlabel("Machine time (hours)")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Expected value of the experiment\n(arbitrary units)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, 0.97, "Figure 6.1  Value of each experiment against machine time "
             "(toy decision model; synthetic values)", fontsize=10, color=INK, va="top")
    fig.tight_layout(rect=(0, 0.07, 0.86, 0.93))

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in (("png", {"dpi": 200}), ("svg", {})):
        fig.savefig(os.path.join(OUT_DIR, f"figure_6_1.{ext}"), facecolor=SURF,
                    bbox_inches="tight", **kw)
    print("saved output/figure_6_1.png and output/figure_6_1.svg")


if __name__ == "__main__":
    main()
