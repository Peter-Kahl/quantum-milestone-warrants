# Changelog

All notable changes to this repository are recorded here. Versions before 0.4.0 were working versions and were not published.

## 0.4.0 — 2026-10-16

First release as a public repository. **Results are identical to 0.3.1**; every JSON result was checked against the 0.3.1 run.

### Repository

- Scripts renamed for publication: `wmw_gonogo.py` → `lambda_blindness_check.py`; `wmw_decision_v03.py` → `decision_model.py`; `wmw_v031_checks.py` → `robustness_checks.py`; `make_figure.py` → `make_figure_6_1.py`.
- Every script carries a header naming the paper and the sections it supports, the author, the repository, the version and date, and the licence (MIT), with fuller comments and docstrings.
- Outputs moved to `output/`, with file names matching the scripts. Scripts write their JSON results there by default.
- The superseded version 0.2.0 moved to `archive/v0.2.0/`, with a header in each file stating the error.
- The exploratory scaling-break script (`check_scaling_break.py`) is not included; Table S of `robustness_checks.py` replaces it and reproduces its results.
- Added `README.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE` and `requirements.txt`.

### `decision_model.py`

- The 50%-mitigation case (E2) and the list of hypotheses under which the standard design fails are now printed by the script itself. In 0.3.x they were produced separately (`output_v03_extra.txt`). Their values are unchanged.
- Each case in the JSON output carries a `key` field (`A`, `B`, … `K`, `E2`), which `make_figure_6_1.py` uses.

### `lambda_blindness_check.py`

- The closing line, formerly 'GO/NO-GO VERDICT', is now 'SUMMARY CHECK'. Calculations unchanged.

### `make_figure_6_1.py`

- The 'perfect information' label sits just above its dashed line, clear of the data. The figure title no longer carries a model version number.

## 0.3.1 — 2026-10-16 (working version)

Robustness checks: partial value of perfect information for each parameter (Table P); a change of scaling exponent beyond distance 15 (Table S); misattribution of ordinary failures to events (Table M). The decision model was changed only to accept hypotheses with extra parameters; its results were unchanged.

## 0.3.0 — 2026-10-16 (working version)

**Corrects a modelling error in 0.2.0.** Version 0.2.0 capped the product r · μ · (d/7)^κ at 1, instead of capping the per-event failure probability μ · (d/7)^κ. The rare-event contribution to the logical error per cycle could therefore reach 1, and the per-event probability exceed 1. From 0.3.0 the per-event probability q(d) = min{1, μ · (d/7)^κ} is capped and the contribution is r · q(d). Several 0.2.0 conclusions do not survive the correction, among them the claim that characterising events captures most of the value while counting them captures little.

Also: the characterisation experiment split into a realistic part (O4a, tag events at distance 7; O5, tag events at distances 7 and 15) and a stipulated part (O4b, footprint signal); O2 restricted to the headline metric; partial values of perfect information; new cases (far from the decision boundary, partial mitigation, narrower and wider κ ranges, half detection efficiency).

## 0.2.0 — 2026-10-16 (working version, superseded)

Full decision model: rare events under every hypothesis, preposterior value-of-information analysis, sensitivity checks. Contains the error corrected in 0.3.0; kept in `archive/v0.2.0/` as a record.

## 0.1.0 — 2026-10-16 (working version)

Go/no-go test: is the fitted suppression factor Λ sensitive to rare correlated events that would change a scaling decision? Now `lambda_blindness_check.py`.
