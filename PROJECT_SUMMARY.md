# Project Summary — Municipal New-Issue Pricing Model

Handoff brief. All figures below are measured, not projected, and are
reproducible from the files in this folder (test_suite.py, the comparison
scripts, and wire_archive/).

## What the project is

An automated system that prices new-issue municipal bonds directly from the
dealer's pricing wire (the Bloomberg message announcing a deal). It reads the
wire text, extracts every tranche, predicts each bond's fair-value yield with
a machine-learning model trained on the secondary market, compares to the
dealer's pricing, and flags which tranches are cheap, rich, or unreliable to
judge. End-to-end time: ~5 seconds per deal, no manual input.

Built August 2026 at Elequin Capital (muni desk) by Charlie Jones, using
Claude (Anthropic) as an AI coding collaborator.

## The problem it replaced

The prior workflow used a linear regression refit separately for each
issuer, with every bond's characteristics typed in by hand (coupon, maturity,
call date, price, size, ratings). It was accurate (~4-5 bps) only for large
issuers with many outstanding bonds, could not price issuers without that
history, took minutes of manual entry per deal, and was error-prone (a
hand-entry mistake was found in the original files).

## What was built (Python 3.12)

- Wire parser: turns dealer wire text into structured deal data; handles
  two dealers' formats (BofA, Loop Capital) plus common variants; loud
  failure on anything unrecognized (tranche-sum check).
- Pricing model: 3-model LightGBM gradient-boosted ensemble predicting ICE
  mid yield from bond structure, credit, and issuer identity, trained on
  ~83,000 tax-exempt fixed-rate bonds (ICE evaluated pricing feed).
- Auto-issuer matching from the wire text; credit-mismatch guard for deals
  that are a new credit of a known issuer.
- New-issue concession: measured from the deal archive (~14 bps on the first
  two deals), self-calibrating as deals accumulate.
- Confidence layer: per-tranche ensemble-disagreement score plus a trust
  filter (rating, conduit, size, tenor, yield level) validated to catch 89%
  of the model's >50 bp errors and 95% of >100 bp errors in advance.
- Risk-adjusted ranking: a single PROFIT ($/mm) figure per tranche = edge
  minus exit cost minus a half-sigma risk charge, times DV01.
- Secondary-market screener separating actionable names from research-only.
- 40-check automated test suite, fuzz tests for the parser, documented
  formulas (MATH.md), README with known limitations.
- Deployed locally (JupyterLab) and packaged for Anaconda.

## Results (measured)

- Universe holdout (16,482 bonds never seen in training, vs ICE marks):
  5.3 bps mean absolute error, 2.2 bps median.
- Head-to-head vs a faithful reconstruction of the prior linear approach,
  identical data/features/splits, 12 repeated simulations: new model won
  12/12 (5.7 vs 20.2 bps universe-wide) and was closer on 85% of individual
  bonds. On the old model's best case (single large issuer, matched
  cross-validation): 3.3 vs 4.0 bps.
- Live dealer wires: LA Metro I-105 toll deal ($509mm, 16 tranches) 2.5-2.8
  bps mean error vs dealer pricing; NYC TFA ($1.5B, 7 tranches, fully
  out-of-sample) 1.0-1.3 bps mean, worst tranche 3.0. Pooled over all 23
  real tranches: ~2.3 bps. ICE's evaluated pricing runs ~5 bps.
- Wire-to-priced-table runtime: ~5 seconds (prior process: minutes of manual
  entry per deal).

## Honest limitations (state these if asked)

- Live validation is 2 deals / 23 tranches so far; the concession estimate
  comes from those same deals. A larger live record is the next step.
- Trained on a single day's data snapshot; time-based validation requires
  accumulating snapshots (archiving is in place).
- The "prior model" in comparisons is a reconstruction from the original
  run scripts; the original notebook engine was not available.
- Parser coverage is two dealer formats plus tested variants; new dialects
  may need one-time extensions (failures are loud, not silent).
- Accuracy is measured against ICE marks and dealer pricing, not trade
  prints.

## Skills demonstrated

Fixed-income analytics (yield/price/DV01, day counts, call structures,
new-issue concession), machine learning (gradient boosting, ensembles,
feature selection by experiment, holdout/cross-validation design),
quantitative research discipline (reproducible experiments, negative results
documented, error decomposition, adverse-selection control), and software
engineering (parsing, pipelines, caching, automated testing, packaging and
deployment).
