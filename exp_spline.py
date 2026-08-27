"""Spline experiment -- runs the moment spline_pull.py has produced data.

1. Evaluator-vs-evaluator: how far apart are Spline 1mm mids and ICE mids on
   the SAME bonds? (Context for any sub-1bp accuracy target: two professional
   evaluators' disagreement is a hard floor.)
2. Model experiment: baseline vs +spline_curve_yield feature (DV01-bucket
   curve level, new-issue-safe). Judged on 8/25 holdout + all four wires.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
import spline_features as sf
from Wire_Parser import parse_wire

try:
    curves = sf.load_curves()
    pricing = sf.load_pricing_1mm()
except FileNotFoundError as e:
    sys.exit(str(e))

df = mm.clean_universe(mm.load_evals(HERE / 'ICE_Evals.csv'))

# ---- 1. evaluator vs evaluator ----
joined = df.join(pricing, how='inner')
gap = (pd.to_numeric(joined['spline_mid_yield'], errors='coerce')
       - joined['target_yield']).dropna() * 100
print(f"Spline-1mm vs ICE, same bonds ({len(gap):,}): "
      f"mean abs {gap.abs().mean():.1f} bps | median {gap.abs().median():.1f} "
      f"| signed {gap.mean():+.1f}")

# ---- 2. curve feature experiment ----
_orig_build = mm.build_features


def build_with_spline(d):
    return sf.attach_curve_yield(_orig_build(d), curves)


WIRES = [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
         ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN'),
         ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY'),
         ('PDX', 'portland_wire.txt', 'PORTLAND ORE SWR')]

for label, patch, feats in [
        ('baseline', False, None),
        ('+spline_curve_yield', True, mm.NUMERIC_FEATURES + ['spline_curve_yield'])]:
    mm.build_features = build_with_spline if patch else _orig_build
    b = mm.train_yield_model(df, numeric=feats, n_ensemble=1, seed=42)
    parts = []
    for wname, wfile, iss in WIRES:
        deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
        templ = mm.template_from(df, issuer_contains=iss)
        r = mm.price_wire(deal, b, templ, concession_bps=7.0)
        parts.append(f"{wname} {r['Error (bps)'].abs().mean():.1f}")
    print(f">>> {label}: holdout {b['mae_bps']:.2f}/{b['median_bps']:.2f} | "
          f"wires: {' | '.join(parts)}")
mm.build_features = _orig_build
