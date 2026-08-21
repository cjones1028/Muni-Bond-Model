"""Profile the holdout error tail: how big is it, and what kinds of bonds
live in it. Reproduces the exact seed-42 holdout split used in training."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
dfF = mm.build_features(df)
bundle = mm.load_bundle(HERE / 'model.joblib')

rng = np.random.RandomState(42)
test = rng.rand(len(dfF)) < 0.2
h = dfF[test].copy()
X = mm._prep_matrix(h, bundle['numeric'], bundle['categorical'], bundle['categories'])
h['err'] = (mm.predict_bundle(bundle, X) - h['target_yield'].to_numpy()) * 100
h['abs_err'] = h['err'].abs()

e = h['abs_err']
print(f"n={len(h):,}  mean {e.mean():.1f}  median {e.median():.1f}")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(e, p):6.1f} bps")
for cut in [10, 25, 50]:
    n = int((e > cut).sum())
    contrib = e[e > cut].sum() / e.sum() * 100
    print(f"  >{cut} bps: {n:5,} bonds ({100*n/len(h):4.1f}%) carrying {contrib:4.0f}% of total error")

tail = h[e > 25]
core = h[e <= 10]
print("\n--- tail (>25 bps) vs core (<=10 bps) profile ---")


def prof(name, col, fmt='{:.1f}', func='median'):
    try:
        tv = getattr(pd.to_numeric(tail[col], errors='coerce'), func)()
        cv = getattr(pd.to_numeric(core[col], errors='coerce'), func)()
        print(f"{name:34s} tail {fmt.format(tv):>10}   core {fmt.format(cv):>10}")
    except Exception as ex:
        print(f"{name}: {ex}")


prof('target yield (%)', 'target_yield')
prof('composite rating (1=AAA..22=D)', 'composite_rating')
prof('years to maturity', 'days_to_maturity_30_360', func='median')
prof('coupon (%)', 'current_coupon_rate')
prof('outstanding ($)', 'outstanding_amount', fmt='{:,.0f}')
prof('principal_factor', 'principal_factor', fmt='{:.3f}')

print("\nshare with rating worse than BBB- (>10):")
for nm, d in [('tail', tail), ('core', core)]:
    r = pd.to_numeric(d['composite_rating'], errors='coerce')
    print(f"  {nm}: {(r > 10).mean() * 100:.0f}%  (unrated/NaN {r.isna().mean()*100:.0f}%)")

print("\nshare with yield > 6%:")
for nm, d in [('tail', tail), ('core', core)]:
    print(f"  {nm}: {(d['target_yield'] > 6).mean() * 100:.0f}%")

print("\ntop sectors in tail:")
print(tail['purpose_class_desc'].value_counts().head(6).to_string())
print("\ntop states in tail:")
print(tail['incorporated_state_code_desc'].value_counts().head(5).to_string())

print("\n10 worst misses:")
cols = [c for c in ['target_yield', 'err', 'composite_rating', 'current_coupon_rate',
                    'days_to_maturity_30_360', 'purpose_class_desc',
                    'incorporated_state_code_desc'] if c in h]
print(h.nlargest(10, 'abs_err')[cols].round(2).to_string())
