"""Can we identify, IN ADVANCE, the bonds the model will miss badly?
Tests candidate trust filters on the holdout: if a filter works, the
'trusted' group should have a thin tail and the 'flagged' group should
contain most of the big errors."""
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

preds = np.array([m.predict(X) for m in bundle['models']])
h['pred'] = preds.mean(axis=0)
h['ens_std_bps'] = preds.std(axis=0) * 100          # model disagreement
h['abs_err'] = (h['pred'] - h['target_yield']).abs() * 100

# ---- candidate flags (all computable BEFORE knowing the true yield) ----
rating = pd.to_numeric(h['composite_rating'], errors='coerce')
flags = pd.DataFrame(index=h.index)
flags['unrated'] = rating.isna()
flags['junk_rated'] = rating > 10
flags['pred_yield_high'] = h['pred'] > 5.5           # uses MODEL yield, not true
flags['conduit'] = h['conduit_obligor_name_id'].notna()
flags['micro'] = pd.to_numeric(h['outstanding_amount'], errors='coerce') < 2_000_000
flags['ens_disagree'] = h['ens_std_bps'] > 5
flags['ultra_short'] = pd.to_numeric(h['days_to_maturity_30_360'], errors='coerce') < 540

for c in flags.columns:
    m = flags[c]
    print(f"{c:16s} flags {m.mean()*100:4.1f}% | mean err flagged {h.loc[m,'abs_err'].mean():6.1f} "
          f"vs unflagged {h.loc[~m,'abs_err'].mean():5.1f} bps")

# ---- combined trust filter ----
flagged = flags.any(axis=1)
t, f = h[~flagged], h[flagged]
print(f"\nCOMBINED: trusted {len(t):,} ({100*(~flagged).mean():.0f}%) | flagged {len(f):,}")
print(f"  trusted : mean {t['abs_err'].mean():.1f} | median {t['abs_err'].median():.1f} "
      f"| p99 {np.percentile(t['abs_err'],99):.1f} | max {t['abs_err'].max():.0f} bps")
print(f"  flagged : mean {f['abs_err'].mean():.1f} | median {f['abs_err'].median():.1f} "
      f"| p99 {np.percentile(f['abs_err'],99):.1f} | max {f['abs_err'].max():.0f} bps")

for cut in [25, 50, 100]:
    big = h['abs_err'] > cut
    if big.sum():
        cap = (big & flagged).sum() / big.sum() * 100
        print(f"  errors >{cut:>3} bps: {big.sum():4d} bonds, {cap:3.0f}% caught by the filter")

# what slips through?
missed = h[(h['abs_err'] > 25) & ~flagged]
print(f"\nworst UNFLAGGED misses ({len(missed)} bonds >25bps escaped the filter):")
cols = [c for c in ['target_yield','pred','abs_err','ens_std_bps','composite_rating',
                    'current_coupon_rate','days_to_maturity_30_360','purpose_class_desc'] if c in h]
print(missed.nlargest(8,'abs_err')[cols].round(2).to_string())
