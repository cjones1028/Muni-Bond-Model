"""Full audit of low-coupon handling, both sides of the model:

  A. SECONDARY (the model's own turf): signed error by coupon bucket on the
     held-out CUSIPs of the newest snapshot. A systematic bias here is a
     model defect and must be fixed in the model.
  B. WIRES (new-issue turf): every sub-5%% tranche across the archived deals,
     with its error, next to the deal's 5%% tranches at the same maturity.
     A one-directional gap here that does NOT appear in (A) is the
     retail/new-issue effect, not a model defect.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402

pd.set_option('display.width', 200)

# ---------------- A. holdout error by coupon bucket ----------------
print("A. HOLDOUT (secondary) signed error by coupon bucket")
snap = sorted((HERE / 'evals_archive').glob('ICE_Evals_*.csv'))[-1]
df = mm.clean_universe(mm.load_evals(snap))
bundle = mm.load_bundle(HERE / 'model.joblib')
held = df[df.index.isin(bundle.get('holdout_cusips', []))].copy()
F = mm.build_features(held)
X = mm._prep_matrix(F, bundle['numeric'], bundle['categorical'], bundle['categories'])
pred = mm.predict_bundle(bundle, X)
err = (pred - held['target_yield'].values) * 100
cpn = pd.to_numeric(held['current_coupon_rate'], errors='coerce').values

print(f"   holdout rows: {len(held)}  (snapshot {snap.name})")
print(f"   {'bucket':12s} {'n':>6} {'signed mean':>12} {'median':>8} {'MAE':>7}")
buckets = [('<3.75%', cpn < 3.75), ('3.75-4.25%', (cpn >= 3.75) & (cpn <= 4.25)),
           ('4.25-5%', (cpn > 4.25) & (cpn < 5)), ('5%', cpn == 5.0),
           ('>5%', cpn > 5.0)]
for tag, mask in buckets:
    m = mask & np.isfinite(err)
    if m.sum():
        print(f"   {tag:12s} {m.sum():>6} {err[m].mean():>+12.2f} "
              f"{np.median(err[m]):>+8.2f} {np.abs(err[m]).mean():>7.2f}")

# low-coupon bias by tenor (is any bias hiding in a tenor pocket?)
print("\n   low-coupon (3.75-4.25) signed error by years-to-maturity:")
yrs = pd.to_datetime(held['maturity_date'], errors='coerce')
tenor = (yrs - pd.Timestamp('2026-08-25')).dt.days.values / 365.25
lo = (cpn >= 3.75) & (cpn <= 4.25) & np.isfinite(err)
for a, b in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 35)]:
    m = lo & (tenor >= a) & (tenor < b)
    if m.sum():
        print(f"     {a:>2}-{b:<2}y  n={m.sum():>4}  signed {err[m].mean():+6.1f}  "
              f"MAE {np.abs(err[m]).mean():5.1f}")

# ---------------- B. every sub-5 tranche across archived deals ----------------
print("\nB. WIRES: every sub-5% tranche we have ever priced (newest run per deal)")
arch = HERE / 'wire_archive'
files = sorted(arch.glob('*.csv'))
newest = {}
for f in files:
    tag = f.name.rsplit('_', 2)[0]
    newest[tag] = f          # sorted -> last wins (newest timestamp)
rows = []
for tag, f in newest.items():
    d = pd.read_csv(f)
    if 'Coupon' not in d.columns or 'Error (bps)' not in d.columns:
        continue
    d['cpn'] = pd.to_numeric(d['Coupon'], errors='coerce')
    subs = d[d['cpn'] < 5.0]
    for _, r in subs.iterrows():
        # same-maturity 5% twin, if the deal has one
        twin = d[(d['Maturity'] == r['Maturity']) & (d['cpn'] == 5.0)]
        twin_err = twin['Error (bps)'].mean() if len(twin) else np.nan
        rows.append({'deal': tag[:28], 'maturity': r['Maturity'], 'coupon': r['cpn'],
                     'err_bps': r['Error (bps)'],
                     'same-mat 5% err': round(twin_err, 1) if np.isfinite(twin_err) else '--'})
if rows:
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    e = pd.to_numeric(out['err_bps'], errors='coerce')
    print(f"\n   sub-5 tranches: n={len(out)}  signed mean {e.mean():+.1f} bp  "
          f"median {e.median():+.1f}  |  positive(=rich-flagged): {(e > 0).sum()}/{len(out)}")
else:
    print("   (no sub-5% tranches found in archives)")
