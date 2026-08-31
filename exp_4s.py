"""Investigate the boss's flag: the 4% coupon tranches screen RICH -- is the
model's coupon penalty real (matches secondary) or a model artifact?

Three measurements, all against the newest eval snapshot:
  1. EMPIRICAL secondary spread: outstanding CA GO 4s vs 5s, matched by
     maturity year -- what does ICE actually mark the coupon spread at?
  2. MODEL accuracy on those same outstanding 4s: does the model reproduce
     ICE's marks on real 4% bonds, or does it overshoot the penalty?
  3. The wire comparison: dealer's 4s-vs-5s spread on the CA deal vs both.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402

pd.set_option('display.width', 200)

# newest snapshot
snap = sorted((HERE / 'evals_archive').glob('ICE_Evals_*.csv'))[-1]
print(f"snapshot: {snap.name}")
df = mm.clean_universe(mm.load_evals(snap))

# ---- CA GO universe ----
name = df['primary_name_abbreviated'].astype(str).str.upper()
exact = name == 'CALIFORNIA ST'
if exact.sum() >= 100:      # the GO issuer itself, no authorities/leases
    ca = df[exact].copy()
else:
    ca = df[name.str.contains('CALIFORNIA ST', na=False)
            & ~name.str.contains('UNIV|DEPT|PUB WKS|EARTHQ', na=False)].copy()
ca['mat_year'] = pd.to_datetime(ca['maturity_date'], errors='coerce').dt.year
ca['cpn'] = pd.to_numeric(ca['current_coupon_rate'], errors='coerce')
ca['yld'] = pd.to_numeric(ca['yield_mid'], errors='coerce')
print(f"CA GO bonds in snapshot: {len(ca)}  "
      f"(issuers: {sorted(ca['primary_name_abbreviated'].unique())[:3]}...)")

# ---- 1. empirical secondary coupon spread, matched maturity years ----
print("\n1. SECONDARY: outstanding CA GO yield by coupon, same maturity year")
print(f"{'year':>6} {'n(5s)':>6} {'5% yld':>8} {'n(4s)':>6} {'4-4.25 yld':>10} {'spread bp':>10}")
rows = []
for yr in range(2031, 2051):
    five = ca[(ca['mat_year'] == yr) & (ca['cpn'] == 5.0)]['yld']
    four = ca[(ca['mat_year'] == yr) & (ca['cpn'].between(3.75, 4.25))]['yld']
    if len(five) >= 2 and len(four) >= 1:
        sp = (four.mean() - five.mean()) * 100
        rows.append(sp)
        print(f"{yr:>6} {len(five):>6} {five.mean():>8.3f} {len(four):>6} "
              f"{four.mean():>10.3f} {sp:>10.1f}")
if rows:
    print(f"   median secondary 4s-over-5s spread: {np.median(rows):.1f} bp")

# ---- 2. model vs ICE on the real outstanding 4s ----
print("\n2. MODEL vs ICE on outstanding CA GO low-coupon bonds (does the model")
print("   reproduce the secondary, or overshoot the coupon penalty?)")
bundle = mm.load_bundle(HERE / 'model.joblib')
ca4 = ca[ca['cpn'].between(3.75, 4.5) & ca['yld'].notna()
         & (ca['mat_year'] >= 2031)].copy()
ca5 = ca[(ca['cpn'] == 5.0) & ca['yld'].notna() & (ca['mat_year'] >= 2031)].copy()
for tag, sub in [('4-4.5%', ca4), ('5%', ca5)]:
    F = mm.build_features(sub)
    X = mm._prep_matrix(F, bundle['numeric'], bundle['categorical'], bundle['categories'])
    pred = mm.predict_bundle(bundle, X)
    err = (pred - sub['yld'].values) * 100
    held = sub.index.isin(bundle.get('holdout_cusips', []))
    print(f"   {tag:7s} n={len(sub):4d}  model-minus-ICE: mean {err.mean():+6.1f} bp  "
          f"median {np.median(err):+6.1f}  |  holdout-only n={held.sum()}: "
          f"mean {err[held].mean():+6.1f}" if held.sum() else
          f"   {tag:7s} n={len(sub):4d}  model-minus-ICE: mean {err.mean():+6.1f} bp  "
          f"median {np.median(err):+6.1f}  (no holdout rows)")

# ---- 3. training-set support ----
print("\n3. TRAINING SUPPORT: coupon mix of the whole clean universe")
allcpn = pd.to_numeric(df['current_coupon_rate'], errors='coerce')
for lo, hi, tag in [(4.9, 5.1, '5s'), (3.75, 4.25, '4-4.25s'), (0, 3.74, '<3.75'),
                    (5.11, 99, '>5')]:
    n = ((allcpn >= lo) & (allcpn <= hi)).sum()
    print(f"   {tag:8s} {n:6d} bonds ({n/len(df)*100:4.1f}%)")

# ---- 4. wire spreads for reference ----
print("\n4. THE CA WIRE: dealer 4s-over-5s spread vs model's")
for yr, w5, w4, m5, m4 in [(2031, 2.93, 2.93, 2.7442, 2.8113),
                           (2034, 3.26, 3.32, 3.0652, 3.1997),
                           (2035, 3.35, 3.42, 3.1616, 3.3334),
                           (2037, 3.60, 3.67, 3.4176, 3.7958),
                           (2046, 4.31, 4.43, 4.2487, 4.4730)]:
    print(f"   {yr}: dealer spread {100*(w4-w5):+5.1f} bp | model spread {100*(m4-m5):+5.1f} bp")
