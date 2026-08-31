"""Deep test of the low-coupon hypothesis:

The model's learned 4s-over-5s penalty could come mostly from LEGACY DISCOUNT
4s (de minimis / dollar-price aversion). New-issue 4s are par/premium bonds.
If the model is unbiased on 4s *on average* but overshoots on PREMIUM 4s
specifically, it is inherently wrong exactly where wires live.

Split outstanding low-coupon bonds by moneyness (yield_mid vs coupon):
  premium  : yield < coupon - 0.25   (trades above par -- like new-issue 4s)
  par-ish  : |yield - coupon| <= 0.25
  discount : yield > coupon + 0.25
  deep disc: yield > coupon + 0.75

Measure per cell, holdout CUSIPs only (leak-free), 5s as control:
  1. model-minus-ICE signed error  -> bias where the wires live?
  2. secondary 4s-over-5s spread, maturity-matched, premium-only vs discount-only
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402

pd.set_option('display.width', 200)

snap = sorted((HERE / 'evals_archive').glob('ICE_Evals_*.csv'))[-1]
df = mm.clean_universe(mm.load_evals(snap))
bundle = mm.load_bundle(HERE / 'model.joblib')

cpn = pd.to_numeric(df['current_coupon_rate'], errors='coerce')
yld = pd.to_numeric(df['target_yield'], errors='coerce')
mat = pd.to_datetime(df['maturity_date'], errors='coerce')
tenor = (mat - pd.Timestamp('2026-08-25')).dt.days / 365.25
money = yld - cpn          # >0 discount, <0 premium

is4 = cpn.between(3.75, 4.25)
is5 = cpn == 5.0
held = df.index.isin(bundle.get('holdout_cusips', []))

# ---------------- 1. model bias by moneyness cell (holdout only) ----------------
print("1. MODEL-minus-ICE signed error by moneyness, HOLDOUT only, tenor>=4y")
sub = df[held & (is4 | is5) & (tenor >= 4)].copy()
F = mm.build_features(sub)
X = mm._prep_matrix(F, bundle['numeric'], bundle['categorical'], bundle['categories'])
pred = mm.predict_bundle(bundle, X)
err = (pred - pd.to_numeric(sub['target_yield'], errors='coerce').values) * 100
scpn = pd.to_numeric(sub['current_coupon_rate'], errors='coerce').values
smon = (pd.to_numeric(sub['target_yield'], errors='coerce')
        - pd.to_numeric(sub['current_coupon_rate'], errors='coerce')).values

cells = [('premium (<-0.25)', smon < -0.25), ('par-ish (+-0.25)', np.abs(smon) <= 0.25),
         ('discount (>+0.25)', (smon > 0.25) & (smon <= 0.75)),
         ('deep disc (>+0.75)', smon > 0.75)]
print(f"   {'cell':20s} {'4s: n':>6} {'signed':>8} {'MAE':>6}   {'5s: n':>6} {'signed':>8} {'MAE':>6}")
for tag, m in cells:
    m = m & np.isfinite(err)
    m4, m5 = m & (scpn <= 4.25) & (scpn >= 3.75), m & (scpn == 5.0)
    f4 = (f"{m4.sum():>6} {err[m4].mean():>+8.2f} {np.abs(err[m4]).mean():>6.2f}"
          if m4.sum() else f"{'0':>6} {'--':>8} {'--':>6}")
    f5 = (f"{m5.sum():>6} {err[m5].mean():>+8.2f} {np.abs(err[m5]).mean():>6.2f}"
          if m5.sum() else f"{'0':>6} {'--':>8} {'--':>6}")
    print(f"   {tag:20s} {f4}   {f5}")

# the exact wire-like cell: premium/par 4s, 5-20y
wire_like = (np.abs(smon + 0.10) <= 0.45) & (scpn >= 3.75) & (scpn <= 4.25) & np.isfinite(err)
print(f"\n   WIRE-LIKE CELL (4s within ~[-0.55,+0.35] of par yield): n={wire_like.sum()}  "
      f"signed {err[wire_like].mean():+.2f} bp  median {np.median(err[wire_like]):+.2f}  "
      f"MAE {np.abs(err[wire_like]).mean():.2f}")

# ---------------- 2. secondary spread: premium 4s vs discount 4s ----------------
print("\n2. SECONDARY 4s-over-5s spread, maturity-matched (full universe, national)")
u = df[(tenor >= 4)].copy()
u['yr'] = mat.dt.year
u['cpn'] = cpn
u['yld'] = yld
u['mon'] = money
sp_prem, sp_disc = [], []
for yr in range(2031, 2052):
    y5 = u[(u['yr'] == yr) & (u['cpn'] == 5.0)]['yld']
    p4 = u[(u['yr'] == yr) & u['cpn'].between(3.75, 4.25) & (u['mon'] < -0.10)]['yld']
    d4 = u[(u['yr'] == yr) & u['cpn'].between(3.75, 4.25) & (u['mon'] > 0.25)]['yld']
    if len(y5) >= 10:
        if len(p4) >= 3:
            sp_prem.append((p4.mean() - y5.mean()) * 100)
        if len(d4) >= 3:
            sp_disc.append((d4.mean() - y5.mean()) * 100)
print(f"   PREMIUM 4s over 5s : median {np.median(sp_prem):+6.1f} bp  ({len(sp_prem)} maturity years)")
print(f"   DISCOUNT 4s over 5s: median {np.median(sp_disc):+6.1f} bp  ({len(sp_disc)} maturity years)")
print("\n   -> if the premium-4 spread is small and the model is unbiased in the")
print("      wire-like cell, the big rich flags on new 4s are coming from the")
print("      model, and something IS wrong. If premium 4s genuinely trade wide")
print("      and the cell is unbiased, the flags are real market structure.")
